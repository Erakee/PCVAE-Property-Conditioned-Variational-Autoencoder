"""
CVAE_DHR Molecular Generation
==============================
既可以作为 CLI 直接运行，也可以作为 Python 模块被 agent 调用：

    from generate_dhr import generate_molecules
    result = generate_molecules(enthalpy=-50.0, num_samples=100)
    print(result['stats'])
    print(result['valid_smiles'][:5])
"""

import argparse
import json
import os
import torch
import model.CVAE_DHR as cvae
from dataset.dataset import SmilesDictDataset
from util.config_loader import load_training_config
from util.checkpoint_paths import (
    attach_checkpoint_weights,
    default_generation_output_dir,
    resolve_generation_checkpoint,
)
from util.enthalpy_predictor import predict_enthalpy
import util.utils as utils
from datetime import datetime

GEN_MODEL_KEY = 'cvae_dhr'


# ── 文件名辅助 ────────────────────────────────────────────────────────────────

def _fmt_float(val: float) -> str:
    """将浮点数转换为文件名安全字符串（无 '-' 和 '.'）。
    示例: -50.8 -> 'neg50p8'  |  52.0 -> '52p0'
    """
    sign = 'neg' if val < 0 else ''
    int_part, dec_part = f'{abs(val):.1f}'.split('.')
    return f'{sign}{int_part}p{dec_part}'


def _build_output_paths(mode: str, num_samples: int,
                        timestamp: str, output_dir: str) -> dict:
    """根据生成模式构造文件名，仅含字母、数字和下划线。

    目录结构:
      {output_dir}/
        {YYYYMMDD}/               ← 按日期自动分组
          {mode}_n{N}_{HHMMSS}_{type}.{ext}

    mode 取值:
      rand            — 完全随机
      h{val}          — 仅焓值条件
      smi             — 仅 SMILES 条件
      smi_h{val}      — SMILES + 焓值联合条件
    """
    date_str  = timestamp[:8]          # 'YYYYMMDD'
    time_str  = timestamp[9:]          # 'HHMMSS'
    dated_dir = os.path.join(output_dir, date_str)
    os.makedirs(dated_dir, exist_ok=True)
    base = f'{mode}_n{num_samples}_{time_str}'
    return {
        'smiles':   os.path.join(dated_dir, f'{base}_smiles.smi'),
        'info':     os.path.join(dated_dir, f'{base}_info.txt'),
        'enthalpy': os.path.join(dated_dir, f'{base}_enthalpy.txt'),
        'summary':  os.path.join(dated_dir, f'{base}_summary.json'),
    }


def _mode_tag(smiles: str, enthalpy) -> str:
    """根据条件类型返回模式标签字符串（文件名安全）。"""
    has_smiles   = len(smiles) > 0
    has_enthalpy = enthalpy is not None
    if not has_smiles and not has_enthalpy:
        return 'rand'
    elif has_smiles and has_enthalpy:
        return f'smi_h{_fmt_float(enthalpy)}'
    elif has_smiles:
        return 'smi'
    else:
        return f'h{_fmt_float(enthalpy)}'


# ── 模型与推理辅助 ────────────────────────────────────────────────────────────

def _load_model(cfg: dict, device):
    """加载并返回已评估模式的 ConVAE 模型（权重路径来自 cfg）。"""
    model = cvae.ConVAE(
        **cfg['vae_param'],
        encoder_state_fname=cfg['fname_vae_encoder_parameters'],
        decoder_state_fname=cfg['fname_vae_decoder_parameters'],
        device=device,
    )
    model.encoder.loadState()
    model.decoder.loadState()
    model.encoder.eval()
    model.decoder.eval()
    return model


def _encode_smiles(smiles: str, tokenizer, maxLength: int, pad_idx: int,
                   nSample: int, device):
    """将单条 SMILES 编码为 token 索引张量并扩展到 nSample。"""
    token_vec = tokenizer.tokenize([smiles], useTokenDict=True)[0]
    num_vec   = tokenizer.getNumVector([token_vec], addStart=True, addEnd=True)[0]

    if max(num_vec) >= tokenizer.getTokensSize():
        print(f'[Warning] Input SMILES contains unknown tokens '
              f'(max index {max(num_vec)} >= vocab size {tokenizer.getTokensSize()}).')

    if len(num_vec) > maxLength:
        num_vec = [num_vec[0]] + num_vec[1:-1][:maxLength - 2] + [num_vec[-1]]
    else:
        num_vec = num_vec + [pad_idx] * (maxLength - len(num_vec))

    return (torch.tensor(num_vec, dtype=torch.long, device=device)
            .unsqueeze(0).expand(nSample, -1).contiguous())


# ── 核心生成函数（可供 agent 直接调用）────────────────────────────────────────

def generate_molecules(
    smiles: str = '',
    enthalpy: float = None,
    num_samples: int = 100,
    output_dir: str = None,
    *,
    checkpoint_dir: str = None,
    config_path: str = None,
) -> dict:
    """生成分子并返回结构化结果，同时将文件写入 output_dir。

    Parameters
    ----------
    smiles : str
        条件 SMILES 字符串，留空表示不使用结构条件。
    enthalpy : float or None
        目标生成焓（kcal/mol）。None 表示不使用焓值条件。
    num_samples : int
        尝试生成的样本数量。
    output_dir : str or None
        输出根路径；默认来自 ``config.yaml`` 的 ``generation_output_root/cvae_dhr``。
    checkpoint_dir : str or None
        含 ``encoder.pt`` / ``decoder.pt`` 的目录，或 ``checkpoint_aliases`` 中的别名。
    config_path : str or None
        YAML 路径；默认仓库根目录 ``config.yaml``。

    Returns
    -------
    dict with keys:
        valid_smiles         : list[str]   — 有效 SMILES 列表
        predicted_enthalpies : list[float] — 对应的预测焓值（kcal/mol）
        output_paths         : dict        — 各输出文件的绝对路径
        stats                : dict        — 统计信息（可直接被 agent 读取）
    """
    cfg = load_training_config(config_path, model='cvae_dhr')
    ckpt = resolve_generation_checkpoint(cfg, GEN_MODEL_KEY, checkpoint_dir)
    cfg = attach_checkpoint_weights(cfg, ckpt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = utils.get_tokenizer(model='cvae_dhr')
    maxLength  = cfg['maxLength']
    pad_idx    = tokenizer.getTokensNum('<pad>')
    latent_dim = cfg['vae_param']['latent_dim']

    dataset = SmilesDictDataset(cfg['fname_dataset'], tokenizer, maxLength)
    lb, ub  = dataset._getbound()

    if output_dir is None:
        output_dir = default_generation_output_dir(cfg, GEN_MODEL_KEY)

    vae_model = _load_model(cfg, device)
    alpha     = vae_model.encoder.alpha

    has_smiles   = len(smiles) > 0
    has_enthalpy = enthalpy is not None

    # ── 构造潜变量与焓条件向量 ────────────────────────────────────────────────
    with torch.no_grad():
        if not has_smiles and not has_enthalpy:
            mu_n     = torch.randn((num_samples, latent_dim), device=device)
            norm_n_h = torch.rand(num_samples, device=device)
            X        = torch.zeros((num_samples, maxLength), dtype=torch.long, device=device)

        else:
            # 构造焓值条件向量
            if has_enthalpy:
                norm_h = (enthalpy - lb) / (ub - lb)
                if not (0 <= norm_h <= 1):
                    raise ValueError(
                        f'Enthalpy {enthalpy:.2f} is outside training range '
                        f'[{lb:.2f}, {ub:.2f}] kcal/mol.')
                h_cond = torch.full((num_samples,), norm_h,
                                    dtype=torch.float32, device=device)
            else:
                h_cond = torch.rand(num_samples, device=device)

            if has_smiles:
                X = _encode_smiles(smiles, tokenizer, maxLength, pad_idx,
                                   num_samples, device)
            else:
                X = torch.zeros((num_samples, maxLength),
                                dtype=torch.long, device=device)

            # 从 prior 分布采样（而非用 posterior 均值），
            # 因为训练时 encoder 存在后验坍缩（mu≈0），直接用 mu
            # 会导致所有样本相同、条件控制失效。prior 的 mu≈0、std≈1，
            # 采样后每个样本有不同的 N(0,1)-scale 向量，与 decoder
            # 训练时的输入分布一致，且焓值条件通过 h_cond 传递给 decoder。
            mu_prior, logvar_prior = vae_model.encoder.prior_block(h_cond.unsqueeze(1))
            std_prior = torch.exp(0.5 * logvar_prior)
            eps = torch.randn((num_samples, latent_dim), device=device)
            mu_n     = mu_prior + eps * std_prior
            norm_n_h = h_cond

        # ── 解码并验证 ────────────────────────────────────────────────────────
        y            = vae_model.decoder(mu_n, norm_n_h, X, freerun=True).cpu()
        smiles_all   = tokenizer.getSmiles(y)
        valid_smiles = [sm for sm in smiles_all if utils.isValidSmiles(sm)]

    pred_enthalpies = [predict_enthalpy(sm) for sm in valid_smiles]

    # ── 构造输出路径并写文件 ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode      = _mode_tag(smiles, enthalpy)
    paths     = _build_output_paths(mode, num_samples, timestamp, output_dir)
    paths     = {k: os.path.abspath(v) for k, v in paths.items()}

    stats = {
        'timestamp':          timestamp,
        'mode':               mode,
        'smiles_input':       smiles or None,
        'enthalpy_target':    enthalpy,
        'n_attempted':        num_samples,
        'n_valid':            len(valid_smiles),
        'valid_rate':         round(len(valid_smiles) / num_samples, 4) if num_samples else 0.0,
        'device':             str(device),
        'output_paths':       paths,
    }

    _write_outputs(paths, valid_smiles, pred_enthalpies, stats)

    return {
        'valid_smiles':         valid_smiles,
        'predicted_enthalpies': pred_enthalpies,
        'output_paths':         paths,
        'stats':                stats,
    }


def _write_outputs(paths: dict, valid_smiles: list,
                   pred_enthalpies: list, stats: dict):
    """将三份数据文件和一份 JSON summary 写入磁盘。"""
    # smiles 文件：每行一个有效 SMILES
    with open(paths['smiles'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_smiles) + ('\n' if valid_smiles else ''))

    # info 文件：每行 "SMILES,predicted_enthalpy"
    with open(paths['info'], 'w', encoding='utf-8') as f:
        for smi, ent in zip(valid_smiles, pred_enthalpies):
            f.write(f'{smi},{ent:.4f}\n')

    # enthalpy 文件：每行一个预测焓值
    with open(paths['enthalpy'], 'w', encoding='utf-8') as f:
        for ent in pred_enthalpies:
            f.write(f'{ent:.4f}\n')

    # summary JSON：结构化元数据，供 agent 直接解析
    with open(paths['summary'], 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    valid_rate_pct = stats['valid_rate'] * 100
    print(f"[{stats['timestamp']}] mode={stats['mode']}  "
          f"valid={stats['n_valid']}/{stats['n_attempted']} ({valid_rate_pct:.1f}%)")
    for key, path in paths.items():
        print(f"  {key:<10} -> {path}")


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--smiles',      type=str,   default='',
                        help='Scaffold SMILES for structure-conditioned generation')
    parser.add_argument('--enthalpy',    type=float, default=None,
                        help='Target enthalpy (kcal/mol) for enthalpy-conditioned generation')
    parser.add_argument('--num_samples', type=int,   default=100,
                        help='Number of molecules to attempt (default: 100)')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config (default: <repo>/config.yaml)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint dir, path under root_path, or checkpoint_aliases key')
    parser.add_argument('--output_dir',  type=str,   default=None,
                        help='Output root (default: generation_output_root/cvae_dhr from config)')
    args = parser.parse_args()

    generate_molecules(
        smiles=args.smiles,
        enthalpy=args.enthalpy,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint,
        config_path=args.config,
    )


if __name__ == '__main__':
    main()

# ── 常用示例（直接复制到命令行）──────────────────────────────────────────────
# 完全随机:
#   python generate_dhr.py --num_samples 500
# 仅焓值条件:
#   python generate_dhr.py --enthalpy -50.0 --num_samples 200
# RDX (SMILES + 焓值):
#   python generate_dhr.py --smiles "C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]" --enthalpy 52.8
# TNT:
#   python generate_dhr.py --smiles "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]" --enthalpy -16.01
# TNB:
#   python generate_dhr.py --smiles "C1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]" --enthalpy -8.89
# TATB:
#   python generate_dhr.py --smiles "C1(=C(C(=C(C(=C1[N+](=O)[O-])N)[N+](=O)[O-])N)[N+](=O)[O-])N" --enthalpy -36.78
# NTO:
#   python generate_dhr.py --smiles "C1(=NC(=O)NN1)[N+](=O)[O-]" --enthalpy -100
