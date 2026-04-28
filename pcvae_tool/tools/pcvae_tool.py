"""
PCVAE Molecular Generation Tool
================================
使用预训练 PCVAE (CVAE_DHR) 生成分子，支持 4 种条件模式。

Agent 调用接口
--------------
    from pcvae_tool.tools.pcvae_tool import (
        generate_random,           # 模式 1: 完全随机
        generate_by_enthalpy,      # 模式 2: 仅焓值条件
        generate_by_smiles,        # 模式 3: 仅 SMILES 骨架条件
        generate_by_smiles_and_enthalpy,  # 模式 4: SMILES + 焓值联合
        generate_molecules,        # 统一接口
        get_enthalpy_range,        # 查询训练时焓值范围
    )

每个生成函数的返回值都是结构化字典，包含：
    - valid_smiles            : list[str]   有效分子的 SMILES
    - predicted_enthalpies    : list[float] 各分子的预测焓 (kcal/mol)
    - stats                   : dict        统计信息（含模式、有效率等）
    - output_paths            : dict|None   若 save=True 则有，记录4份输出文件路径
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch

from ..core._config import load_config, pkg_path
from ..core.enthalpy_predictor import predict_one, is_valid_smiles
from ..core.pcvae_model import PCVAE
from ..core.tokenizer import get_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 单例缓存：模型/分词器/焓值范围只在首次调用时初始化
# ─────────────────────────────────────────────────────────────────────────────

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_MODEL: Optional[PCVAE] = None
_BOUNDS: Optional[tuple] = None  # (lb, ub)


def _compute_enthalpy_bounds(csv_path: str, threshold: float) -> tuple[float, float]:
    """从训练 CSV 计算焓值归一化使用的 [lb, ub]。

    与 dataset.preprocess 保持一致：上下界向外扩展 threshold * 总跨度。
    """
    df = pd.read_csv(csv_path)
    enthalpy = np.asarray(df['heat_of_formation'])
    lo, hi = float(enthalpy.min()), float(enthalpy.max())
    diff = hi - lo
    return lo - threshold * diff, hi + threshold * diff


def _get_model_and_bounds() -> tuple[PCVAE, float, float]:
    """惰性加载 PCVAE 与焓值范围。返回 (model, lb, ub)。"""
    global _MODEL, _BOUNDS
    if _MODEL is None:
        cfg = load_config()
        _ = get_tokenizer()  # 触发 tokenizer 缓存
        m = PCVAE(
            **cfg['vae_param'],
            encoder_state_fname=cfg['fname_pcvae_encoder'],
            decoder_state_fname=cfg['fname_pcvae_decoder'],
            device=_DEVICE,
        )
        m.load()
        _MODEL = m
        _BOUNDS = _compute_enthalpy_bounds(cfg['fname_dataset'], cfg['threshold'])
    return _MODEL, _BOUNDS[0], _BOUNDS[1]


# ─────────────────────────────────────────────────────────────────────────────
# 内部辅助
# ─────────────────────────────────────────────────────────────────────────────

def _encode_smiles(smiles: str, tokenizer, maxLength: int, pad_idx: int,
                   nSample: int) -> torch.Tensor:
    token_vec = tokenizer.tokenize([smiles], useTokenDict=True)[0]
    num_vec = tokenizer.getNumVector([token_vec], addStart=True, addEnd=True)[0]
    if max(num_vec) >= tokenizer.getTokensSize():
        raise ValueError(
            f'Input SMILES contains tokens outside the vocabulary '
            f'(max index {max(num_vec)} >= vocab size {tokenizer.getTokensSize()}).')

    if len(num_vec) > maxLength:
        num_vec = [num_vec[0]] + num_vec[1:-1][:maxLength - 2] + [num_vec[-1]]
    else:
        num_vec = num_vec + [pad_idx] * (maxLength - len(num_vec))

    return (torch.tensor(num_vec, dtype=torch.long, device=_DEVICE)
            .unsqueeze(0).expand(nSample, -1).contiguous())


def _fmt_float_safe(val: float) -> str:
    """文件名友好的浮点表示：-50.8 -> 'neg50p8'，52.0 -> '52p0'。"""
    sign = 'neg' if val < 0 else ''
    int_part, dec_part = f'{abs(val):.1f}'.split('.')
    return f'{sign}{int_part}p{dec_part}'


def _mode_tag(smiles: Optional[str], enthalpy: Optional[float]) -> str:
    has_smi = bool(smiles)
    has_ent = enthalpy is not None
    if not has_smi and not has_ent:
        return 'rand'
    if has_smi and has_ent:
        return f'smi_h{_fmt_float_safe(enthalpy)}'
    if has_smi:
        return 'smi'
    return f'h{_fmt_float_safe(enthalpy)}'


def _build_output_paths(output_dir: str, mode: str, num_samples: int) -> dict:
    """生成按日期分组的输出路径，文件名仅含 [a-zA-Z0-9_]。"""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    date_str, time_str = ts[:8], ts[9:]
    dated = os.path.join(output_dir, date_str)
    os.makedirs(dated, exist_ok=True)
    base = f'{mode}_n{num_samples}_{time_str}'
    return {
        'smiles':   os.path.abspath(os.path.join(dated, f'{base}_smiles.smi')),
        'info':     os.path.abspath(os.path.join(dated, f'{base}_info.txt')),
        'enthalpy': os.path.abspath(os.path.join(dated, f'{base}_enthalpy.txt')),
        'summary':  os.path.abspath(os.path.join(dated, f'{base}_summary.json')),
    }


def _write_outputs(paths: dict, valid_smiles: list, pred_enthalpies: list,
                   stats: dict):
    with open(paths['smiles'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_smiles) + ('\n' if valid_smiles else ''))
    with open(paths['info'], 'w', encoding='utf-8') as f:
        for smi, ent in zip(valid_smiles, pred_enthalpies):
            f.write(f'{smi},{ent:.4f}\n')
    with open(paths['enthalpy'], 'w', encoding='utf-8') as f:
        for ent in pred_enthalpies:
            f.write(f'{ent:.4f}\n')
    with open(paths['summary'], 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# 核心采样逻辑（4 种模式共用）
# ─────────────────────────────────────────────────────────────────────────────

def _sample(smiles: Optional[str], enthalpy: Optional[float],
            num_samples: int) -> tuple[list, list]:
    """根据条件采样并返回 (valid_smiles, predicted_enthalpies)。"""
    cfg = load_config()
    model, lb, ub = _get_model_and_bounds()
    tokenizer = get_tokenizer()

    maxLength = cfg['maxLength']
    latent_dim = cfg['vae_param']['latent_dim']
    pad_idx = tokenizer.getTokensNum('<pad>')
    alpha = model.encoder.alpha

    has_smi = bool(smiles)
    has_ent = enthalpy is not None

    with torch.no_grad():
        # ── 1) 完全随机 ───────────────────────────────────────────────────────
        if not has_smi and not has_ent:
            mu_n = torch.randn((num_samples, latent_dim), device=_DEVICE)
            norm_h = torch.rand(num_samples, device=_DEVICE)
            X = torch.zeros((num_samples, maxLength), dtype=torch.long, device=_DEVICE)

        else:
            # 结构后验
            if has_smi:
                X = _encode_smiles(smiles, tokenizer, maxLength, pad_idx, num_samples)
                _, mu_x, logvar_x, _, _ = model.encoder(
                    X, torch.zeros(1, device=_DEVICE), alpha=1)
            else:
                X = torch.zeros((num_samples, maxLength), dtype=torch.long, device=_DEVICE)
                mu_x = torch.randn((num_samples, latent_dim), device=_DEVICE)
                logvar_x = torch.zeros((num_samples, latent_dim), device=_DEVICE)

            # 焓先验
            if has_ent:
                norm_h_val = (enthalpy - lb) / (ub - lb)
                if not (0.0 <= norm_h_val <= 1.0):
                    raise ValueError(
                        f'Target enthalpy {enthalpy:.2f} kcal/mol is outside '
                        f'training range [{lb:.2f}, {ub:.2f}].')
                h_t = torch.tensor([norm_h_val], dtype=torch.float32, device=_DEVICE)
                mu_p, lv_p = model.encoder.prior_block(h_t.unsqueeze(1))
                mu_prior = mu_p.expand(num_samples, -1).contiguous()
                logvar_prior = lv_p.expand(num_samples, -1).contiguous()
                norm_h = h_t.expand(num_samples).contiguous()
            else:
                norm_h = torch.rand(num_samples, device=_DEVICE)
                mu_prior, logvar_prior = model.encoder.prior_block(norm_h.unsqueeze(1))

            mu = alpha * mu_x + (1 - alpha) * mu_prior
            logvar = 0.5 * (alpha * logvar_x + (1 - alpha) * logvar_prior)
            mu_n = model.encoder.reparameterize(mu, logvar)

        y = model.decoder(mu_n, norm_h, X, freerun=True).cpu()
        smiles_all = tokenizer.getSmiles(y)

    valid = [s for s in smiles_all if is_valid_smiles(s)]
    enthalpies = [predict_one(s) for s in valid]
    return valid, enthalpies


def _run_pipeline(smiles: Optional[str], enthalpy: Optional[float],
                  num_samples: int, save: bool, output_dir: str) -> dict:
    """采样 + 统计 + （可选）写文件。"""
    valid_smiles, pred_enthalpies = _sample(smiles, enthalpy, num_samples)
    mode = _mode_tag(smiles, enthalpy)

    stats = {
        'mode':              mode,
        'smiles_input':      smiles or None,
        'enthalpy_target':   enthalpy,
        'n_attempted':       num_samples,
        'n_valid':           len(valid_smiles),
        'valid_rate':        round(len(valid_smiles) / num_samples, 4) if num_samples else 0.0,
        'device':            str(_DEVICE),
        'timestamp':         datetime.now().strftime('%Y%m%d_%H%M%S'),
    }

    output_paths = None
    if save:
        output_paths = _build_output_paths(output_dir, mode, num_samples)
        stats['output_paths'] = output_paths
        _write_outputs(output_paths, valid_smiles, pred_enthalpies, stats)

    return {
        'valid_smiles':         valid_smiles,
        'predicted_enthalpies': pred_enthalpies,
        'stats':                stats,
        'output_paths':         output_paths,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent Tool 接口
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = pkg_path('outputs')


def get_enthalpy_range() -> dict:
    """查询训练数据决定的焓值合法范围（kcal/mol）。

    Returns
    -------
    dict with keys: lower_bound, upper_bound, unit
    """
    _, lb, ub = _get_model_and_bounds()
    return {
        'lower_bound': float(lb),
        'upper_bound': float(ub),
        'unit':        'kcal/mol',
    }


def generate_random(num_samples: int = 100,
                    save: bool = True,
                    output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    """模式 1：完全随机生成（无任何条件）。"""
    return _run_pipeline(None, None, num_samples, save, output_dir)


def generate_by_enthalpy(enthalpy: float,
                         num_samples: int = 100,
                         save: bool = True,
                         output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    """模式 2：仅以焓值为条件生成。

    Parameters
    ----------
    enthalpy : float
        目标生成焓 (kcal/mol)。必须在 get_enthalpy_range() 给出的范围内。
    """
    return _run_pipeline(None, enthalpy, num_samples, save, output_dir)


def generate_by_smiles(smiles: str,
                       num_samples: int = 100,
                       save: bool = True,
                       output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    """模式 3：仅以 SMILES 骨架为条件生成（焓值随机采样）。"""
    if not smiles:
        raise ValueError('smiles must be a non-empty string.')
    return _run_pipeline(smiles, None, num_samples, save, output_dir)


def generate_by_smiles_and_enthalpy(smiles: str,
                                    enthalpy: float,
                                    num_samples: int = 100,
                                    save: bool = True,
                                    output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    """模式 4：SMILES 骨架 + 目标焓值联合条件生成。"""
    if not smiles:
        raise ValueError('smiles must be a non-empty string.')
    return _run_pipeline(smiles, enthalpy, num_samples, save, output_dir)


def generate_molecules(smiles: Optional[str] = None,
                       enthalpy: Optional[float] = None,
                       num_samples: int = 100,
                       save: bool = True,
                       output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    """统一入口：根据传入的 smiles/enthalpy 是否为 None 自动选择 4 种模式之一。"""
    return _run_pipeline(smiles or None, enthalpy, num_samples, save, output_dir)
