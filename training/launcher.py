"""
Unified training launcher for all VAE / CVAE variants in this repo.

Use the CLI: ``python train.py --model <name>`` from the project root.
"""
from __future__ import annotations

import importlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    'cvae_dhr': {
        'module': 'model.CVAE_DHR',
        'class_name': 'ConVAE',
        'config_model_arg': 'cvae_dhr',
        'dataset': 'indices',
        'tokenizer_mode': 'full',
        'scheduler_step_gamma': (15, 0.9),
        'output_subdir': 'CVAE_DHR',
    },
    'cvae_hc': {
        'module': 'model.CVAE_HC_SEED',
        'class_name': 'ConVAE',
        'config_model_arg': 'cvae_hc',
        'dataset': 'onehot',
        'tokenizer_mode': 'minus2',
        'scheduler_step_gamma': (5, 0.95),
        'output_subdir': 'CVAE_HC',
    },
    'vae_h': {
        'module': 'model.VAE_H_SEED',
        'class_name': 'VAE',
        'config_model_arg': 'vae_h',
        'dataset': 'onehot',
        'tokenizer_mode': 'minus2',
        'scheduler_step_gamma': (5, 0.95),
        'output_subdir': 'VAE_H',
    },
}


def list_models() -> str:
    lines = ['Registered models:']
    for name, meta in MODEL_REGISTRY.items():
        lines.append(
            f"  {name:10}  dataset={meta['dataset']:7}  tokenizer={meta['tokenizer_mode']}"
        )
    return '\n'.join(lines)


def set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _get_tokenizer(model_key: str):
    import util.utils as utils

    ma = MODEL_REGISTRY[model_key]['config_model_arg']
    if MODEL_REGISTRY[model_key]['tokenizer_mode'] == 'full':
        return utils.get_tokenizer(model=ma)
    return utils.ori_get_tokenizer(model=ma)


def _build_dataloader(model_key: str, cfg: dict, tokenizer, seed: int, num_workers: int):
    import torch
    from dataset.dataset import SmilesDataset, SmilesDictDataset

    meta = MODEL_REGISTRY[model_key]
    gen = torch.Generator().manual_seed(seed)
    path = cfg['fname_dataset']
    max_len = cfg['maxLength']

    if meta['dataset'] == 'indices':
        ds = SmilesDictDataset(path, tokenizer, max_len)
        collate = SmilesDictDataset.collate_fn
    else:
        ds = SmilesDataset(path, tokenizer, max_len)
        collate = ds.collate_fn

    return torch.utils.data.DataLoader(
        ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate,
        generator=gen,
    ), ds


def _build_model(model_key: str, cfg: dict, device: 'torch.device',
                 encoder_path: str, decoder_path: str):
    meta = MODEL_REGISTRY[model_key]
    mod = importlib.import_module(meta['module'])
    cls = getattr(mod, meta['class_name'])
    return cls(
        **cfg['vae_param'],
        encoder_state_fname=encoder_path,
        decoder_state_fname=decoder_path,
        device=device,
    )


def _build_optimizers(model_obj, cfg: dict, model_key: str):
    import torch

    enc_opt = torch.optim.AdamW(
        model_obj.encoder.parameters(),
        lr=cfg['lr'],
        weight_decay=1e-6,
        eps=1e-9,
    )
    dec_opt = torch.optim.Adam(
        model_obj.decoder.parameters(),
        lr=cfg['lr'],
        weight_decay=1e-6,
        eps=1e-9,
    )
    step, gamma = MODEL_REGISTRY[model_key]['scheduler_step_gamma']
    enc_sched = torch.optim.lr_scheduler.StepLR(enc_opt, step_size=step, gamma=gamma)
    dec_sched = torch.optim.lr_scheduler.StepLR(dec_opt, step_size=step, gamma=gamma)
    return enc_opt, dec_opt, enc_sched, dec_sched


def resolve_exp_dir(
    exp_root: str,
    model_key: str,
    seed: int,
    tag: Optional[str],
) -> Path:
    sub = MODEL_REGISTRY[model_key]['output_subdir']
    base = Path(exp_root) / sub
    if tag:
        base = base / tag
    return base / f'seed_{seed}'


def run_training(
    model_key: str,
    seed: int,
    *,
    config_path: Optional[str] = None,
    exp_root: Optional[str] = None,
    tag: Optional[str] = None,
    num_workers: int = 4,
    print_interval: int = 50,
    kld_alpha: float = 1.0,
    lr: Optional[float] = None,
    num_epoch: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    dry_run: bool = False,
) -> Optional[Path]:
    """
    Run one training job. Checkpoints and logs go under
    ``{exp_root}/{MODEL}/{tag}/seed_{seed}`` (``tag`` omitted if None).

    Returns experiment directory path, or None if ``dry_run``.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model {model_key!r}. {list_models()}')

    from util.config_loader import load_training_config, resolve_config_path

    meta = MODEL_REGISTRY[model_key]
    cfg = load_training_config(config_path, model=meta['config_model_arg'])
    effective_exp_root = exp_root if exp_root is not None else cfg['training_runs_root']

    if lr is not None:
        cfg['lr'] = lr
    if num_epoch is not None:
        cfg['num_epoch'] = num_epoch
    if batch_size is not None:
        cfg['batch_size'] = batch_size

    exp_dir = resolve_exp_dir(effective_exp_root, model_key, seed, tag)
    exp_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        disp_dev = device if device else 'auto (cuda if available)'
        print(list_models())
        print(f'Config file: {resolve_config_path(config_path)}')
        print(f'Would train {model_key} with seed={seed} on {disp_dev}')
        print(f'Training runs root (effective): {effective_exp_root}')
        print(f'Output directory: {exp_dir.resolve()}')
        print(f'lr={cfg["lr"]}, num_epoch={cfg["num_epoch"]}, batch_size={cfg["batch_size"]}')
        return None

    import torch

    dev = torch.device(
        device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    manifest = {
        'model_key': model_key,
        'seed': seed,
        'tag': tag,
        'exp_root': effective_exp_root,
        'config_resolved': resolve_config_path(config_path),
        'device': str(dev),
        'lr': cfg['lr'],
        'num_epoch': cfg['num_epoch'],
        'batch_size': cfg['batch_size'],
        'kld_alpha': kld_alpha,
    }
    with open(exp_dir / 'run_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    from util.utils import temp_seed

    set_seed(seed)

    with temp_seed(seed):
        tokenizer = _get_tokenizer(model_key)
        dataloader, dataset = _build_dataloader(
            model_key, cfg, tokenizer, seed, num_workers
        )
        lb, ub = dataset._getbound()

        model_obj = _build_model(
            model_key,
            cfg,
            dev,
            str(exp_dir / 'encoder.pt'),
            str(exp_dir / 'decoder.pt'),
        )

    enc_opt, dec_opt, enc_sched, dec_sched = _build_optimizers(
        model_obj, cfg, model_key
    )

    model_obj.trainModel(
        dataloader,
        enc_opt,
        dec_opt,
        enc_sched,
        dec_sched,
        kld_alpha,
        cfg['num_epoch'],
        tokenizer,
        print_interval,
        lb,
        ub,
        seed,
        log_dir=str(exp_dir),
    )
    return exp_dir
