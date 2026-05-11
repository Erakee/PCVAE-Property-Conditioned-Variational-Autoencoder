"""
YAML training configuration — no PyTorch import.

Used by ``train.py --dry-run`` and by ``util.utils`` after Torch is available.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def project_root() -> Path:
    """Directory containing the repo (parent of ``util/``)."""
    return Path(__file__).resolve().parent.parent


def default_config_path() -> Path:
    return project_root() / 'config.yaml'


def resolve_config_path(fyaml: Optional[str] = None) -> str:
    if fyaml is None:
        return str(default_config_path())
    return os.path.abspath(fyaml)


def load_training_config(fyaml: Optional[str] = None, model: str = 'vae_h') -> Dict[str, Any]:
    """
    Load ``config.yaml``-style dict with absolute paths joined to ``root_path``.

    ``root_path`` in YAML may be relative to the YAML file directory.
    """
    fyaml = resolve_config_path(fyaml)
    with open(fyaml, 'r', encoding='utf-8') as f:
        config = yaml.full_load(f)
    cfg_dir = os.path.dirname(fyaml)
    rp = config.get('root_path', '.')
    if not os.path.isabs(rp):
        config['root_path'] = os.path.abspath(os.path.join(cfg_dir, rp))
    else:
        config['root_path'] = os.path.abspath(rp)
    config['fname_dataset'] = os.path.join(config['root_path'], config['fname_dataset'])
    config['token_file'] = os.path.join(config['root_path'], config['token_file'])
    config['fname_fps'] = os.path.join(config['root_path'], config['fname_fps'])
    config['fname_tokenizer'] = os.path.join(config['root_path'], config['fname_tokenizer'])
    config['sampled_dir'] = os.path.join(config['root_path'], config['sampled_dir'])
    config['vae_param']['maxLength'] = config['maxLength']
    if model == 'vae_h':
        config['fname_vae_encoder_parameters'] = os.path.join(
            config['root_path'], config['fname_enc_params_VAE_H'])
        config['fname_vae_decoder_parameters'] = os.path.join(
            config['root_path'], config['fname_dec_params_VAE_H'])
    elif model == 'cvae_hc':
        config['fname_vae_encoder_parameters'] = os.path.join(
            config['root_path'], config['fname_enc_params_CVAE_HC'])
        config['fname_vae_decoder_parameters'] = os.path.join(
            config['root_path'], config['fname_dec_params_CVAE_HC'])
    elif model == 'cvae_dhr':
        config['fname_vae_encoder_parameters'] = os.path.join(
            config['root_path'], config['fname_enc_params_CVAE_DHR'])
        config['fname_vae_decoder_parameters'] = os.path.join(
            config['root_path'], config['fname_dec_params_CVAE_DHR'])

    _apply_path_layout_env(config)

    return config


def _resolve_under_root(root: str, path_str: str) -> str:
    path_str = (path_str or '').strip()
    if not path_str:
        return path_str
    if os.path.isabs(path_str):
        return os.path.abspath(path_str)
    return os.path.abspath(os.path.join(root, path_str))


def _apply_path_layout_env(config: Dict[str, Any]) -> None:
    """
    Set ``training_runs_root``, ``generation_output_root``, and resolve checkpoint maps.

    Optional environment overrides (absolute paths recommended):

    - ``EVAE_TRAINING_RUNS_ROOT`` — default root for ``train.py`` outputs
    - ``EVAE_GENERATION_OUTPUT_ROOT`` — base folder for generation scripts
    """
    root = config['root_path']

    tr = os.environ.get('EVAE_TRAINING_RUNS_ROOT')
    if tr:
        config['training_runs_root'] = os.path.abspath(tr)
    else:
        tr_rel = config.get('training_runs_root', 'training_params')
        config['training_runs_root'] = _resolve_under_root(root, tr_rel)

    gen = os.environ.get('EVAE_GENERATION_OUTPUT_ROOT')
    if gen:
        config['generation_output_root'] = os.path.abspath(gen)
    else:
        gen_rel = config.get('generation_output_root', 'generations')
        config['generation_output_root'] = _resolve_under_root(root, gen_rel)

    aliases_in = config.get('checkpoint_aliases') or {}
    config['checkpoint_aliases'] = {
        name: _resolve_under_root(root, p)
        for name, p in aliases_in.items()
        if p
    }

    defaults_in = config.get('default_generation_checkpoint') or {}
    config['default_generation_checkpoint'] = {
        k: _resolve_under_root(root, v)
        for k, v in defaults_in.items()
        if v
    }
