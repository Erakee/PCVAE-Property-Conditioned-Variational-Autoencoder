"""
Resolve checkpoint directories (folders containing ``encoder.pt`` + ``decoder.pt``)
for training outputs and generation.

Paths come from ``config.yaml`` (``default_generation_checkpoint``, ``checkpoint_aliases``)
and optional env overrides loaded by :mod:`util.config_loader`.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _abs_under_root(root: str, path: str) -> str:
    path = path.strip()
    if not path:
        raise ValueError('Empty path')
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(root, path))


def resolve_generation_checkpoint(
    cfg: Dict[str, Any],
    model_key: str,
    override: Optional[str] = None,
) -> str:
    """
    Return absolute directory that contains ``encoder.pt`` and ``decoder.pt``.

    Parameters
    ----------
    cfg :
        Output of :func:`util.config_loader.load_training_config` for the matching ``model``.
    model_key :
        One of ``cvae_dhr``, ``cvae_hc``, ``vae_h``.
    override :
        Absolute directory, path relative to ``root_path``, or a key in ``checkpoint_aliases``.
    """
    root = cfg['root_path']
    aliases = cfg.get('checkpoint_aliases') or {}

    if override:
        o = override.strip()
        if o in aliases:
            return os.path.abspath(aliases[o])
        if os.path.isdir(o):
            return os.path.abspath(o)
        return _abs_under_root(root, o)

    defaults = cfg.get('default_generation_checkpoint') or {}
    if model_key in defaults and defaults[model_key]:
        return _abs_under_root(root, defaults[model_key])

    enc = cfg.get('fname_vae_encoder_parameters')
    if enc:
        return os.path.abspath(os.path.dirname(enc))

    raise ValueError(
        f'No checkpoint directory for model {model_key!r}: set '
        f'default_generation_checkpoint.{model_key} in config.yaml, '
        f'or pass --checkpoint / checkpoint_dir=.'
    )


def attach_checkpoint_weights(cfg: Dict[str, Any], checkpoint_dir: str) -> Dict[str, Any]:
    """Shallow-copy ``cfg`` and point encoder/decoder state paths at ``checkpoint_dir``."""
    out = dict(cfg)
    d = os.path.abspath(checkpoint_dir)
    out['fname_vae_encoder_parameters'] = os.path.join(d, 'encoder.pt')
    out['fname_vae_decoder_parameters'] = os.path.join(d, 'decoder.pt')
    return out


def default_generation_output_dir(cfg: Dict[str, Any], model_key: str) -> str:
    """``generation_output_root`` + model subfolder (for organizing outputs by architecture)."""
    base = cfg.get('generation_output_root')
    if not base:
        base = os.path.join(cfg['root_path'], 'generations')
    safe = {'cvae_dhr': 'cvae_dhr', 'cvae_hc': 'cvae_hc', 'vae_h': 'vae_h',
            'cvae_film': 'cvae_film'}.get(
        model_key, model_key
    )
    return os.path.join(base, safe)
