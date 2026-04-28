"""配置加载 — 所有路径相对 pcvae_tool/ 根目录。"""
from __future__ import annotations
import os
from functools import lru_cache
import yaml

# pcvae_tool 根目录（pcvae_tool/）
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


@lru_cache(maxsize=1)
def load_config() -> dict:
    """读取 pcvae_tool/config.yaml 并把所有相对路径解析为绝对路径。

    Returns
    -------
    dict
        所有 fname_* 字段已被替换为绝对路径。
    """
    cfg_path = os.path.join(PKG_ROOT, 'config.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.full_load(f)

    for key in list(cfg.keys()):
        if key.startswith('fname_') or key == 'token_file':
            cfg[key] = os.path.join(PKG_ROOT, cfg[key])

    cfg['vae_param']['maxLength'] = cfg['maxLength']
    return cfg


def pkg_path(*parts: str) -> str:
    """返回 pcvae_tool 包内的绝对路径。"""
    return os.path.join(PKG_ROOT, *parts)
