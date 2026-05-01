"""
Molecular Similarity Tool
=========================
基于摩根指纹（Morgan Fingerprint）的分子相似度与多样性计算。

Agent 调用接口
--------------
    from pcvae_tool.tools.similarity_tool import (
        tanimoto,              # 两个分子之间的相似度
        similarity_matrix,     # N×N 相似度矩阵
        most_similar,          # 从列表中找最相似的 top-N
        diversity_score,       # 列表内部平均多样性（0~1）
        novelty_score,         # 生成集对参考集的新颖性
        filter_by_similarity,  # 按相似度阈值筛选/剔除相似分子
    )

指纹参数默认值（与分子生成领域常用设定一致）：
    radius=2, nbits=2048  →  ECFP4
"""
from __future__ import annotations
from typing import Iterable, Optional

import numpy as np
import warnings
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


# ── 指纹工具 ──────────────────────────────────────────────────────────────────

def _fp(smiles: str, radius: int = 2, nbits: int = 2048):
    """计算摩根指纹；SMILES 无效时返回 None。"""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def _tanimoto_fps(fp_a, fp_b) -> float:
    if fp_a is None or fp_b is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


# ── 单对计算 ──────────────────────────────────────────────────────────────────

def tanimoto(smiles_a: str, smiles_b: str,
             radius: int = 2, nbits: int = 2048) -> dict:
    """计算两个分子的 Tanimoto 相似度（基于 ECFP4 指纹）。

    Returns
    -------
    dict
        - smiles_a, smiles_b : 输入 SMILES
        - similarity         : float [0, 1]；任一无效时返回 0.0
        - valid_a, valid_b   : bool
    """
    fp_a = _fp(smiles_a, radius, nbits)
    fp_b = _fp(smiles_b, radius, nbits)
    return {
        'smiles_a':   smiles_a,
        'smiles_b':   smiles_b,
        'similarity': round(_tanimoto_fps(fp_a, fp_b), 6),
        'valid_a':    fp_a is not None,
        'valid_b':    fp_b is not None,
    }


# ── 列表操作 ──────────────────────────────────────────────────────────────────

def most_similar(query: str, smiles_list: Iterable[str],
                 top_n: int = 5,
                 radius: int = 2, nbits: int = 2048) -> dict:
    """从 smiles_list 中找出与 query 最相似的 top_n 个分子。

    Returns
    -------
    dict
        - query      : 查询 SMILES
        - query_valid: bool
        - results    : list[dict]  按相似度降序，每项含 smiles / similarity
    """
    fp_q = _fp(query, radius, nbits)
    if fp_q is None:
        return {'query': query, 'query_valid': False, 'results': []}

    smiles_list = list(smiles_list)
    scored = []
    for smi in smiles_list:
        fp = _fp(smi, radius, nbits)
        scored.append({
            'smiles':     smi,
            'similarity': round(_tanimoto_fps(fp_q, fp), 6),
            'valid':      fp is not None,
        })

    scored.sort(key=lambda x: x['similarity'], reverse=True)
    return {
        'query':       query,
        'query_valid': True,
        'results':     scored[:top_n],
    }


def similarity_matrix(smiles_list: Iterable[str],
                      radius: int = 2, nbits: int = 2048) -> dict:
    """计算列表内所有分子两两之间的 Tanimoto 相似度矩阵。

    适合小规模列表（≤200 个），大规模列表建议用 diversity_score 代替。

    Returns
    -------
    dict
        - smiles        : list[str]  有效 SMILES（输入无效项已过滤）
        - matrix        : list[list[float]]  N×N 矩阵（行 = 列 = smiles 索引）
        - n_molecules   : int
    """
    smiles_list = list(smiles_list)
    valid_pairs = [(s, _fp(s, radius, nbits)) for s in smiles_list]
    valid_pairs = [(s, fp) for s, fp in valid_pairs if fp is not None]

    smiles_valid = [s for s, _ in valid_pairs]
    fps          = [fp for _, fp in valid_pairs]
    n = len(fps)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            sim = round(_tanimoto_fps(fps[i], fps[j]), 6)
            matrix[i][j] = sim
            matrix[j][i] = sim

    return {
        'smiles':      smiles_valid,
        'matrix':      matrix,
        'n_molecules': n,
    }


def diversity_score(smiles_list: Iterable[str],
                    radius: int = 2, nbits: int = 2048) -> dict:
    """计算列表的内部多样性（internal diversity）。

    定义：1 - 所有分子对平均 Tanimoto 相似度。
    值越接近 1 表示分子集越多样，接近 0 表示高度相似。

    Returns
    -------
    dict
        - diversity    : float [0, 1]  分子少于 2 个时返回 None
        - mean_sim     : float         平均两两相似度
        - n_valid      : int           参与计算的有效分子数
        - n_invalid    : int
    """
    smiles_list = list(smiles_list)
    fps = [(s, _fp(s, radius, nbits)) for s in smiles_list]
    valid_fps = [fp for _, fp in fps if fp is not None]
    n_valid   = len(valid_fps)
    n_invalid = len(fps) - n_valid

    if n_valid < 2:
        return {'diversity': None, 'mean_sim': None,
                'n_valid': n_valid, 'n_invalid': n_invalid}

    sims = []
    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            sims.append(_tanimoto_fps(valid_fps[i], valid_fps[j]))

    mean_sim = float(np.mean(sims))
    return {
        'diversity': round(1.0 - mean_sim, 6),
        'mean_sim':  round(mean_sim, 6),
        'n_valid':   n_valid,
        'n_invalid': n_invalid,
    }


def novelty_score(generated: Iterable[str], reference: Iterable[str],
                  threshold: float = 0.4,
                  radius: int = 2, nbits: int = 2048) -> dict:
    """计算生成集对参考集的新颖性。

    定义：生成集中与参考集任意分子最大相似度 < threshold 的比例。
    常用 threshold=0.4（ECFP4 + Tanimoto 下的常用新颖性阈值）。

    Returns
    -------
    dict
        - novelty      : float [0, 1]  越高越新颖
        - threshold    : float
        - n_novel      : int   最大相似度 < threshold 的生成分子数
        - n_not_novel  : int
        - n_gen_valid  : int   参与计算的有效生成分子数
        - details      : list[dict]  每个生成分子 {smiles, max_sim, is_novel}
    """
    gen_list = list(generated)
    ref_list = list(reference)

    ref_fps = [fp for fp in (_fp(s, radius, nbits) for s in ref_list) if fp is not None]
    if not ref_fps:
        return {'novelty': None, 'threshold': threshold,
                'n_novel': 0, 'n_not_novel': 0, 'n_gen_valid': 0, 'details': []}

    details = []
    for smi in gen_list:
        fp = _fp(smi, radius, nbits)
        if fp is None:
            continue
        max_sim = max(_tanimoto_fps(fp, r) for r in ref_fps)
        details.append({
            'smiles':   smi,
            'max_sim':  round(max_sim, 6),
            'is_novel': max_sim < threshold,
        })

    n_novel = sum(1 for d in details if d['is_novel'])
    n_gen   = len(details)

    return {
        'novelty':      round(n_novel / n_gen, 6) if n_gen else None,
        'threshold':    threshold,
        'n_novel':      n_novel,
        'n_not_novel':  n_gen - n_novel,
        'n_gen_valid':  n_gen,
        'details':      details,
    }


def filter_by_similarity(smiles_list: Iterable[str],
                         reference: str,
                         min_sim: float = 0.0,
                         max_sim: float = 1.0,
                         radius: int = 2, nbits: int = 2048) -> dict:
    """保留与参考分子相似度在 [min_sim, max_sim] 范围内的分子。

    典型用法：
    - 找"相似但不同"的分子：min_sim=0.3, max_sim=0.8
    - 过滤掉与参考分子几乎相同的分子：max_sim=0.9

    Returns
    -------
    dict
        - reference       : 参考 SMILES
        - filtered_smiles : list[str]   符合条件的 SMILES
        - details         : list[dict]  每项含 smiles / similarity / kept
        - n_kept          : int
        - n_removed       : int
    """
    smiles_list = list(smiles_list)
    fp_ref = _fp(reference, radius, nbits)
    if fp_ref is None:
        return {
            'reference':       reference,
            'filtered_smiles': [],
            'details':         [],
            'n_kept':          0,
            'n_removed':       len(smiles_list),
        }

    details = []
    for smi in smiles_list:
        fp = _fp(smi, radius, nbits)
        sim = round(_tanimoto_fps(fp_ref, fp), 6)
        kept = min_sim <= sim <= max_sim
        details.append({'smiles': smi, 'similarity': sim, 'kept': kept})

    filtered = [d['smiles'] for d in details if d['kept']]
    return {
        'reference':       reference,
        'filtered_smiles': filtered,
        'details':         details,
        'n_kept':          len(filtered),
        'n_removed':       len(smiles_list) - len(filtered),
    }
