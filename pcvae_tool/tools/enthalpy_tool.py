"""
Enthalpy Prediction Tool
========================
使用 MPNN 模型预测分子的生成焓 (kcal/mol)。

Agent 调用接口
--------------
    from pcvae_tool.tools.enthalpy_tool import predict_enthalpy, predict_enthalpies

    >>> predict_enthalpy('CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]')
    {'smiles': '...', 'enthalpy': -16.78, 'valid': True}

    >>> predict_enthalpies(['C', 'CC', 'invalid_smi'])
    {'results': [...], 'n_valid': 2, 'n_invalid': 1}
"""
from __future__ import annotations
from typing import Iterable

from ..core.enthalpy_predictor import predict_one, is_valid_smiles


def predict_enthalpy(smiles: str) -> dict:
    """预测单个 SMILES 的生成焓。

    Parameters
    ----------
    smiles : str
        分子的 SMILES 表达式。

    Returns
    -------
    dict
        - smiles   : 输入 SMILES（原样返回）
        - valid    : RDKit 是否能解析（与分子大小无关）
        - enthalpy : 预测焓值 (kcal/mol)；
                     若 RDKit 解析失败或分子过小（≤2 字符）则为 None
    """
    valid = is_valid_smiles(smiles)
    predictable = valid and len(smiles) > 2
    return {
        'smiles':   smiles,
        'valid':    valid,
        'enthalpy': predict_one(smiles) if predictable else None,
    }


def predict_enthalpies(smiles_list: Iterable[str]) -> dict:
    """批量预测一组 SMILES 的生成焓。

    Parameters
    ----------
    smiles_list : Iterable[str]
        SMILES 字符串列表。

    Returns
    -------
    dict
        - results   : list[dict]    每个元素同 predict_enthalpy 返回值
        - n_total   : int           输入总数
        - n_valid   : int           有效结构数
        - n_invalid : int           无效结构数
    """
    results = [predict_enthalpy(s) for s in smiles_list]
    n_valid = sum(1 for r in results if r['valid'])
    return {
        'results':   results,
        'n_total':   len(results),
        'n_valid':   n_valid,
        'n_invalid': len(results) - n_valid,
    }
