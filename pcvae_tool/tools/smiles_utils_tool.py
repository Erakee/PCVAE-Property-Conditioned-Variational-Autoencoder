"""
SMILES Utilities Tool
=====================
SMILES 有效性验证、规范化与分子基础属性计算。

所有函数纯 RDKit 操作，无模型推理，调用速度极快。

Agent 调用接口
--------------
    from pcvae_tool.tools.smiles_utils_tool import (
        validate,           # 检查单个 SMILES
        canonicalize,       # 返回规范 SMILES
        get_properties,     # 获取单分子属性
        batch_validate,     # 批量验证
        batch_canonicalize, # 批量规范化
        batch_properties,   # 批量属性
        deduplicate,        # 列表去重（按规范 SMILES）
    )

返回值均为普通 dict / list，JSON 可序列化。
"""
from __future__ import annotations
from typing import Iterable, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


# ── 单分子操作 ────────────────────────────────────────────────────────────────

def validate(smiles: str) -> dict:
    """验证单个 SMILES 是否合法。

    Returns
    -------
    dict
        - smiles  : 原始输入
        - valid   : bool
        - reason  : str  不合法时的简短说明，合法时为 None
    """
    if not smiles or not isinstance(smiles, str):
        return {'smiles': smiles, 'valid': False, 'reason': 'empty or non-string input'}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'smiles': smiles, 'valid': False, 'reason': 'RDKit cannot parse SMILES'}
    return {'smiles': smiles, 'valid': True, 'reason': None}


def canonicalize(smiles: str) -> dict:
    """将 SMILES 转换为 RDKit 规范形式，消除写法差异。

    Returns
    -------
    dict
        - smiles_input     : 原始输入
        - smiles_canonical : 规范 SMILES；解析失败时为 None
        - valid            : bool
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {'smiles_input': smiles, 'smiles_canonical': None, 'valid': False}
    return {
        'smiles_input':     smiles,
        'smiles_canonical': Chem.MolToSmiles(mol),
        'valid':            True,
    }


def get_properties(smiles: str) -> dict:
    """计算单个分子的基础属性。

    Returns
    -------
    dict
        - smiles          : 规范 SMILES（输入无效时保留原始值）
        - valid           : bool
        - formula         : str   分子式，如 'C7H5N3O6'
        - molecular_weight: float  g/mol，精确到小数点后 4 位
        - num_atoms       : int   重原子数
        - num_rings       : int
        - num_rotatable_bonds : int
        - elements        : list[str]  出现的元素列表（去重，排序）
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {
            'smiles': smiles, 'valid': False,
            'formula': None, 'molecular_weight': None,
            'num_atoms': None, 'num_rings': None,
            'num_rotatable_bonds': None, 'elements': None,
        }
    mol_h = Chem.AddHs(mol)
    elements = sorted({atom.GetSymbol() for atom in mol_h.GetAtoms()})
    return {
        'smiles':               Chem.MolToSmiles(mol),
        'valid':                True,
        'formula':              CalcMolFormula(mol),
        'molecular_weight':     round(Descriptors.MolWt(mol), 4),
        'num_atoms':            mol.GetNumAtoms(),
        'num_rings':            rdMolDescriptors.CalcNumRings(mol),
        'num_rotatable_bonds':  rdMolDescriptors.CalcNumRotatableBonds(mol),
        'elements':             elements,
    }


# ── 批量操作 ──────────────────────────────────────────────────────────────────

def batch_validate(smiles_list: Iterable[str]) -> dict:
    """批量验证 SMILES 列表。

    Returns
    -------
    dict
        - results   : list[dict]  每项同 validate() 返回值
        - n_total   : int
        - n_valid   : int
        - n_invalid : int
        - valid_smiles   : list[str]  仅有效的原始 SMILES
        - invalid_smiles : list[str]  仅无效的原始 SMILES
    """
    results = [validate(s) for s in smiles_list]
    valid   = [r['smiles'] for r in results if r['valid']]
    invalid = [r['smiles'] for r in results if not r['valid']]
    return {
        'results':       results,
        'n_total':       len(results),
        'n_valid':       len(valid),
        'n_invalid':     len(invalid),
        'valid_smiles':  valid,
        'invalid_smiles': invalid,
    }


def batch_canonicalize(smiles_list: Iterable[str]) -> dict:
    """批量规范化 SMILES 列表。

    Returns
    -------
    dict
        - results          : list[dict]   每项同 canonicalize() 返回值
        - canonical_smiles : list[str]    仅有效且已规范化的 SMILES
        - n_total          : int
        - n_valid          : int
        - n_invalid        : int
    """
    results   = [canonicalize(s) for s in smiles_list]
    canonical = [r['smiles_canonical'] for r in results if r['valid']]
    return {
        'results':          results,
        'canonical_smiles': canonical,
        'n_total':          len(results),
        'n_valid':          len(canonical),
        'n_invalid':        len(results) - len(canonical),
    }


def batch_properties(smiles_list: Iterable[str],
                     skip_invalid: bool = True) -> dict:
    """批量计算分子属性。

    Parameters
    ----------
    smiles_list  : SMILES 字符串列表
    skip_invalid : True 时返回 results 里只含有效分子

    Returns
    -------
    dict
        - results : list[dict]  每项同 get_properties() 返回值
        - n_total : int
        - n_valid : int
    """
    raw = [get_properties(s) for s in smiles_list]
    results = [r for r in raw if r['valid']] if skip_invalid else raw
    return {
        'results': results,
        'n_total': len(raw),
        'n_valid': sum(1 for r in raw if r['valid']),
    }


def deduplicate(smiles_list: Iterable[str],
                keep_invalid: bool = False) -> dict:
    """按规范 SMILES 对列表去重。

    去重规则：以 RDKit 生成的规范 SMILES 作为唯一键；无法解析的 SMILES
    若 keep_invalid=True 则原样保留（不参与去重逻辑）。

    Parameters
    ----------
    smiles_list  : 原始 SMILES 列表（可含重复）
    keep_invalid : 是否保留无法解析的 SMILES（默认丢弃）

    Returns
    -------
    dict
        - unique_smiles  : list[str]  去重后的规范 SMILES 列表
        - n_input        : int   输入总数
        - n_unique       : int   去重后数量
        - n_removed      : int   删去的重复数
        - n_invalid      : int   解析失败数（已丢弃或保留，取决于 keep_invalid）
    """
    smiles_list = list(smiles_list)
    seen: dict[str, bool] = {}   # canonical -> seen
    unique: list[str] = []
    n_invalid = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            n_invalid += 1
            if keep_invalid and smi not in seen:
                seen[smi] = True
                unique.append(smi)
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen:
            seen[canon] = True
            unique.append(canon)

    return {
        'unique_smiles': unique,
        'n_input':       len(smiles_list),
        'n_unique':      len(unique),
        'n_removed':     len(smiles_list) - len(unique) - (n_invalid if not keep_invalid else 0),
        'n_invalid':     n_invalid,
    }
