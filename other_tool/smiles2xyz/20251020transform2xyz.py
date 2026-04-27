#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
smiles_txt_to_xyz.py
从当前目录的 txt 文件（每行一个 SMILES）读取，生成 3D 构型，并写入同一个 .xyz 文件（多分子块）。

用法示例：
    python smiles_txt_to_xyz.py --in smiles.txt --out molecules.xyz
"""

import argparse
from typing import List, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import MMFFGetMoleculeProperties
from rdkit.Chem import rdDistGeom

def read_smiles_lines(path: str) -> List[str]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            smi = raw.strip()
            if not smi:
                continue
            lines.append(smi)
    return lines

def embed_and_optimize(mol_in: Chem.Mol, seed: int = 2025) -> Tuple[Optional[Chem.Mol], str]:
    """
    对传入分子加氢、ETKDG嵌入、MMFF(或UFF)优化。
    成功：返回 (mol_with_conformer, "")
    失败：返回 (None, reason)
    """
    try:
        mol = Chem.AddHs(mol_in)
    except Exception as e:
        return None, f"AddHs failed: {e}"

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.1
    params.useRandomCoords = True

    try:
        cid = AllChem.EmbedMolecule(mol, params)
        if cid == -1:
            return None, "EmbedMolecule failed"
    except Exception as e:
        return None, f"EmbedMolecule exception: {e}"

    # 优化：优先 MMFF，失败再 UFF
    try:
        mp = MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if mp is not None:
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=1000)
        else:
            raise ValueError("MMFF parameters unavailable, fallback to UFF.")
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
        except Exception as e2:
            return None, f"Forcefield optimization failed: {e2}"

    return mol, ""

def mol_to_xyz_block(mol: Chem.Mol, title: str) -> str:
    """
    将带构象的分子转为一个 XYZ 块（不含末尾空行）。
    第一行：原子数
    第二行：标题（这里写 SMILES）
    后续：元素 x y z
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("No conformer on molecule.")
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    lines = [str(n), title]
    for idx in range(n):
        atom = mol.GetAtomWithIdx(idx)
        pos = conf.GetAtomPosition(idx)
        lines.append(f"{atom.GetSymbol():<2} {pos.x: .6f} {pos.y: .6f} {pos.z: .6f}")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Read SMILES from a txt file and write a multi-molecule XYZ file")
    parser.add_argument("--in", dest="infile", type=str, default="20251020.txt", help="输入 SMILES 文本文件（每行一个SMILES）")
    parser.add_argument("--out", dest="outfile", type=str, default="molecules.xyz", help="输出 XYZ 文件名")
    parser.add_argument("--max", type=int, default=None, help="最多处理多少个分子（调试用）")
    parser.add_argument("--seed", type=int, default=2025, help="嵌入随机种子（保证可复现）")
    args = parser.parse_args()

    smiles_list = read_smiles_lines(args.infile)
    if args.max is not None:
        smiles_list = smiles_list[:args.max]

    xyz_blocks: List[str] = []
    failed: List[Tuple[int, str, str]] = []

    for i, smi in enumerate(smiles_list, start=1):
        mol0 = Chem.MolFromSmiles(smi)
        if mol0 is None:
            failed.append((i, smi, "MolFromSmiles failed"))
            continue

        mol3d, err = embed_and_optimize(mol0, seed=args.seed)
        if mol3d is None:
            failed.append((i, smi, err))
            continue

        try:
            block = mol_to_xyz_block(mol3d, title=smi)
        except Exception as e:
            failed.append((i, smi, f"mol_to_xyz_block: {e}"))
            continue

        xyz_blocks.append(block)

    with open(args.outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(xyz_blocks))
        # f.write("\n")

    print(f"[OK] 写入 {len(xyz_blocks)} 个分子到文件：{args.outfile}")
    if failed:
        print("[WARN] 以下分子未能生成：")
        for idx, smi, reason in failed:
            print(f"  #{idx}: {smi} -> {reason}")

if __name__ == "__main__":
    main()
