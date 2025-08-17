import os
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel
from collections import Counter

def has_duplicate_coordinates(mol, digits=3):
    conf = mol.GetConformer()
    coords = [(round(conf.GetAtomPosition(i).x, digits),
               round(conf.GetAtomPosition(i).y, digits),
               round(conf.GetAtomPosition(i).z, digits)) for i in range(mol.GetNumAtoms())]
    count = Counter(coords)
    return any(v > 1 for v in count.values())

def smiles_to_xyz_batch(input_file, output_dir, add_smiles_comment=True):
    os.makedirs(output_dir, exist_ok=True)
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("mol", "xyz")

    with open(input_file, 'r') as f:
        for idx, line in enumerate(f):
            smiles = line.strip()
            if not smiles:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"[Warning] Invalid SMILES at line {idx + 1}: {smiles}")
                continue
            mol = Chem.AddHs(mol)

            params = AllChem.ETKDGv3()
            params.randomSeed = 43
            status = AllChem.EmbedMolecule(mol, params)
            if status != 0:
                print(f"[Warning] Embedding failed at line {idx + 1}: {smiles}")
                continue

            # 能量优化
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except:
                print(f"[Warning] Optimization failed at line {idx + 1}: {smiles}")
                continue

            # 检查是否存在重复原子坐标
            if has_duplicate_coordinates(mol):
                print(f"[Warning] Duplicate coordinates detected at line {idx + 1}: {smiles}")
                continue

            mol_block = Chem.MolToMolBlock(mol)
            obMol = openbabel.OBMol()
            obConversion.ReadString(obMol, mol_block)
            xyz_str = obConversion.WriteString(obMol).strip().splitlines()

            if add_smiles_comment and len(xyz_str) >= 2:
                xyz_str[1] = f"SMILES: {smiles}"

            output_path = os.path.join(output_dir, f'molecule_{idx + 1}.xyz')
            with open(output_path, 'w') as out_f:
                out_f.write('\n'.join(xyz_str) + '\n')

    print(f"\n✔️ 所有合法 SMILES 已转换为 XYZ 文件，输出目录：{output_dir}")


# 示例调用
if __name__ == '__main__':
    input_file = 'select_711.smi'
    output_dir = r'D:\Project\EVAE_paper\other_tool\smiles2xyz\batch3'
    smiles_to_xyz_batch(input_file, output_dir)
