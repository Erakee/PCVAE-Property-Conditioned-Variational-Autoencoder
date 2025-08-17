import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from joblib import dump, load
from dscribe.descriptors import SOAP
from ase import Atoms
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem


def get_soap_descriptors(smiles_list, species):
    valid_mols = []
    valid_smiles = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                mol = Chem.AddHs(mol)
                result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                if result == -1:
                    continue
                atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
                conf = mol.GetConformer()
                positions = conf.GetPositions()
                ase_atoms = Atoms(numbers=atomic_numbers, positions=positions)
                valid_mols.append(ase_atoms)
                valid_smiles.append(smi)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error processing SMILES {smi}: {e}")
                continue

    cutoff = 6.0
    nmax = 8
    lmax = 6
    soap = SOAP(species=species, r_cut=cutoff, n_max=nmax, l_max=lmax, periodic=False)

    descriptors = []
    for mol in valid_mols:
        desc = soap.create(mol)
        mol_desc = np.mean(desc, axis=0)
        descriptors.append(mol_desc)

    return np.array(descriptors), valid_smiles, valid_indices


def main():
    # 文件路径
    db_chon_path = 'database_CHON.csv'
    em_train_path = 'em_train.csv'

    # 读取并清洗数据：确保 SMILES 和焓值都存在
    db_valid_df = pd.read_csv(db_chon_path)[['SMILES_e', 'HOF/kJ/mol']].dropna()
    em_valid_df = pd.read_csv(em_train_path)[['smiles', 'heat_of_formation']].dropna()

    db_smiles_all = db_valid_df['SMILES_e'].tolist()
    db_enthalpy_all = (db_valid_df['HOF/kJ/mol'] / 4.184).tolist()  # 转为 kcal/mol

    em_smiles_all = em_valid_df['smiles'].tolist()
    em_enthalpy_all = em_valid_df['heat_of_formation'].tolist()

    # 提取统一 species
    all_species = set()
    for smi in em_smiles_all + db_smiles_all:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            all_species.update([atom.GetSymbol() for atom in mol.GetAtoms()])
    all_species = sorted(list(all_species))

    # 提取 em_train 特征
    print("Processing em_train...")
    em_X, em_valid_smiles, em_valid_indices = get_soap_descriptors(em_smiles_all, all_species)
    em_enthalpy_valid = [em_enthalpy_all[i] for i in em_valid_indices]

    # 提取 database_CHON 特征
    print("Processing database_CHON...")
    db_X, db_valid_smiles, db_valid_indices = get_soap_descriptors(db_smiles_all, all_species)
    db_enthalpy_valid = [db_enthalpy_all[i] for i in db_valid_indices]

    # 合并数据
    combined_X = np.vstack((em_X, db_X))
    combined_enthalpy = em_enthalpy_valid + db_enthalpy_valid

    # t-SNE 降维
    print("Running t-SNE...")
    tsne_model = TSNE(n_components=2, perplexity=30, random_state=42)
    combined_X_tsne = tsne_model.fit_transform(combined_X)
    dump(tsne_model, 'tsne_combined_model.joblib')

    # 拆分坐标与焓值
    em_tsne = combined_X_tsne[:len(em_X)]
    db_tsne = combined_X_tsne[len(em_X):]
    em_eof = em_enthalpy_valid
    db_eof = db_enthalpy_valid

    # 可视化
    print("Plotting...")
    plt.figure(figsize=(10, 6))

    # em_train 点（无边框）
    sc1 = plt.scatter(em_tsne[:, 0], em_tsne[:, 1],
                      c=em_eof, cmap='RdBu_r', alpha=0.8,
                      vmin=-450, vmax=300,
                      label='em_train', linewidths=0)

    # database_CHON 点（黑边框）
    sc2 = plt.scatter(db_tsne[:, 0], db_tsne[:, 1],
                      c=db_eof, cmap='RdBu_r', alpha=0,
                      vmin=-450, vmax=300,
                      label='database_CHON',
                      edgecolors='black', linewidths=0.4)

    # 添加颜色条与图例
    cbar = plt.colorbar(sc1)
    cbar.set_label("Enthalpy of Formation (kcal/mol)")
    plt.legend()
    plt.title("t-SNE Projection Colored by Enthalpy of Formation")
    plt.tight_layout()
    plt.savefig("tsne_colored_by_enthalpy_with_outline.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
