import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import textwrap


def visualize_molecules(smiles_list, enthalpy_list, n_cols=6, img_size=(350, 350), save_path=None):
    mols = []
    valid_enthalpy = []
    valid_smiles = []

    for smi, enthalpy in zip(smiles_list, enthalpy_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            AllChem.Compute2DCoords(mol)
            mols.append(mol)
            valid_enthalpy.append(enthalpy)
            valid_smiles.append(smi)
        else:
            print(f"Warning: Could not parse SMILES: {smi}")

    if not mols:
        print("No valid molecules to display")
        return

    n_mols = len(mols)
    n_rows = int(np.ceil(n_mols / n_cols))
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))

    for idx, (mol, enthalpy, smi) in enumerate(zip(mols, valid_enthalpy, valid_smiles)):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        img = Draw.MolToImage(mol, size=img_size, kekulize=True, wedgeBonds=True, bgColor=None)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"ΔH: {enthalpy:.1f}", fontsize=14, y=1.05)

        # 添加 SMILES，自动换行显示（每行限制40字符）
        wrapped_smi = "\n".join(textwrap.wrap(smi, width=40))
        ax.text(0.5, -0.2, wrapped_smi, ha='center', va='top', fontsize=10, transform=ax.transAxes)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


def select_subset(df, num=12, mode="random", index_list=None):
    if mode == "random":
        return df.sample(n=num, random_state=42).reset_index(drop=True)
    elif mode == "head":
        return df.head(num).reset_index(drop=True)
    elif mode == "index" and index_list is not None:
        return df.loc[index_list].reset_index(drop=True)
    else:
        raise ValueError("Invalid mode or missing index_list.")


def main():
    # 路径设置
    csv_path = r"D:\Project\EVAE_paper\data\em_train.csv"
    save_path = r"D:\Project\EVAE_paper\visualization\molecule_visualization\dataset_sampled_molecules.png"

    # 抽样方式
    sample_num = 35
    sample_mode = "random"
    index_list = [0, 5, 10, 15]

    df = pd.read_csv(csv_path)
    sampled_df = select_subset(df, num=sample_num, mode=sample_mode, index_list=index_list)

    smiles_list = sampled_df['smiles'].tolist()
    enthalpy_list = sampled_df['heat_of_formation'].tolist()

    visualize_molecules(smiles_list, enthalpy_list, n_cols=6, img_size=(350, 350), save_path=save_path)


if __name__ == "__main__":
    main()


