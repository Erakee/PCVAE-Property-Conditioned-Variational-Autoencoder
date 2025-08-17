from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np


def visualize_molecules(smiles_list, enthalpy_list, n_cols=5, img_size=(300, 300), save_path=None):
    """
    Visualize a list of molecules in a grid layout and annotate with enthalpy

    Args:
        smiles_list (list): List of SMILES strings
        enthalpy_list (list): List of corresponding enthalpy values
        n_cols (int): Number of columns in the grid
        img_size (tuple): Size of each molecule image (width, height)
        save_path (str): Path to save the combined image (optional)
    """
    # Convert SMILES to RDKit molecules
    mols = []
    valid_enthalpy = []
    for smi, enthalpy in zip(smiles_list, enthalpy_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # Generate 2D coordinates for the molecule
            AllChem.Compute2DCoords(mol)
            mols.append(mol)
            valid_enthalpy.append(enthalpy)
        else:
            print(f"Warning: Could not parse SMILES: {smi}")

    if not mols:
        print("No valid molecules to display")
        return

    # Calculate grid dimensions
    n_mols = len(mols)
    n_rows = int(np.ceil(n_mols / n_cols))

    # Create figure
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))

    # Draw each molecule
    for idx, (mol, enthalpy) in enumerate(zip(mols, valid_enthalpy)):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        img = Draw.MolToImage(mol, size=img_size)
        ax.imshow(img)
        ax.axis('off')
        # Add enthalpy as title
        ax.set_title(f"ΔH: {enthalpy:.2f} kcal/mol", fontsize=22, wrap=True, y=-0.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def main():
    # 指定文件地址
    model_name = 'cvae_dhr'
    set_enthalpy = 'no'
    annotation = 'NTO_'
    file_path = fr'D:\Project\EVAE_paper\generate_smi\{model_name}\{annotation}generated_info.txt'
    # 从文件读取SMILES和生成焓
    smiles_list = ['C1(=NC(=O)NN1)[N+](=O)[O-]']
    enthalpy_list = [-12.00]
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                smiles_list.append(parts[0])
                enthalpy_list.append(float(parts[1]))
            else:
                print(f"Warning: Invalid line format: {line}")

    visualize_molecules(
        smiles_list,
        enthalpy_list,
        n_cols=7,
        img_size=(300, 300),
        save_path=fr'D:\Project\EVAE_paper\generate_smi\{model_name}\{annotation}_with_{set_enthalpy}_enthalpy.png'
    )


if __name__ == "__main__":
    main()