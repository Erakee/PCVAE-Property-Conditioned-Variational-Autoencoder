from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np


def visualize_molecules(smiles_list, n_cols=5, img_size=(300, 300), save_path=None):
    """
    Visualize a list of molecules in a grid layout

    Args:
        smiles_list (list): List of SMILES strings
        n_cols (int): Number of columns in the grid
        img_size (tuple): Size of each molecule image (width, height)
        save_path (str): Path to save the combined image (optional)
    """
    # Convert SMILES to RDKit molecules
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # Generate 2D coordinates for the molecule
            AllChem.Compute2DCoords(mol)
            mols.append(mol)
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
    for idx, mol in enumerate(mols):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        img = Draw.MolToImage(mol, size=img_size)
        ax.imshow(img)
        ax.axis('off')
        # Add SMILES as title (optional)
        # ax.set_title(Chem.MolToSmiles(mol), fontsize=8, wrap=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def main():
    # 指定文件地址
    file_path = r'nto_neg100_selected.smi'
    # file_path = r'D:\Project\EVAE_paper\generate_smi\cvae_dhr\NTO_selected.smi'
    # 从文件读取SMILES
    with open(file_path, 'r') as f:
        smiles_list = [line.strip() for line in f]

    visualize_molecules(
        smiles_list,
        n_cols=5,
        img_size=(300, 300),
        save_path='molecules_neg100.png'
    )


if __name__ == "__main__":
    main()

    # rdx 52.8  C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]
    # hmx 60.4  C1N(CN(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]