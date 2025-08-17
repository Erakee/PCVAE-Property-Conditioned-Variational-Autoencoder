import sys
import os
from matplotlib import pyplot as plt
from rdkit.Chem import AllChem
import numpy as np
from rdkit import Chem
from sklearn.manifold import TSNE
import argparse
import torch
# import model.CVAE_DHR as cvae
from dataset.dataset import SmilesDictDataset
# from util.enthalpy_predictor import predict_enthalpy
from util.utils import get_soap_descriptors
from util.tokens import getTokenizer
import util.utils as utils
from datetime import datetime
from joblib import dump, load
from dscribe.descriptors import SOAP
from ase import Atoms
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from rdkit.Chem import Draw
from PIL import Image
from matplotlib.backend_bases import MouseButton


def get_morgan_fingerprints(smiles_list, radius=2, n_bits=1024):
    mols = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_smiles.append(smi)
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits) for mol in mols]
    return np.array([list(fp) for fp in fps]), valid_smiles


def main(seed=42, set_enthalpy='_no', model='cvae_dhr'):
    COLOR_RANGE = (-450, 300)
    parser = argparse.ArgumentParser(description='Molecular Generation with CVAE')
    parser.add_argument('--random', action='store_true', help='Random generation')
    parser.add_argument('--info_output', type=str, default=f'generate_smi/{model}/generated_info.txt', help='Output file path')
    parser.add_argument('--output', type=str, default=f'generate_smi/{model}/generated_smiles.smi', help='Output file path')
    args = parser.parse_args()

    config = utils.config
    tokenizer = utils.get_tokenizer()
    smilesDataset = SmilesDictDataset(config['fname_dataset'], tokenizer, config['maxLength'])
    dataset_smiles_list = smilesDataset.smiles
    dataset_enthalpy = smilesDataset.data['heat_of_formation'].tolist()
    dataset_X, dataset_valid_smiles = get_soap_descriptors(dataset_smiles_list)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    dataset_X_tsne = tsne.fit_transform(dataset_X)
    dump(tsne, f'tsne_model_original_dataset.joblib')

    smi_file_path = fr'D:\Project\EVAE_paper\generate_smi\{model}\NTO_selected.smi'
    with open(smi_file_path, 'r') as f:
        new_smiles_list = [line.strip() for line in f]
    new_X, new_valid_smiles = get_soap_descriptors(new_smiles_list)

    loaded_tsne = load(f'tsne_model_original_dataset.joblib')
    combined_X = np.vstack((dataset_X, new_X))
    combined_X_tsne = loaded_tsne.fit_transform(combined_X)
    sampled_X_tsne = combined_X_tsne[len(dataset_X):]

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.scatter(dataset_X_tsne[:, 0], dataset_X_tsne[:, 1],
               c=dataset_enthalpy, cmap='RdBu_r',
               alpha=0.15, vmin=COLOR_RANGE[0], vmax=COLOR_RANGE[1])

    image_size = (150, 150)
    image_zoom = 0.5

    def smiles_to_image(smiles, size=(100, 100)):
        mol = Chem.MolFromSmiles(smiles)
        return Draw.MolToImage(mol, size=size) if mol is not None else Image.new('RGB', size, color='white')

    image_artists = {}
    draggable_abox = {}

    def on_click(event):
        if event.inaxes != ax:
            return
        x_click = event.xdata
        y_click = event.ydata
        if x_click is None or y_click is None:
            return

        min_dist = float('inf')
        closest_index = None
        for i, (x, y) in enumerate(sampled_X_tsne):
            dist = (x - x_click) ** 2 + (y - y_click) ** 2
            if dist < min_dist:
                min_dist = dist
                closest_index = i

        if closest_index is None:
            return

        if event.button == MouseButton.RIGHT:
            if closest_index in image_artists:
                ab = image_artists.pop(closest_index)
                ab.remove()
                ax.figure.canvas.draw()
            else:
                img = smiles_to_image(new_valid_smiles[closest_index], size=image_size)
                imagebox = OffsetImage(img, zoom=image_zoom)
                ab = AnnotationBbox(imagebox, sampled_X_tsne[closest_index], frameon=True)
                ax.add_artist(ab)
                image_artists[closest_index] = ab
                draggable_abox[closest_index] = ab
                ax.figure.canvas.draw()

    def on_motion(event):
        if event.inaxes != ax or not hasattr(on_motion, 'drag_index') or on_motion.drag_index is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        ab = draggable_abox.get(on_motion.drag_index)
        if ab:
            ab.xy = (event.xdata, event.ydata)
            ax.figure.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax or event.button != MouseButton.LEFT:
            return
        if event.xdata is None or event.ydata is None:
            return

        for idx, ab in draggable_abox.items():
            ab_pos = ab.xy
            # 设置一个更宽松的点击范围
            if abs(ab_pos[0] - event.xdata) < 0.2 and abs(ab_pos[1] - event.ydata) < 0.2:
                on_motion.drag_index = idx
                return

        on_motion.drag_index = None

    def on_release(event):
        on_motion.drag_index = None

    plt.scatter(sampled_X_tsne[:, 0], sampled_X_tsne[:, 1],
                edgecolors='black', facecolors='none', c='yellow',
                linewidths=1.2, s=60)
    plt.title("Interactive t-SNE Visualization of Molecules")
    plt.tight_layout()

    fig = plt.gcf()
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    plt.savefig(fr'D:\Project\EVAE_paper\visualization\data-analysis/annotation_plot/{model}/interactive_tsne_plot.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
