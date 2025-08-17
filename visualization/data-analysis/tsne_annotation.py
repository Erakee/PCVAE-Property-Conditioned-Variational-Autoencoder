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


def get_morgan_fingerprints(smiles_list, radius=2, n_bits=1024):
    mols = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_smiles.append(smi)
    # 使用旧版方法
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits) for mol in mols]
    return np.array([list(fp) for fp in fps]), valid_smiles



def main(seed=42, set_enthalpy='-100', model='cvae_dhr'):
    # 在main函数开头定义全局色标范围
    COLOR_RANGE = (-450, 300)  # 新增全局变量
    parser = argparse.ArgumentParser(description='Molecular Generation with CVAE')
    parser.add_argument('--random', action='store_true', help='Random generation')
    parser.add_argument('--info_output', type=str, default=f'generate_smi/{model}/generated_info.txt', help='Output file path')
    parser.add_argument('--output', type=str, default=f'generate_smi/{model}/generated_smiles.smi',
                        help='Output file path')
    args = parser.parse_args()
    config = utils.config
    tokenizer = utils.get_tokenizer()
    smilesDataset = SmilesDictDataset(config['fname_dataset'], tokenizer, config['maxLength'])
    dataset_smiles_list = smilesDataset.smiles
    dataset_enthalpy = smilesDataset.data['heat_of_formation'].tolist()
    dataset_X, dataset_valid_smiles = get_soap_descriptors(dataset_smiles_list)
    # dataset_X, dataset_valid_smiles = get_morgan_fingerprints(dataset_smiles_list)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    dataset_X_tsne = tsne.fit_transform(dataset_X)
    dump(tsne, f'tsne_model_original_dataset.joblib')

    smi_file_path = fr'D:\Project\EVAE_paper\generate_smi\{model}\nto_neg100.smi'
    with open(smi_file_path, 'r') as f:
        new_smiles_list = [line.strip() for line in f]
    new_X, new_valid_smiles = get_soap_descriptors(new_smiles_list)
    # new_X, new_valid_smiles = get_morgan_fingerprints(new_smiles_list)
    # new_gen_enthalpy = [predict_enthalpy(smi) for smi in new_valid_smiles]

    loaded_tsne = load(f'tsne_model_original_dataset.joblib')
    combined_X = np.vstack((dataset_X, new_X))
    combined_X_tsne = loaded_tsne.fit_transform(combined_X)
    sampled_X_tsne = combined_X_tsne[len(dataset_X):]

    # 1. 原始数据集绘图部分
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(dataset_X_tsne[:, 0], dataset_X_tsne[:, 1],
                          c=dataset_enthalpy, cmap='RdBu_r',
                          alpha=0.7, vmin=COLOR_RANGE[0], vmax=COLOR_RANGE[1])  # 添加vmin/vmax
    cbar = plt.colorbar(scatter)
    cbar.set_label('Heat of Formation (kcal/mol)')
    plt.title("t-SNE Visualization of Original Molecular Space")
    plt.tight_layout()
    plt.savefig(fr'D:\Project\EVAE_paper\visualization\data-analysis/annotation_plot/{model}/620NTO_original_dataset_tsne_plot.png', dpi=300)
    # plt.show()

    def smiles_to_image(smiles, size=(100, 100)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            img = Draw.MolToImage(mol, size=size)
            return img
        else:
            return Image.new('RGB', size, color='white')

    # 可调参数：
    image_zoom = 0.5  # 缩放倍数（越小图越小）
    offset = (15, 20)  # 图片相对偏移量 (x, y)，单位为像素，可以调位置
    image_size = (150, 150)  # RDKit生成图片的尺寸

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # 背景原始数据（淡色显示）
    ax.scatter(dataset_X_tsne[:, 0], dataset_X_tsne[:, 1],
               c=dataset_enthalpy, cmap='RdBu_r',
               alpha=0.15, vmin=COLOR_RANGE[0], vmax=COLOR_RANGE[1])

    # 主体：采样点 + 小图 + 线
    # for i, (x, y) in enumerate(sampled_X_tsne):
    #     smi = new_valid_smiles[i]
    #     img = smiles_to_image(smi, size=image_size)
    #     imagebox = OffsetImage(img, zoom=image_zoom)
    #
    #     # 偏移坐标（控制图片位置）
    #     xybox = (x + offset[0], y + offset[1])
    #
    #     # 画 AnnotationBbox：包含图像 + 连线
    #     ab = AnnotationBbox(imagebox, (x, y), xybox=xybox,
    #                         frameon=True, pad=0.2,
    #                         arrowprops=dict(arrowstyle='-', color='gray', lw=1),
    #                         bboxprops=dict(edgecolor='black', linewidth=0.5))
    #     ax.add_artist(ab)
    # 获取 t-SNE 坐标中心，作为偏移参考
    x_center = np.mean(dataset_X_tsne[:, 0])
    y_center = np.mean(dataset_X_tsne[:, 1])

    # 可选：手动修图偏移（你可以填索引和新 offset）
    manual_offsets = {
        # 例如：第3个分子偏移大一点
        # 2: (-30, 50),
        # 5: (40, -40),
    }

    # 主体：采样点 + 小图 + 动态偏移 + 线
    for i, (x, y) in enumerate(sampled_X_tsne):
        smi = new_valid_smiles[i]
        img = smiles_to_image(smi, size=image_size)
        imagebox = OffsetImage(img, zoom=image_zoom)

        # 计算相对中心的方向
        dx = 1 if x < x_center else -1
        dy = 1 if y < y_center else -1

        # 默认偏移
        offset_x, offset_y = dx * 25, dy * 30

        # 如果有手动偏移，替换掉默认
        if i in manual_offsets:
            offset_x, offset_y = manual_offsets[i]

        xybox = (x + offset_x, y + offset_y)

        ab = AnnotationBbox(imagebox, (x, y), xybox=xybox,
                            frameon=True, pad=0.2,
                            arrowprops=dict(arrowstyle='-', color='gray', lw=1),
                            bboxprops=dict(edgecolor='black', linewidth=0.5))
        ax.add_artist(ab)

    # 绘制黄色边框标记点（可选）
    ax.scatter(sampled_X_tsne[:, 0], sampled_X_tsne[:, 1],
               edgecolors='black', facecolors='none', c='yellow',
               linewidths=1.2, s=60)

    plt.title("t-SNE of Sampled Molecules with Structure or Index")
    # plt.colorbar(label='Heat of Formation (kcal/mol)')
    plt.tight_layout()
    plt.savefig(fr'D:\Project\EVAE_paper\visualization\data-analysis/annotation_plot/{model}/620NTO_sampled_tsne_with_structures{set_enthalpy}.png', dpi=300)
    # plt.show()

    # 3. 组合绘图部分
    plt.figure(figsize=(10, 6))
    # 原始数据点
    sc1 = plt.scatter(dataset_X_tsne[:, 0], dataset_X_tsne[:, 1],
                      c=dataset_enthalpy, cmap='RdBu_r',
                      alpha=0.7, vmin=COLOR_RANGE[0], vmax=COLOR_RANGE[1],  # 统一范围
                      label='Original Data')

    # 生成数据点
    sc2 = plt.scatter(sampled_X_tsne[:, 0], sampled_X_tsne[:, 1],
                      c='yellow',
                      alpha=0.7, #vmin=COLOR_RANGE[0], vmax=COLOR_RANGE[1],  # 统一范围
                      edgecolors='black', linewidths=1,
                      label='Sampled Data')

    # 统一色标
    cbar = plt.colorbar(sc1)
    cbar.set_label('Heat of Formation (kcal/mol)')
    plt.title("Combined t-SNE Visualization of Molecular Space")
    plt.legend(handles=[sc1, sc2], prop={'size': 14}, markerscale=2)
    plt.tight_layout()
    plt.savefig(fr'D:\Project\EVAE_paper\visualization\data-analysis/annotation_plot/{model}/620NTO_combined_tsne_plot_{set_enthalpy}.png', dpi=300)
    # plt.show()


if __name__ == '__main__':
    main()
