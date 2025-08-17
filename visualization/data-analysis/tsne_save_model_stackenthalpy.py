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
import plotly.graph_objects as go
import pandas as pd


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



def main(seed=42, set_enthalpy='300', model='cvae_dhr'):
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

    smi_file_path = fr'D:\Project\EVAE_paper\generate_smi\{model}\nto_100_generated_smiles.smi'
    with open(smi_file_path, 'r') as f:
        new_smiles_list = [line.strip() for line in f]
    new_X, new_valid_smiles = get_soap_descriptors(new_smiles_list)
    # new_X, new_valid_smiles = get_morgan_fingerprints(new_smiles_list)
    # new_gen_enthalpy = [predict_enthalpy(smi) for smi in new_valid_smiles]

    loaded_tsne = load(f'tsne_model_original_dataset.joblib')
    combined_X = np.vstack((dataset_X, new_X))
    combined_X_tsne = loaded_tsne.fit_transform(combined_X)
    sampled_X_tsne = combined_X_tsne[len(dataset_X):]

    # # 1. 原始数据集绘图部分
    # plt.figure(figsize=(10, 6))
    # scatter = plt.scatter(dataset_X_tsne[:, 0], dataset_X_tsne[:, 1],
    #                       c=dataset_enthalpy, cmap='RdBu_r',
    #                       alpha=0.7, vmin=COLOR_RANGE[0], vmax=COLOR_RANGE[1])  # 添加vmin/vmax
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Heat of Formation (kcal/mol)')
    # plt.title("t-SNE Visualization of Original Molecular Space")
    # plt.tight_layout()
    # plt.savefig(fr'gen_tsne_plt/{model}/original_dataset_tsne_plot.png', dpi=300)
    # # plt.show()
    #
    # # 2. 采样数据绘图部分
    # plt.figure(figsize=(14, 8))
    # scatter = plt.scatter(sampled_X_tsne[:, 0], sampled_X_tsne[:, 1],
    #                       c='yellow', edgecolors='black',
    #                       alpha=0.7)#, vmin=COLOR_RANGE[0], vmax=COLOR_RANGE[1])  # 添加vmin/vmax
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Heat of Formation (kcal/mol)')
    # plt.title("t-SNE Visualization of Sampled Molecular Space")
    # plt.tight_layout()
    # plt.savefig(fr'gen_tsne_plt/{model}_{set_enthalpy}.png', dpi=300)

    # 3. 多条件叠加绘图
    plt.figure(figsize=(14, 8))

    # 原始训练集点（背景）
    sc1 = plt.scatter(dataset_X_tsne[:, 0], dataset_X_tsne[:, 1],
                      c=dataset_enthalpy, cmap='RdBu_r',
                      alpha=0.7, vmin=COLOR_RANGE[0], vmax=COLOR_RANGE[1],
                      label='Training set', s=75)

    # 指定四个目标 EoF 条件
    target_point_size = 135
    target_eof_values = [-300, -100, 100, 250]
    color_map = {
        -300: '#2F7FC1',#'gold',
        -100: '#96C37D',#'limegreen',
        100: '#F3D266',#'orangered',
        250: '#D8383A',#'purple'
    }
    # target_eof_values = [100]
    # color_map = {
    #     -300: '#2F7FC1',  # 'gold',
    #     -100: '#96C37D',  # 'limegreen',
    #     100: '#F3D266',  # 'orangered',
    #     250: '#D8383A',  # 'purple'
    # }
    legend_handles = [sc1]

    for target_eof in target_eof_values:
         # 构建文件名（负号要特殊处理，改成下划线或其它替换避免路径问题）
        safe_eof_name = str(target_eof).replace('-', '_')
        smi_path = fr'D:\Project\EVAE_paper\generate_smi\cvae_dhr\gen_{safe_eof_name}.smi'
        # smi_path = r'D:\Project\EVAE_paper\generate_smi\vae_h\em_vae500.smi'

        # 读取并计算 t-SNE
        with open(smi_path, 'r') as f:
            smiles = [line.strip() for line in f]
        new_X, new_valid_smiles = get_soap_descriptors(smiles)
        new_tsne = loaded_tsne.fit_transform(np.vstack((dataset_X, new_X)))[len(dataset_X):]

        # 绘制生成分子点
        sc = plt.scatter(new_tsne[:, 0], new_tsne[:, 1],
                         color=color_map[target_eof], alpha=0.9, edgecolors='black',
                         label=f'Target EoF = {target_eof}', linewidths=1.5,
                         s=target_point_size)
        # sc = plt.scatter(new_tsne[:, 0], new_tsne[:, 1],
        #                   color=color_map[target_eof], alpha=0.9, edgecolors='black',
        #                   label=f'Sampled Data', linewidths=1.5,
        #                   s=target_point_size)
        legend_handles.append(sc)

    # 颜色条
    cbar = plt.colorbar(sc1)
    cbar.set_label('Enthalpy of Formation (kcal/mol)', fontsize=22)
    cbar.ax.tick_params(labelsize=22)

    # 标题和布局
    # plt.title("t-SNE Distribution of Generated Molecules under Different EoF Conditions", fontsize=16)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    # 图例放在右上角，避免遮挡数据，字体加大
    plt.legend(handles=legend_handles,
               prop={'size': 18},
               loc='upper right',
               scatterpoints=1,
               markerscale=1.2,
               framealpha=1.0)

                # bbox_to_anchor=(1.02, 1.0))  # 右上偏外
               # borderaxespad=0.5)

    plt.tight_layout()  #rect=[0, 0, 0.85, 1])  # 收缩图像区域以空出 legend 区
    plt.savefig(fr'gen_tsne_plt/{model}/combined_tsne_multiple_eof_overlay.png', dpi=450)

    # plt.savefig(fr'gen_tsne_plt/{model}/combined_random.png', dpi=450)


if __name__ == '__main__':
    main()
