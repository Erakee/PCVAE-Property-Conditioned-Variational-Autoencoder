import argparse
import os
import sys
import torch
import numpy as np
import model.VAE_H_SEED as vae
from dataset.dataset import SmilesDictDataset
from util.enthalpy_predictor import predict_enthalpy
from util.tokens import getTokenizer  # 根据你的实际导入路径调整
import util.utils as utils
from datetime import datetime


def main():
    model = 'vae_h'
    parser = argparse.ArgumentParser(description='Molecular Generation with CVAE')
    parser.add_argument('--random', action='store_true', help='Random generation')
    # parser.add_argument('--conditional', type=bool, default='True', help='Whether to generate conditional')
    parser.add_argument('--smiles', type=str, default='', help='Input SMILES for conditional generation')
    parser.add_argument('--enthalpy', type=float, default=None, help='Target enthalpy value')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples to generate')
    # parser.add_argument('--info_output', type=str, default='generate_smi/generated_info.txt', help='Output file path')
    # parser.add_argument('--ent_output', type=str, default='generate_smi/generated_enthalpy.txt', help='Output file path')
    # parser.add_argument('--output', type=str, default='generate_smi/generated_smiles.smi', help='Output file path')
    parser.add_argument('--info_output', type=str, default=f'generate_smi/{model}/generated_info.txt', help='Output file path')
    parser.add_argument('--ent_output', type=str, default=f'generate_smi/{model}/generated_enthalpy.txt', help='Output file path')
    parser.add_argument('--output', type=str, default=f'generate_smi/{model}/generated_smiles.smi', help='Output file path')
    args = parser.parse_args()
    # 加载配置
    config = utils.p_cfg(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = utils.get_tokenizer()
    maxLength = config['maxLength']
    pad_idx = tokenizer.getTokensNum('<pad>')
    smilesDataset = SmilesDictDataset(config['fname_dataset'], tokenizer, config['maxLength'])
    # lb, ub = smilesDataset._getbound()
    # vae_model = cvae.ConVAE(**config['vae_param'],
    #                         encoder_state_fname=config['fname_vae_encoder_parameters'],
    #                         decoder_state_fname=config['fname_vae_decoder_parameters'],
    #                         device=device)
    # vae_model.encoder.loadState()
    # vae_model.decoder.loadState()
    # vae_model.encoder.eval()
    # vae_model.decoder.eval()


    # 生成逻辑
    with torch.no_grad():
        # mu_conditions = []
        # if_full_cond = (len(args.smiles) > 1) and (args.enthalpy is not None)
        # nSample = args.num_samples
        # x = torch.zeros(nSample, maxLength, config['vae_param']['num_vocabs'])
        # if args.enthalpy and (len(args.smiles) < 1):
        #     norm_h = (args.enthalpy - lb) / (ub - lb)
        #     assert ((norm_h <= 1) and (norm_h > 0)), f'Given enthalpy is out of range:{lb}~{ub}.'
        #     h_tensor = torch.tensor([norm_h], dtype=torch.float32, device=device)
        #     norm_h_tensor = torch.tensor([norm_h], dtype=torch.float32, device=device)
        #     norm_n_h = norm_h_tensor.unsqueeze(0).repeat(nSample, 1).squeeze(1)
        #     latent_vec = torch.randn((nSample, config['vae_param']['latent_dim']), device=device)
        #     pred_y = vae_model.decoder(latent_vec, norm_n_h, None, freerun=True)
        #     pred_one_hot = torch.zeros_like(x, dtype=h_tensor.dtype)
        #     pred_y_argmax = torch.nn.functional.softmax(pred_y, dim=2).argmax(dim=2)
        #     for i in range(pred_one_hot.shape[0]):
        #         for j in range(pred_one_hot.shape[1]):
        #             pred_one_hot[i, j, pred_y_argmax[i, j]] = 1
        #     predicted_indices = pred_one_hot.argmax(dim=2)  # 处理后的索引 (0~16)
        #     predicted_indices_original = predicted_indices + 2  # 恢复为原始索引 (2~18)
        #     predicted_smiles = tokenizer.getSmiles(predicted_indices_original)

        file_path = r'D:\Project\EVAE_paper\generate_smi\vae_h\em_vae500.smi'
        with open(file_path, 'r') as f:
            predicted_smiles = [line.strip() for line in f]

        # 验证有效性
        validSmilesStrs = [sm for sm in predicted_smiles if utils.isValidSmiles(sm)]

        gen_enthalpy = []
        for smi in validSmilesStrs:
            gen_enthalpy.append(predict_enthalpy(smi))
        # 将有效分子和对应的生成焓组合成字符串列表
        output_lines = []
        output_ents = []
        for smi, enthalpy in zip(validSmilesStrs, gen_enthalpy):
            output_lines.append(f"{smi},{enthalpy}")
            output_ents.append(f"{enthalpy}")


        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 将当前时间添加到输出内容的开头
        output_lines.insert(0, current_time)
        output_ents.insert(0, current_time)
        stat_info = f"成功生成 {len(validSmilesStrs)} 个有效分子，有效率 {len(validSmilesStrs) / args.num_samples:.1%}"
        # 将统计信息添加到输出内容中
        output_lines.append(stat_info)
        output_ents.append(f"{stat_info}--enthalpy setting={args.enthalpy}\n")

        with open(args.info_output, 'a') as f:
            f.write('\n')
            f.write('\n'.join(output_lines))
            f.write('\n')
        with open(args.output, 'a') as f:
            f.write('\n')
            f.write('\n'.join(validSmilesStrs))
            f.write('\n')
        with open(args.ent_output, 'a') as f:
            f.write('\n')
            f.write('\n'.join(output_ents))
            f.write('\n')

        print(stat_info)
        print(f"结果已保存至 {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
# --smiles C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-] --enthalpy 52.8  #  RDX
# --smiles Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-] --enthalpy −16.01  # TNT
# CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]  TNT
#TNB  --smiles C1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-] --enthalpy -8.89 wiki     -78.4 kJ/mol (crystalline solid); -13.4 kJ/mol (gas)
#TATB --smiles C1(=C(C(=C(C(=C1[N+](=O)[O-])N)[N+](=O)[O-])N)[N+](=O)[O-])N --enthalpy -36.78  wiki