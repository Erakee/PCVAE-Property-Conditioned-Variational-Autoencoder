import argparse
import os
import sys
import torch
import numpy as np
import model.CVAE_DHR as cvae
from dataset.dataset import SmilesDictDataset
from util.enthalpy_predictor import predict_enthalpy
from util.tokens import getTokenizer  # 根据你的实际导入路径调整
import util.utils as utils
from datetime import datetime


def main():
    model = 'cvae_dhr'
    enthalpy = 'neg100_'
    parser = argparse.ArgumentParser(description='Molecular Generation with CVAE')
    parser.add_argument('--random', action='store_true', help='Random generation')
    # parser.add_argument('--conditional', type=bool, default='True', help='Whether to generate conditional')
    parser.add_argument('--smiles', type=str, default='', help='Input SMILES for conditional generation')
    parser.add_argument('--enthalpy', type=float, default=None, help='Target enthalpy value')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to generate')
    # parser.add_argument('--info_output', type=str, default='generate_smi/generated_info.txt', help='Output file path')
    # parser.add_argument('--ent_output', type=str, default='generate_smi/generated_enthalpy.txt', help='Output file path')
    # parser.add_argument('--output', type=str, default='generate_smi/generated_smiles.smi', help='Output file path')
    parser.add_argument('--info_output', type=str, default=f'generate_smi/{model}/{enthalpy}NTO_generated_info.txt', help='Output file path')
    parser.add_argument('--ent_output', type=str, default=f'generate_smi/{model}/{enthalpy}NTO_generated_enthalpy.txt', help='Output file path')
    parser.add_argument('--output', type=str, default=f'generate_smi/{model}/{enthalpy}NTO_generated_smiles.smi', help='Output file path')
    args = parser.parse_args()
    # 加载配置
    config = utils.p_cfg(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = utils.get_tokenizer()
    maxLength = config['maxLength']
    pad_idx = tokenizer.getTokensNum('<pad>')
    smilesDataset = SmilesDictDataset(config['fname_dataset'], tokenizer, config['maxLength'])
    lb, ub = smilesDataset._getbound()
    vae_model = cvae.ConVAE(**config['vae_param'],
                            encoder_state_fname=config['fname_vae_encoder_parameters'],
                            decoder_state_fname=config['fname_vae_decoder_parameters'],
                            device=device)
    vae_model.encoder.loadState()
    vae_model.decoder.loadState()
    vae_model.encoder.eval()
    vae_model.decoder.eval()
    alpha = vae_model.encoder.alpha

    # 生成逻辑
    with torch.no_grad():
        mu_conditions = []
        # 明确条件类型：是否同时有SMILES和焓值
        has_smiles = len(args.smiles) > 0
        has_enthalpy = args.enthalpy is not None
        nSample = args.num_samples  # 明确样本数
        if (len(args.smiles) < 1) and (args.enthalpy is None):
            completely_rand = True
            latent_n_vec = torch.randn((nSample, vae_model.latent_dim), device=device)
            norm_n_h = torch.randn(nSample, device=device)
            X = torch.rand(nSample, maxLength, device=device)
        else:
            if args.smiles: # 有SMILES enthalpy不限的情况
                token_vector = tokenizer.tokenize([args.smiles], useTokenDict=True)[0]
                num_vector = tokenizer.getNumVector([token_vector], addStart=True, addEnd=True)[0]
                if max(num_vector) >= tokenizer.getTokensSize():
                    print(f"Warning: token index {max(num_vector)} exceeds vocabulary size {tokenizer.getTokensSize()}")
                # 截断/填充到 maxLength
                if len(num_vector) > maxLength:
                    truncated = [num_vector[0]] + num_vector[1:-1][:maxLength - 2] + [num_vector[-1]]
                    num_vector = truncated
                else:
                    padding = [pad_idx] * (maxLength - len(num_vector))
                    num_vector = num_vector + padding
                X = torch.tensor(num_vector, dtype=torch.long, device=device).unsqueeze(0).expand(nSample, -1)
                latent_x, mu_x, logvar_x,_,_ = vae_model.encoder(X, torch.zeros(1, device=device), alpha=1)
                mu_n_x = mu_x
                logvar_n_x = logvar_x
                # mu_conditions.append((0.6, mu_x))
            elif args.enthalpy: # 无SMILES 有enthalpy的情况
                X = torch.rand(nSample, maxLength, device=device)
                mu_n_x = torch.randn((nSample, config['vae_param']['latent_dim']), device=device)


            if args.enthalpy is not None: # 有SMILES 有 Enthalpy的情况
                norm_h = (args.enthalpy - lb) / (ub - lb)
                assert ((norm_h<= 1) and (norm_h > 0)), f'Given enthalpy is out of range:{lb}~{ub}.'
                h_tensor = torch.tensor([norm_h], dtype=torch.float32, device=device)
                mu_prior, logvar_prior = vae_model.encoder.prior_block(h_tensor.unsqueeze(1))
                mu_n_prior = mu_prior.repeat(nSample, 1)
                logvar_n_prior = logvar_prior.repeat(nSample, 1)
                norm_h_tensor = torch.tensor([norm_h], dtype=torch.float32, device=device)
                norm_n_h = norm_h_tensor.unsqueeze(0).repeat(nSample, 1).squeeze(1)
            else:
                norm_n_h = torch.rand(nSample)
                h_tensor = torch.tensor(norm_n_h, dtype=torch.float32, device=device)
                mu_n_prior, logvar_n_prior = vae_model.encoder.prior_block(h_tensor.unsqueeze(1))  #检查维度和上面的是不是一样！！！！！！！！！！！！！！！！！！！！！！！！！！！

            mu_n = alpha * mu_n_x + (1 - alpha) * mu_n_prior
            logvar_n = 0.5 * (alpha * logvar_n_x + (1 - alpha) * logvar_n_prior)
            latent_n_x = vae_model.encoder.reparameterize(mu_n, logvar_n)

        # 使用decoder进行生成
        y = vae_model.decoder(mu_n, norm_n_h, X, freerun=True).cpu()#也要改，X分单条件和多条件的，有无给定smiles影响X维度
        # 将decoder输出还原成smiles
        smilesStrs = tokenizer.getSmiles(y)
        # 验证有效性
        validSmilesStrs = [sm for sm in smilesStrs if utils.isValidSmiles(sm)]

        if args.enthalpy is not None:
            gen_enthalpy = []
            for smi in validSmilesStrs:
                gen_enthalpy.append(predict_enthalpy(smi))
            # 将有效分子和对应的生成焓组合成字符串列表
            output_lines = []
            output_ents = []
            for smi, enthalpy in zip(validSmilesStrs, gen_enthalpy):
                output_lines.append(f"{smi},{enthalpy}")
                output_ents.append(f"{enthalpy}")
        else:
            gen_enthalpy = []
            output_lines = []
            for smi in validSmilesStrs:
                gen_enthalpy.append(predict_enthalpy(smi))

            output_ents = []
            for smi, enthalpy in zip(validSmilesStrs, gen_enthalpy):
                output_lines.append(f"{smi},{enthalpy}")
                output_ents.append(f"{enthalpy}")

        # 准备三份内容
        smiles_lines = validSmilesStrs
        info_lines = [f"{smi},{enthalpy}" for smi, enthalpy in zip(validSmilesStrs, gen_enthalpy)]
        enthalpy_lines = [f"{enthalpy}" for enthalpy in gen_enthalpy]

        # 添加时间和统计
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stat_info = f"成功生成 {len(validSmilesStrs)} 个有效分子，有效率 {len(validSmilesStrs) / args.num_samples:.1%}"

        # 写 info（SMILES + enthalpy）
        with open(args.info_output, 'a') as f:
            f.write(f'\n{current_time}\n')
            f.write('\n'.join(info_lines) + '\n')
            f.write(stat_info + '\n')

        # 写 smiles（只写 smiles）
        with open(args.output, 'a') as f:
            f.write('\n' + '\n'.join(smiles_lines) + '\n')

        # 写 enthalpy（只写 enthalpy）
        with open(args.ent_output, 'a') as f:
            f.write(f'\n{current_time}\n')
            f.write('\n'.join(enthalpy_lines) + '\n')
            f.write(f"{stat_info}--enthalpy setting={args.enthalpy}\n")
        # # 获取当前时间
        # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # # 将当前时间添加到输出内容的开头
        # output_lines.insert(0, current_time)
        # output_ents.insert(0, current_time)
        # stat_info = f"成功生成 {len(validSmilesStrs)} 个有效分子，有效率 {len(validSmilesStrs) / args.num_samples:.1%}"
        # # 将统计信息添加到输出内容中
        # output_lines.append(stat_info)
        # output_ents.append(f"{stat_info}--enthalpy setting={args.enthalpy}\n")

        # with open(args.info_output, 'a') as f:
        #     f.write('\n')
        #     f.write('\n'.join(output_lines))
        #     f.write('\n')
        # with open(args.output, 'a') as f:
        #     f.write('\n')
        #     f.write('\n'.join(validSmilesStrs))
        #     f.write('\n')
        # with open(args.ent_output, 'a') as f:
        #     f.write('\n')
        #     f.write('\n'.join(output_ents))
        #     f.write('\n')
        # 写 info（带 enthalpy 的，带时间和统计）


        print(stat_info)
        print(f"结果已保存至 {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
# --smiles C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-] --enthalpy 52.8  #  RDX
# --smiles Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-] --enthalpy −16.01  # TNT
# CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]  TNT
#TNB  --smiles C1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-] --enthalpy -8.89 wiki     -78.4 kJ/mol (crystalline solid); -13.4 kJ/mol (gas)
#TATB --smiles C1(=C(C(=C(C(=C1[N+](=O)[O-])N)[N+](=O)[O-])N)[N+](=O)[O-])N --enthalpy -36.78  wiki