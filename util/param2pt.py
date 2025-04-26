import argparse
import os
import sys
import torch
import numpy as np
import model.convae_310_resnet_seed_logger_multiseed as cvae
from dataset.dataset import SmilesDictDataset
from util.enthalpy_predictor import predict_enthalpy
from util.tokens import getTokenizer  # 根据你的实际导入路径调整
import util.utils as utils
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Molecular Generation with CVAE')
    parser.add_argument('--random', action='store_true', help='Random generation')
    # parser.add_argument('--conditional', type=bool, default='True', help='Whether to generate conditional')
    parser.add_argument('--smiles', type=str, default='', help='Input SMILES for conditional generation')
    parser.add_argument('--enthalpy', type=float, default=None, help='Target enthalpy value')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--info_output', type=str, default='generate_smi/generated_info.txt', help='Output file path')
    parser.add_argument('--output', type=str, default='generate_smi/generated_smiles.smi',
                        help='Output file path')
    args = parser.parse_args()
    # 加载配置
    config = utils.config
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
    torch.save(vae_model.encoder.state_dict(), r"D:\Project\VAE_Related\ConVAE/parameters/encoder.pt")
    torch.save(vae_model.decoder.state_dict(), r"D:\Project\VAE_Related\ConVAE/parameters/decoder.pt")



if __name__ == "__main__":
    main()
