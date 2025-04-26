import sys
import os
# 将 ConVAE 目录添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.dataset import SmilesDataset
from util.tokens import Tokenizer
import util.utils as utils
import torch
# import rnn
# import vae
import model.CVAE_HC as cvae
import argparse
import multiprocessing

# parser = argparse.ArgumentParser()
# parser.add_argument('-m', '--model', type=str)
# parser.add_argument('-i', '--printInterval', type=int)
# parser.add_argument('--gpu', action='store_true', default=False)
# args = parser.parse_args()
# model_type = args.model
# printInterval = args.printInterval
# useGPU = args.gpu and torch.cuda.is_available()
def main():
    model = 'cvae_hc'
    printInterval = 40
    useGPU = True
    device = torch.device('cuda' if useGPU else 'cpu')
    print(f'useGPU = {useGPU}')

    tokenizer = utils.get_tokenizer(model)
    print(tokenizer.tokensDict)
    cfg = utils.__load_config(model=model)#需要在tokenizer后面实例化，因为tokenizer会自动添加一行num_vocab数据
    smilesDataset = SmilesDataset(
        cfg['fname_dataset'], tokenizer, cfg['maxLength'])
    lb, ub = smilesDataset._getbound()
    # if model_type == 'cvae':
    smilesDataloader = torch.utils.data.DataLoader(
        smilesDataset, batch_size=cfg['batch_size'], 
        shuffle=True, num_workers=4) 
        # collate_fn=smilesDataset.one_hot_collate_fn)

    vae_model = cvae.ConVAE(**cfg['vae_param'],
                        encoder_state_fname=cfg['fname_vae_encoder_parameters'],
                        decoder_state_fname=cfg['fname_vae_decoder_parameters'],
                        device=device)

    for name, layer in vae_model.encoder.named_parameters():
        print(name, layer.shape, layer.dtype,
                layer.requires_grad, layer.device)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    for name, layer in vae_model.decoder.named_parameters():
        print(name, layer.shape, layer.dtype,
                layer.requires_grad, layer.device)

    encoderOptimizer = torch.optim.Adam(
        vae_model.encoder.parameters(), 
        lr=cfg['lr'], 
        weight_decay=1.0e-5, # 权重衰减，惩罚过大的权重
        eps=1e-8
    )
    decoderOptimizer = torch.optim.Adam(
        vae_model.decoder.parameters(), 
        lr=cfg['lr'], 
        weight_decay=1.0e-5,
        eps=1e-8
    )
    encoderScheduler = torch.optim.lr_scheduler.StepLR(
        encoderOptimizer, step_size=5, gamma=0.95) # 每经过x个epoch，学习率乘以y
    decoderScheduler = torch.optim.lr_scheduler.StepLR(
        decoderOptimizer, step_size=5, gamma=0.95)
    
    vae_model.trainModel(smilesDataloader, encoderOptimizer, decoderOptimizer, 
                        encoderScheduler, decoderScheduler, 1.0, 
                        cfg['num_epoch'], tokenizer, printInterval, lb, ub)

    # elif model_type == 'rnn':
    #     smilesDataloader = torch.utils.data.DataLoader(
    #         smilesDataset, batch_size=cfg['batch_size'], 
    #         shuffle=True, num_workers=4, 
    #         collate_fn=smilesDataset.collate_fn)
        
    #     rnn_model = rnn.RNN(
    #         **cfg['rnn_param'], 
    #         state_fname=cfg['fname_rnn_parameters'], 
    #         device=device)
        
    #     for name, layer in rnn_model.named_parameters():
    #         print(name, layer.shape, layer.dtype,
    #               layer.requires_grad, layer.device)
        
    #     optimizer = torch.optim.Adam(
    #         rnn_model.parameters(), lr=cfg['lr'], weight_decay=1.0e-4)
    #     scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer, step_size=1, gamma=0.95)
        
    #     rnn_model.trainModel(smilesDataloader, optimizer, scheduler,
    #                         cfg['num_epoch'], cfg['maxLength'], 
    #                         tokenizer, printInterval)

if __name__ == '__main__':
    # from multiprocessing import freeze_support
    # freeze_support()
    # 如果要冻结成exe还需要加这行
    # multiprocessing.freeze_support()
    # 保护程序入口，防止在 Windows 上使用 multiprocessing 时出错
    multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.freeze_support()
    main()
