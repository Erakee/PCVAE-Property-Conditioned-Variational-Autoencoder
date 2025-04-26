import sys
import os
import random
# 将 ConVAE 目录添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from dataset.dataset import SmilesDictDataset
import util.utils as utils
from util.utils import temp_seed
import torch
import model.CVAE_DHR as cvae
import multiprocessing
import numpy as np


# 定义种子范围
SEEDS = [42]  # range(100, 110)  # 20-29共10个种子

def run_experiment(seed):
    model = 'cvae_dhr'
    # 创建种子专用目录
    exp_dir = Path(f"training_params/CVAE_DHR/seed_{seed}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 设置全局随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 初始化组件（保持与原始main函数相同）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with temp_seed(seed):
        tokenizer = utils.get_tokenizer(model)
        cfg = utils.__load_config(model=model)
        smilesDataset = SmilesDictDataset(
            cfg['fname_dataset'], tokenizer, cfg['maxLength'])
        lb, ub = smilesDataset._getbound()

        # 修改DataLoader初始化使用固定种子
        generator = torch.Generator().manual_seed(seed)
        smilesDataloader = torch.utils.data.DataLoader(
            smilesDataset,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=4,
            drop_last=True,
            collate_fn=SmilesDictDataset.collate_fn,
            generator=generator
        )

        # 初始化模型（修改保存路径）
        vae_model = cvae.ConVAE(
            **cfg['vae_param'],
            encoder_state_fname=str(exp_dir / "encoder.pt"),
            decoder_state_fname=str(exp_dir / "decoder.pt"),
            device=device
        )

    # 修改训练配置
    encoderOptimizer = torch.optim.AdamW(
        vae_model.encoder.parameters(),
        lr=cfg['lr'],
        weight_decay=1e-6,
        eps=1e-9
    )
    decoderOptimizer = torch.optim.Adam(
        vae_model.decoder.parameters(),
        lr=cfg['lr'],
        weight_decay=1e-6,
        eps=1e-9
    )

    encoderScheduler = torch.optim.lr_scheduler.StepLR(
        encoderOptimizer, step_size=5, gamma=0.7)
    decoderScheduler = torch.optim.lr_scheduler.StepLR(
        decoderOptimizer, step_size=5, gamma=0.7)

    # 运行训练（假设已修改trainModel支持自定义logger）
    vae_model.trainModel(
        smilesDataloader, encoderOptimizer, decoderOptimizer,
        encoderScheduler, decoderScheduler, 1.0,
        cfg['num_epoch'], tokenizer,
        printInterval=50, lb=lb, ub=ub, seed=seed,
        log_dir=str(exp_dir))  # 添加log_dir参数


def main():
    multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.freeze_support()

    # 并行运行所有种子实验（建议按顺序运行以避免资源竞争）
    for seed in SEEDS:
        print(f"\n{'=' * 40}")
        print(f"Running experiment with seed: {seed}")
        print(f"{'=' * 40}")
        run_experiment(seed)


if __name__ == '__main__':
    main()

# def main():
#     # 设置全局随机种子
#     seed = 20
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#
#     # 如果使用 GPU，还需要设置以下内容
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
#     printInterval = 50
#     useGPU = True
#     device = torch.device('cuda' if useGPU else 'cpu')
#     print(f'useGPU = {useGPU}')
#
#     # 在创建 tokenizer 时使用 temp_seed 确保可重复性
#     with temp_seed(seed):
#         tokenizer = utils.get_tokenizer()
#     print(tokenizer.tokensDict)
#
#     # 在创建 SmilesDictDataset 时使用 temp_seed 确保可重复性
#     with temp_seed(seed):
#         smilesDataset = SmilesDictDataset(
#             cfg['fname_dataset'], tokenizer, cfg['maxLength'])
#     lb, ub = smilesDataset._getbound()
#
#     # 在创建 DataLoader 时使用 temp_seed 确保数据打乱顺序一致
#     with temp_seed(seed):
#         smilesDataloader = torch.utils.data.DataLoader(
#             smilesDataset, batch_size=cfg['batch_size'],
#             shuffle=True, num_workers=4, drop_last=True, collate_fn=SmilesDictDataset.collate_fn)
#
#     # 在创建模型时使用 temp_seed 确保模型初始化一致
#     with temp_seed(seed):
#         vae_model = cvae.ConVAE(**cfg['vae_param'],
#                                 encoder_state_fname=cfg['fname_vae_encoder_parameters'],
#                                 decoder_state_fname=cfg['fname_vae_decoder_parameters'],
#                                 device=device)
#
#     for name, layer in vae_model.encoder.named_parameters():
#         print(name, layer.shape, layer.dtype,
#               layer.requires_grad, layer.device)
#     for name, layer in vae_model.decoder.named_parameters():
#         print(name, layer.shape, layer.dtype,
#               layer.requires_grad, layer.device)
#
#     # 在创建优化器时使用 temp_seed 确保优化器初始化一致
#     with temp_seed(seed):
#         encoderOptimizer = torch.optim.AdamW(
#             vae_model.encoder.parameters(),
#             lr=cfg['lr'],
#             weight_decay=1e-6,
#             eps=1e-9
#         )
#         decoderOptimizer = torch.optim.Adam(
#             vae_model.decoder.parameters(),
#             lr=cfg['lr'],
#             weight_decay=1e-6,
#             eps=1e-9
#         )
#
#     # 在创建学习率调度器时使用 temp_seed 确保调度器初始化一致
#     with temp_seed(seed):
#         encoderScheduler = torch.optim.lr_scheduler.StepLR(
#             encoderOptimizer, step_size=5, gamma=0.95)  # 每经过x个epoch，学习率乘以y
#         decoderScheduler = torch.optim.lr_scheduler.StepLR(
#             decoderOptimizer, step_size=5, gamma=0.95)
#
#     vae_model.trainModel(smilesDataloader, encoderOptimizer, decoderOptimizer,
#                          encoderScheduler, decoderScheduler, 1.0,
#                          cfg['num_epoch'], tokenizer, printInterval, lb, ub, seed)
#
#
# if __name__ == '__main__':
#     # from multiprocessing import freeze_support
#     # freeze_support()
#     # 如果要冻结成exe还需要加这行
#     # multiprocessing.freeze_support()
#     # 保护程序入口，防止在 Windows 上使用 multiprocessing 时出错
#     multiprocessing.set_start_method('spawn', force=True)
#     multiprocessing.freeze_support()
#     main()
