import sys
import os
from pathlib import Path
from datetime import datetime
# Add the parent directory to the system path
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import util.utils as utils
# from utils import utils
import time
from util.enthalpy_predictor import predict_enthalpy

class TrainingLogger:
    def __init__(self, base_dir=None):
        if base_dir:
            self.log_dir = Path(base_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            base_dir = os.path.join("training_params", "VAE_H")
            self.exp_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_dir = os.path.join(base_dir, self.exp_time)
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化数据存储
        self.log_data = pd.DataFrame(columns=[
            'epoch', 'recon_loss', 'kld_loss',
            'total_loss', 'valid_rate'
        ])

        # 图表样式设置
        plt.style.use('seaborn')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    def log_metrics(self, epoch, metrics_dict):
        """记录单epoch指标"""
        new_row = pd.DataFrame([{
            'epoch': epoch,
            **metrics_dict
        }])
        self.log_data = pd.concat([self.log_data, new_row], ignore_index=True)

        # 实时保存到Excel
        excel_path = os.path.join(self.log_dir, 'training_log.xlsx')
        self.log_data.to_excel(excel_path, index=False)

    def plot_losses(self, epoch_interval=50):
        """绘制损失曲线并保存"""
        if len(self.log_data) == 0:
            return

        plt.figure(figsize=(12, 6), facecolor='white')  # 设置图表背景为白色

        # 绘制主损失曲线
        plt.subplot(1, 2, 1)
        for i, col in enumerate(['recon_loss', 'kld_loss']):
            line, = plt.plot(self.log_data['epoch'], self.log_data[col],
                             color=self.colors[i], label=col.replace('_', ' ').title())
            plt.setp(line, linewidth=2.0)  # 设置线条宽度为2.0，使线条更清晰
        plt.xlabel('Epoch', fontsize=12)  # 设置x轴标签字体大小为12
        plt.ylabel('Loss', fontsize=12)  # 设置y轴标签字体大小为12
        plt.legend(fontsize=12)  # 设置图例字体大小为12

        # 绘制验证率和总损失
        plt.subplot(1, 2, 2)
        ax1 = plt.gca()
        line1, = ax1.plot(self.log_data['epoch'], self.log_data['total_loss'],
                          color=self.colors[3], label='Total Loss')
        plt.setp(line1, linewidth=2.0)  # 设置线条宽度为2.0，使线条更清晰
        ax1.set_xlabel('Epoch', fontsize=12)  # 设置x轴标签字体大小为12
        ax1.set_ylabel('Loss', fontsize=12)  # 设置y轴标签字体大小为12

        ax2 = ax1.twinx()
        line2, = ax2.plot(self.log_data['epoch'], self.log_data['valid_rate'] * 100,
                          color='#9467bd', linestyle='--', label='Valid Rate (%)')
        plt.setp(line2, linewidth=2.0)  # 设置线条宽度为2.0，使线条更清晰
        ax2.set_ylabel('Validation Rate (%)', fontsize=12)  # 设置y轴标签字体大小为12

        plt.title(f"Training Progress @ Epoch {self.log_data['epoch'].max()}", fontsize=14)  # 设置标题字体大小为14
        plt.tight_layout()

        # 保存图片
        plot_path = os.path.join(self.log_dir,
                                 f"loss_plot_epoch_{self.log_data['epoch'].max()}.png")
        plt.savefig(plot_path)
        plt.close()


class Encoder(torch.nn.Module):
    def __init__(self, maxLength, num_vocabs, con_dims, fc_dims, latent_dim, state_fname, device) -> None:
        super().__init__()
        self.state_fname = state_fname
        self.device = device
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(maxLength * (num_vocabs-2),
                            fc_dims[0], device=self.device),
            torch.nn.ReLU()
        )
        for i in range(1, len(fc_dims)):  # [1,3)
            self.fc.append(torch.nn.Linear(
                fc_dims[i - 1], fc_dims[i], device=self.device))
            self.fc.append(torch.nn.ReLU())
        self.mu = torch.nn.Linear(fc_dims[-1], latent_dim, device=self.device)
        self.logvar = torch.nn.Linear(
            fc_dims[-1], latent_dim, device=self.device)

    def forward(self, X):
        X = self.fc(X)
        mu, logvar = self.mu(X), self.logvar(X)
        return self.reparameterize(mu, logvar), mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def loadState(self):
        if os.path.isfile(self.state_fname):
            self.load_state_dict(torch.load(self.state_fname))
        else:
            print("state file is not found")

    def saveState(self):
        dir_name = os.path.dirname(self.state_fname)
        utils.mkdir_multi(dir_name)
        torch.save(self.state_dict(), self.state_fname)


class Decoder(torch.nn.Module):
    def __init__(self, maxLength, num_vocabs, con_dims, latent_dim, hidden_dim, num_hidden, state_fname,
                 device) -> None:
        super().__init__()
        self.state_fname = state_fname
        self.maxLength = maxLength
        self.num_vacabs = num_vocabs
        self.device = device
        self.con_dims = con_dims
        self.gru = torch.nn.GRU(latent_dim, hidden_dim,
                                num_hidden, batch_first=True, device=self.device)
        self.fc = torch.nn.Linear(hidden_dim, num_vocabs, device=self.device)

    def forward(self, latent_vec, enthalpy, inp, freerun=False, randomchoose=True):
        latent_vec = latent_vec.to(self.device)  # (512, 64)
        enthalpy_ori = enthalpy
        enthalpy = enthalpy.unsqueeze(1).unsqueeze(2).expand(-1, self.maxLength, -1)  # (512, 128, 1)

        if not freerun:
            X = latent_vec.unsqueeze(1).expand(-1, self.maxLength, -1)
            X, _ = self.gru(X)
            return self.fc(X)
        else:
            out = torch.zeros((latent_vec.shape[0], self.maxLength, self.num_vacabs), dtype=torch.float32)
            X_latent = latent_vec.unsqueeze(1)
            X = X_latent
            shift = X_latent.shape[-1]
            hidden = None
            for i in range(self.maxLength):
                y, hidden = self.gru(X, hidden)
                y = torch.nn.functional.softmax(self.fc(y), dim=-1)
                if randomchoose:
                    selected = torch.multinomial(y.squeeze(1), 1).flatten()
                else:
                    selected = torch.argmax(y.squeeze(1), dim=1)
                X[:, 0, shift:] = 0
                for j in range(len(selected)):
                    X[j, 0, shift + selected[j]] = 1
                    out[j, i, selected[j]] = 1
            return out

    def loadState(self):
        if os.path.isfile(self.state_fname):
            self.load_state_dict(torch.load(self.state_fname))
        else:
            print("state file is not found")

    def saveState(self):
        dir_name = os.path.dirname(self.state_fname)
        utils.mkdir_multi(dir_name)
        torch.save(self.state_dict(), self.state_fname)


class ConVAE(object):
    def __init__(self, maxLength, num_vocabs, con_dims, fc_dims, latent_dim, hidden_dim, num_hidden,
                 encoder_state_fname, decoder_state_fname, device) -> None:
        self.latent_dim = latent_dim
        self.device = device
        self.encoder = Encoder(maxLength, num_vocabs, con_dims,
                               fc_dims, latent_dim, encoder_state_fname, device)
        self.decoder = Decoder(maxLength, num_vocabs, con_dims, latent_dim,
                               hidden_dim, num_hidden, decoder_state_fname, device)

    def reconstruction_quality_per_sample(self, X, enthalpy):
        self.encoder.eval()
        self.decoder.eval()

        # 1. 通过编码器获取隐空间表示
        latent_vec, mu, logvar = self.encoder(X)
        # 2. 通过解码器重构输入
        pred_y = self.decoder(mu, enthalpy, X)

        # 3. 将预测转换为one-hot形式
        pred_one_hot = torch.zeros_like(X, dtype=X.dtype) # 先生成一个和输入形状相同的张量，后续再通过预测值的argmax填充
        pred_y_argmax = torch.nn.functional.softmax(
            pred_y, dim=2).argmax(dim=2)
        for i in range(pred_one_hot.shape[0]):
            for j in range(pred_one_hot.shape[1]):
                pred_one_hot[i, j, pred_y_argmax[i, j]] = 1  # 将先前生成的全0矩阵按照预测结果赋值，形成one-hot

        # 4. 计算重构准确度
        diff = self.decoder.maxLength - torch.abs(pred_one_hot - X).sum(dim=-1).sum(dim=-1) * 0.5
        # 一个one-hot只有一个位置是1，预测出来不在同一个索引上相减会在两个索引上累计，所以乘0.5，表示有一个位置上重建的不对。
        # 用maxlength去减，就能看出在生成的字符串上有几个位置上重建出来的和原始X不同
        return diff

    def sample(self, nSample):
        latent_vec = torch.randn(
            (nSample, self.latent_dim), device=self.device)
        _enthalpy = torch.randn(nSample, device=self.device)
        y = self.decoder(latent_vec, _enthalpy, None, freerun=True)
        numVectors = y.argmax(dim=2) + 2
        return numVectors.cpu(), None

    def latent_space_quality(self, nSample, tokenizer=None):
        self.decoder.eval()
        # 从隐空间采样并生成分子
        numVectors, _ = self.sample(nSample)
        smilesStrs = tokenizer.getSmiles(numVectors)
        validSmilesStrs = []
        for sm in smilesStrs:
            if utils.isValidSmiles(sm):
                validSmilesStrs.append(sm)
        return len(validSmilesStrs)

    def calculate_enthalpy_loss(self, gen_smiles, cond_enthalpy, lb, ub):
        gt_enthalpy = cond_enthalpy
        predicted_enthalpy = torch.zeros(len(gen_smiles))
        valid_enthalpy = []
        _mask = torch.zeros(len(gen_smiles))
        _valid_id_mask = []

        for idx, smiles in enumerate(gen_smiles):
            if utils.isValidSmiles(smiles):
                predicted_value = predict_enthalpy(smiles)
                if predicted_value != 0:
                    predicted_enthalpy[idx] = predicted_value
                    _mask[idx] = 1
                    _valid_id_mask.append(idx)
                    valid_enthalpy.append(predicted_value)

        predicted_enthalpy_tensor = torch.tensor(predicted_enthalpy, device=self.device)
        valid_enthalpy_tensor = torch.tensor(valid_enthalpy, device=self.device)
        gt_enthalpy_tensor = torch.tensor(gt_enthalpy, device=self.device)
        gt_mask_enthalpy_tensor = torch.tensor([gt_enthalpy_tensor[i] for i in range(len(gt_enthalpy_tensor)) if _mask[i] == 1], device=self.device)

        if len(valid_enthalpy_tensor) > 0:
            normed_valid_enthalpy_tensor = (valid_enthalpy_tensor - lb) / (ub - lb)  # Reverse normalization
            cond_loss_mean = torch.nn.functional.mse_loss(normed_valid_enthalpy_tensor, gt_mask_enthalpy_tensor)  # 只计算有效样本的损失
        else:
            cond_loss_mean = torch.tensor(0.0, device=self.device)  # 如果没有有效样本，返回1

        return cond_loss_mean  # loss_per_sample_tensor, predicted_enthalpy_tensor, valid_enthalpy_tensor

    def loss_per_sample(self, pred_y, y, mu, logvar, gen_smiles, true_enthalpy=None, lb=None, ub=None):
        reconstruction_loss = torch.nn.functional.cross_entropy(
            pred_y.transpose(1, 2), y.transpose(1, 2), reduction='none').sum(dim=1)
        kld_loss = torch.sum(-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()), dim=1)
        return reconstruction_loss, kld_loss

    def trainModel(self, dataloader, encoderOptimizer, decoderOptimizer, encoderScheduler, decoderScheduler, KLD_alpha,
                   nepoch, tokenizer, printInterval, lb, ub, log_dir='outputs_data/VAE_H'):
        self.encoder.loadState()
        self.decoder.loadState()
        self.lb = lb
        self.ub = ub
        logger = TrainingLogger(base_dir=log_dir)
        minloss = None
        numSample = 100  # 训练过程中采样，用于计算valid数量
        for epoch in range(1, nepoch + 1):
            reconstruction_loss_list, accumulated_reconstruction_loss, kld_loss_list, accumulated_kld_loss, cond_loss_list, accumulated_cond_loss = [], 0, [], 0, [], 0
            quality_list, numValid_list = [], []
            for nBatch, (X, enthalpy) in enumerate(dataloader, 1):
                self.encoder.train()
                self.decoder.train()
                X = X.to(torch.float32)
                enthalpy = enthalpy.to(torch.float32)
                X = X.to(self.device)
                enthalpy = enthalpy.to(self.device)
                latent_vec, mu, logvar = self.encoder(X)
                pred_y = self.decoder(latent_vec, enthalpy, X)
                pred_one_hot = torch.zeros_like(X, dtype=X.dtype)
                pred_y_argmax = torch.nn.functional.softmax(pred_y, dim=2).argmax(dim=2)
                for i in range(pred_one_hot.shape[0]):
                    for j in range(pred_one_hot.shape[1]):
                        pred_one_hot[i, j, pred_y_argmax[i, j]] = 1
                predicted_indices = pred_one_hot.argmax(dim=2)
                predicted_smiles = tokenizer.getSmileshot(predicted_indices)
                reconstruction_loss, kld_loss= self.loss_per_sample(
                    pred_y, X, mu, logvar, predicted_smiles)
                reconstruction_mean, kld_mean = reconstruction_loss.mean(), kld_loss.mean() * \
                                                           KLD_alpha
                loss = reconstruction_mean + kld_mean
                encoderOptimizer.zero_grad()
                decoderOptimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1)
                loss.backward()
                encoderOptimizer.step()
                decoderOptimizer.step()

                reconstruction_loss_list.append(reconstruction_mean.item())
                kld_loss_list.append(kld_mean.item())
                accumulated_reconstruction_loss += reconstruction_mean.item()
                accumulated_kld_loss += kld_mean.item()

                if (nBatch == 1 or nBatch % printInterval == 0):
                    quality = self.reconstruction_quality_per_sample(X, enthalpy).mean()
                    numValid = self.latent_space_quality(numSample, tokenizer)
                    quality_list.append(quality)
                    numValid_list.append(numValid)
                    print("[%s] Epoch %4d & Batch %4d: Reconstruction_Loss= %.5e KLD_Loss= %.5e Quality= %3d/%3d Valid= %3d/%3d" % (time.ctime(), epoch, nBatch, sum(
                        reconstruction_loss_list) / len(reconstruction_loss_list), sum(kld_loss_list) / len(kld_loss_list), quality, self.decoder.maxLength, numValid, numSample))
                    reconstruction_loss_list.clear()
                    kld_loss_list.clear()
                    if minloss is None:
                        minloss = loss.item()
                    elif loss.item() < minloss:
                        self.encoder.saveState()
                        self.decoder.saveState()
                        minloss = loss.item()
            encoderScheduler.step()
            decoderScheduler.step()
            reconstruction_loss_list = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in
                                        reconstruction_loss_list]
            kld_loss_list = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in kld_loss_list]
            total_loss_list = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in
                               total_loss_list]
            quality_list = [quality.cpu().numpy() if isinstance(quality, torch.Tensor) else quality for quality in
                            quality_list]
            numValid_list = [numValid.cpu().numpy() if isinstance(numValid, torch.Tensor) else numValid for numValid in
                             numValid_list]
            avg_recon_loss = np.mean(reconstruction_loss_list)
            avg_kld_loss = np.mean(kld_loss_list)
            avg_total_loss = np.mean(total_loss_list)
            avg_quality = np.mean(quality_list)
            avg_valid_rate = np.mean(numValid_list) / numSample

            metrics = {
                'recon_loss': avg_recon_loss,
                'kld_loss': avg_kld_loss,
                'total_loss': avg_total_loss,
                'valid_rate': avg_valid_rate
            }
            logger.log_metrics(epoch, metrics)

            print(
                "[%s] Epoch %4d: Reconstruction_Loss= %.5e KLD_Loss= %.5e Quality= %3d/%3d Valid= %3d/%3d" % (
                time.ctime(), epoch, accumulated_reconstruction_loss / nBatch, accumulated_kld_loss / nBatch,
                sum(quality_list) / len(quality_list), self.decoder.maxLength,
                sum(numValid_list) / len(numValid_list), numSample))
