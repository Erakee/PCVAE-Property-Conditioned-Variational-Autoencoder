import sys
import os
from pathlib import Path

import pandas as pd
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import util.utils as utils
import time
from util.enthalpy_predictor import predict_enthalpy

class TrainingLogger:
    def __init__(self, base_dir=None):
        if base_dir:
            self.log_dir = Path(base_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            base_dir = r"D:\Project\VAE_Related\ConVAE\Results4Paper\outputs_data\VAE_H"
            self.exp_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_dir = os.path.join(base_dir, self.exp_time)
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_data = pd.DataFrame(columns=[
            'epoch', 'recon_loss', 'kld_loss',
            'total_loss', 'valid_rate'
        ])

        # 图表样式设置
        plt.style.use('seaborn')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    def log_metrics(self, epoch, metrics_dict):
        new_row = pd.DataFrame([{
            'epoch': epoch,
            **metrics_dict
        }])
        self.log_data = pd.concat([self.log_data, new_row], ignore_index=True)

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
    def __init__(self, maxLength, num_vocabs, con_dims, fc_dims, latent_dim,
                 state_fname, device) -> None:
        super().__init__()
        self.num_vocabs = num_vocabs
        self.state_fname = state_fname
        self.device = device
        input_dim = maxLength * num_vocabs
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim, fc_dims[0], device=self.device),
            torch.nn.ReLU()
        )
        for i in range(1, len(fc_dims)):  # [1,3)
            self.fc.append(torch.nn.Linear(fc_dims[i - 1], fc_dims[i], device=self.device))
            self.fc.append(torch.nn.ReLU())
        self.mu = torch.nn.Linear(fc_dims[-1], latent_dim, device=self.device)
        self.logvar = torch.nn.Linear(
            fc_dims[-1], latent_dim, device=self.device)

    def forward(self, X):
        X = self.fc(X.flatten(start_dim=1))  # (256, 64) #.flatten(start_dim=1) hou mian jia de
        mu, logvar = self.mu(X), self.logvar(X)  # (256, 64)
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
        self.num_vocabs = num_vocabs
        self.device = device
        self.gru = torch.nn.GRU(latent_dim, hidden_dim, num_hidden, batch_first=True, device=self.device)
        self.fc = torch.nn.Linear(hidden_dim, self.num_vocabs, device=self.device)

    def forward(self, latent_vec, freerun=False, randomchoose=True):  # decoder(latent_vec, enthalpy, X)
        latent_vec = latent_vec.to(self.device)  # (256, 64)
        batch_size = latent_vec.size(0)

        if not freerun:
            # Teacher-forcing 训练模式
            # 将 latent_vec 扩展为序列输入 (batch, seq_len, latent_dim)
            latent_expanded = latent_vec.unsqueeze(1).expand(-1, self.maxLength, -1)

            # GRU处理
            output, _ = self.gru(latent_expanded)  # output形状: (batch, seq_len, hidden_dim)

            # 全连接层输出
            pred_logits = self.fc(output)  # (batch, seq_len, num_vocabs)
            return pred_logits
        else:
            # freerun模式（自回归生成）
            outputs = []
            input_step = latent_vec.unsqueeze(1)  # (batch, 1, latent_dim)
            hidden = None  # 初始没有hidden状态

            for _ in range(self.maxLength):
                # GRU一次处理一步
                output, hidden = self.gru(input_step, hidden)  # output: (batch, 1, hidden_dim)

                # 映射到词表空间
                logits = self.fc(output.squeeze(1))  # (batch, num_vocabs)

                if randomchoose:
                    # 根据概率分布采样
                    prob = torch.softmax(logits, dim=-1)
                    sampled_token = torch.multinomial(prob, num_samples=1).squeeze(1)  # (batch,)
                else:
                    # 直接取最大概率
                    sampled_token = torch.argmax(logits, dim=-1)  # (batch,)

                outputs.append(logits.unsqueeze(1))  # 把logits（不是sampled_token）存起来供后面计算loss

                # 准备下一个时间步的输入
                # 把 sampled_token 转成 one-hot，再映射回 latent_dim 空间
                # 这里为了简单，直接将 latent_vec 作为每步输入，也可以改得更复杂（比如根据sampled token编码）
                input_step = latent_vec.unsqueeze(1)  # (batch, 1, latent_dim)

            outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, num_vocabs)
            return outputs

            # # 生成模式：自回归生成
            # output_sequence = torch.zeros(
            #     (batch_size, self.maxLength, self.num_vocabs),
            #     device=self.device
            # )
            #
            # # 初始隐藏状态（与GRU层数一致）
            # hidden = torch.zeros(
            #     self.gru.num_layers,
            #     batch_size,
            #     self.gru.hidden_size,
            #     device=self.device
            # )
            #
            # # 初始输入：潜在向量作为首个时间步输入
            # current_input = latent_vec.unsqueeze(1)  # (batch, 1, latent_dim)
            #
            # for t in range(self.maxLength):
            #     gru_output, hidden = self.gru(current_input, hidden)
            #     logits = self.fc(gru_output.squeeze(1))  # (batch, num_vocabs)
            #     probs = torch.nn.functional.softmax(logits, dim=-1)
            #
            #     # 选择下一个token
            #     if randomchoose:
            #         selected = torch.multinomial(probs, 1)  # (batch, 1)
            #     else:
            #         selected = torch.argmax(probs, dim=-1, keepdim=True)
            #
            #     # 生成one-hot并更新输入
            #     one_hot = torch.zeros_like(probs)
            #     one_hot.scatter_(1, selected, 1.0)
            #     output_sequence[:, t, :] = one_hot
            #     current_input = latent_vec.unsqueeze(1)  # 持续使用潜在向量
            #
            # return output_sequence
            #
            # out = torch.zeros((latent_vec.shape[0], self.maxLength, self.num_vocabs), dtype=torch.float32)
            # X_latent = latent_vec.unsqueeze(1)  # (256, 1, 64)
            # X = torch.concat([X_latent, torch.zeros((latent_vec.shape[0], 1, self.num_vocabs), dtype=torch.float32,
            #                                         device=self.device)], dim=2)  # (256, 1, 64+17)
            #
            # X = torch.concat(
            #     [X, torch.zeros((latent_vec.shape[0], 1, self.con_dims), dtype=torch.float32, device=self.device)],
            #     dim=2)
            # shift = X_latent.shape[-1]
            # hidden = None
            # for i in range(self.maxLength):
            #     y, hidden = self.gru(X, hidden)
            #     y = torch.nn.functional.softmax(self.fc(y), dim=-1)
            #     if randomchoose:
            #         selected = torch.multinomial(y.squeeze(1), 1).flatten()
            #     else:
            #         selected = torch.argmax(y.squeeze(1), dim=1)
            #     X[:, 0, shift:] = 0
            #     for j in range(len(selected)):
            #         X[j, 0, shift + selected[j]] = 1
            #         out[j, i, selected[j]] = 1
            # return out

    def loadState(self):
        if os.path.isfile(self.state_fname):
            self.load_state_dict(torch.load(self.state_fname))
        else:
            print("state file is not found")

    def saveState(self):
        dir_name = os.path.dirname(self.state_fname)
        utils.mkdir_multi(dir_name)
        torch.save(self.state_dict(), self.state_fname)


class VAE(object):
    def __init__(self, maxLength, num_vocabs, con_dims, fc_dims, latent_dim, hidden_dim, num_hidden,
                 encoder_state_fname, decoder_state_fname, device) -> None:
        self.latent_dim = latent_dim
        self.device = device
        self.encoder = Encoder(maxLength, num_vocabs, con_dims,
                               fc_dims, latent_dim, encoder_state_fname, device)
        self.decoder = Decoder(maxLength, num_vocabs, con_dims, latent_dim,
                               hidden_dim, num_hidden, decoder_state_fname, device)

    def reconstruction_quality_per_sample(self, X):
        self.encoder.eval()
        self.decoder.eval()

        # 1. 通过编码器获取隐空间表示aaaa
        latent_vec, _, _ = self.encoder(X)
        # 2. 通过解码器重构输入
        pred_y = self.decoder(latent_vec)
        # reconstruction_loss, kld_loss = self.loss_per_sample(pred_y, X, mu, logvar)

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
    # def reconstruction_quality_per_sample(self, X):
    #     with torch.no_grad():
    #         latent_vec, _, _ = self.encoder(X)
    #         pred_y = self.decoder(latent_vec)  # (batch, seq_len=64, vocab_size=17)
    #         pred_indices = torch.nn.functional.softmax(pred_y, dim=2).argmax(dim=2)  # (batch, seq_len)
    #
    #         true_indices = X.argmax(dim=2)  # (batch, seq_len)
    #
    #         correct = (pred_indices == true_indices).float()  # (batch, seq_len)
    #         quality_per_sample = correct.mean(dim=1)  # (batch,)
    #
    #     return quality_per_sample.cpu()

    def sample(self, nSample):
        latent_vec = torch.randn((nSample, self.latent_dim), device=self.device)
        # 通过解码器生成分子
        y = self.decoder(latent_vec, freerun=True)
        numVectors = y.argmax(dim=2) + 2 #+2是因为0和1对应起始和结束的索引，隐空间拟合时不包含这两个，采样后转换为实际分子时需补上
        return numVectors.cpu(), None

    def latent_space_quality(self, nSample, tokenizer=None):
        self.decoder.eval()
        # 从隐空间采样并生成分子
        numVectors, _ = self.sample(nSample)# 采样得到用于表示分子的数字序列
        # 将数字序列转换回SMILES字符串
        smilesStrs = tokenizer.getSmiles(numVectors)
        # 检查每个SMILES的有效性
        validSmilesStrs = []
        for sm in smilesStrs:
            if utils.isValidSmiles(sm):
                validSmilesStrs.append(sm)
        # print("ValidSmilesStrs: %s" % (validSmilesStrs,))
        return len(validSmilesStrs)


    # def loss_per_sample(self, pred_y, y, mu, logvar, gen_smiles):
    #     reconstruction_loss = torch.nn.functional.cross_entropy(  # 都是(256, 64, 17)
    #         pred_y.transpose(1, 2), y.transpose(1, 2), reduction='none').sum(dim=1)  # 输出结果是个tensor(256)
    #     kld_loss = torch.sum(-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()), dim=1)  # 输出结果是个tensor(256)
    #     return reconstruction_loss, kld_loss

    def loss_per_sample(self, pred_y, y, mu, logvar):
        y_label = y.argmax(dim=2)  # one-hot还原成类别index
        reconstruction_loss = torch.nn.functional.cross_entropy(
            pred_y.transpose(1, 2), y_label, reduction='none'
        ).sum(dim=1)  # sum over sequence length
        kld_loss = torch.sum(
            -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()), dim=1
        )
        return reconstruction_loss, kld_loss

    # def trainModel(self, dataloader, encoderOptimizer, decoderOptimizer, encoderScheduler, decoderScheduler, KLD_alpha,
    #                nepoch, tokenizer, printInterval, lb, ub, seed, log_dir='training_params/VAE_H/default_dir'):
    #     self.encoder.loadState()
    #     self.decoder.loadState()
    #     self.lb = lb
    #     self.ub = ub
    #     logger = TrainingLogger(base_dir=log_dir)
    #     minloss = None
    #     numSample = 100  # 训练过程中采样，用于计算valid数量
    #     scheduler_count = 0
    #
    #     for epoch in range(1, nepoch + 1):
    #         reconstruction_loss_list, accumulated_reconstruction_loss, kld_loss_list, accumulated_kld_loss, total_loss_list, accumulated_total_loss = [], 0, [], 0, [], 0
    #         quality_list, numValid_list = [], []
    #
    #         for nBatch, (X, _) in enumerate(dataloader, 1):
    #             self.encoder.train()
    #             self.decoder.train()
    #
    #             X = X.to(torch.float32)  # 将 X 转换为 float32
    #             X = X.to(self.device)
    #
    #             latent_vec, mu, logvar = self.encoder(X)  # (256, 64)
    #             pred_y = self.decoder(latent_vec)  # (256, 64,17)  # 对比时改成latent_vec
    #             pred_one_hot = torch.zeros_like(X, dtype=X.dtype)
    #             pred_y_argmax = torch.nn.functional.softmax(pred_y, dim=2).argmax(dim=2)
    #             for i in range(pred_one_hot.shape[0]):
    #                 for j in range(pred_one_hot.shape[1]):
    #                     pred_one_hot[i, j, pred_y_argmax[i, j]] = 1
    #             predicted_indices = pred_one_hot.argmax(dim=2)  # 处理后的索引 (0~16)
    #             predicted_indices_original = predicted_indices + 2  # 恢复为原始索引 (2~18)
    #             predicted_smiles = tokenizer.getSmiles(predicted_indices_original)
    #
    #             reconstruction_loss, kld_loss= self.loss_per_sample(
    #                 pred_y, X, mu, logvar, predicted_smiles)
    #             reconstruction_mean, kld_mean= reconstruction_loss.mean(), kld_loss.mean() * KLD_alpha
    #
    #             total_loss = reconstruction_mean + kld_mean
    #
    #             encoderOptimizer.zero_grad()
    #             decoderOptimizer.zero_grad()
    #             torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1)
    #             torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1)
    #             total_loss.backward()
    #             encoderOptimizer.step()
    #             decoderOptimizer.step()
    #
    #             reconstruction_loss_list.append(reconstruction_mean.item())
    #             kld_loss_list.append(kld_mean.item())
    #             total_loss_list.append(total_loss.item())
    #
    #             accumulated_reconstruction_loss += reconstruction_mean.item()
    #             accumulated_kld_loss += kld_mean.item()
    #             accumulated_total_loss += total_loss.item()
    #             if (nBatch == 1 or nBatch % printInterval == 0):
    #                 quality = self.reconstruction_quality_per_sample(X).mean()
    #                 numValid = self.latent_space_quality(numSample, tokenizer)
    #                 quality_list.append(quality)
    #                 numValid_list.append(numValid)
    #                 print("[%s] Epoch %4d & Batch %4d: Reconstruction_Loss= %.5e KLD_Loss= %.5e Quality= %3d/%3d Valid= %3d/%3d" % (time.ctime(), epoch, nBatch, sum(
    #                     reconstruction_loss_list) / len(reconstruction_loss_list), sum(kld_loss_list) / len(kld_loss_list), quality, self.decoder.maxLength, numValid, numSample))
    #                 reconstruction_loss_list.clear()
    #                 kld_loss_list.clear()
    #                 if minloss is None:
    #                     minloss = total_loss.item()
    #                 elif total_loss.item() < minloss:
    #                     self.encoder.saveState()
    #                     self.decoder.saveState()
    #                     minloss = total_loss.item()
    #         encoderScheduler.step()
    #         decoderScheduler.step()
    #
    #         # 计算 epoch 平均指标
    #         # 转换为 CPU 张量并转换为 NumPy 数组
    #         reconstruction_loss_list = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in
    #                                     reconstruction_loss_list]
    #         kld_loss_list = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in kld_loss_list]
    #         total_loss_list = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in
    #                            total_loss_list]
    #         quality_list = [quality.cpu().numpy() if isinstance(quality, torch.Tensor) else quality for quality in
    #                         quality_list]
    #         numValid_list = [numValid.cpu().numpy() if isinstance(numValid, torch.Tensor) else numValid for numValid in
    #                          numValid_list]
    #         avg_recon_loss = np.mean(reconstruction_loss_list)
    #         avg_kld_loss = np.mean(kld_loss_list)
    #         avg_total_loss = np.mean(total_loss_list)
    #         avg_quality = np.mean(quality_list)
    #         avg_valid_rate = np.mean(numValid_list) / numSample
    #
    #         # 打印 epoch 总结
    #         Enc_lr = encoderOptimizer.param_groups[0]['lr']
    #         Dec_lr = decoderOptimizer.param_groups[0]['lr']
    #         print(
    #             f"[{time.ctime()}] Epoch {epoch:4d}: "
    #             f"Reconstruction_Loss= {avg_recon_loss:.5e} "
    #             f"KLD_Loss= {avg_kld_loss:.5e} "
    #             f"Total_Loss= {avg_total_loss:.5e} "
    #             f"Quality= {avg_quality:.0f}/{self.decoder.maxLength} "
    #             f"Valid= {avg_valid_rate * 100:.1f}% "
    #             f"Enc_lr= {Enc_lr:.5e} "
    #             f"Dec_lr= {Dec_lr:.5e} "
    #         )
    #
    #         # 记录指标
    #         metrics = {
    #             'recon_loss': avg_recon_loss,
    #             'kld_loss': avg_kld_loss,
    #             'total_loss': avg_total_loss,
    #             'valid_rate': avg_valid_rate
    #         }
    #         logger.log_metrics(epoch, metrics)
    #
    #         # 定期生成图表
    #         if epoch % 50 == 0:
    #             logger.plot_losses()
    #
    #         if (avg_valid_rate > 0.25) and (scheduler_count == 0):
    #             encoderOptimizer.param_groups[0]['lr'] = 1e-5
    #             decoderOptimizer.param_groups[0]['lr'] = 1e-5
    #             scheduler_count = 1
    def trainModel(self, dataloader, encoderOptimizer, decoderOptimizer, encoderScheduler, decoderScheduler, KLD_alpha,
                   nepoch, tokenizer, printInterval, lb, ub, seed, log_dir='training_params/VAE_H/default_dir'):
        self.encoder.loadState()
        self.decoder.loadState()
        self.lb = lb
        self.ub = ub
        logger = TrainingLogger(base_dir=log_dir)
        minloss = None
        numSample = 100
        scheduler_count = 0

        for epoch in range(1, nepoch + 1):
            accumulated_reconstruction_loss, accumulated_kld_loss, accumulated_total_loss = 0, 0, 0
            quality_list, numValid_list = [], []
            nbatches = 0  # 用于最后取平均

            for nBatch, (X, _) in enumerate(dataloader, 1):
                self.encoder.train()
                self.decoder.train()

                X = X.to(torch.float32).to(self.device)

                latent_vec, mu, logvar = self.encoder(X)
                pred_y = self.decoder(latent_vec)
                pred_one_hot = torch.zeros_like(X, dtype=X.dtype)
                pred_y_argmax = torch.nn.functional.softmax(pred_y, dim=2).argmax(dim=2)
                for i in range(pred_one_hot.shape[0]):
                    for j in range(pred_one_hot.shape[1]):
                        pred_one_hot[i, j, pred_y_argmax[i, j]] = 1
                predicted_indices = pred_one_hot.argmax(dim=2)
                predicted_indices_original = predicted_indices + 2
                predicted_smiles = tokenizer.getSmiles(predicted_indices_original)

                reconstruction_loss, kld_loss = self.loss_per_sample(pred_y, X, mu, logvar)
                reconstruction_mean, kld_mean = reconstruction_loss.mean(), kld_loss.mean() * KLD_alpha
                total_loss = reconstruction_mean + kld_mean

                encoderOptimizer.zero_grad()
                decoderOptimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1)
                total_loss.backward()
                encoderOptimizer.step()
                decoderOptimizer.step()

                accumulated_reconstruction_loss += reconstruction_mean.item()
                accumulated_kld_loss += kld_mean.item()
                accumulated_total_loss += total_loss.item()
                nbatches += 1

                if (nBatch == 1 or nBatch % printInterval == 0):
                    quality = self.reconstruction_quality_per_sample(X).mean()
                    numValid = self.latent_space_quality(numSample, tokenizer)
                    quality_list.append(quality)
                    numValid_list.append(numValid)

                    avg_recon = accumulated_reconstruction_loss / nbatches
                    avg_kld = accumulated_kld_loss / nbatches

                    print(
                        "[%s] Epoch %4d & Batch %4d: Reconstruction_Loss= %.5e KLD_Loss= %.5e Quality= %3d/%3d Valid= %3d/%3d" % (
                            time.ctime(), epoch, nBatch, avg_recon, avg_kld, quality, self.decoder.maxLength, numValid,
                            numSample))

                    # 更新最优模型
                    if minloss is None or total_loss.item() < minloss:
                        self.encoder.saveState()
                        self.decoder.saveState()
                        minloss = total_loss.item()

            encoderScheduler.step()
            decoderScheduler.step()

            # 计算 epoch 平均指标
            avg_recon_loss = accumulated_reconstruction_loss / nbatches
            avg_kld_loss = accumulated_kld_loss / nbatches
            avg_total_loss = accumulated_total_loss / nbatches
            avg_quality = np.mean(quality_list) if quality_list else 0
            avg_valid_rate = np.mean(numValid_list) / numSample if numValid_list else 0

            # 打印 epoch 总结
            Enc_lr = encoderOptimizer.param_groups[0]['lr']
            Dec_lr = decoderOptimizer.param_groups[0]['lr']
            print(
                f"[{time.ctime()}] Epoch {epoch:4d}: "
                f"Reconstruction_Loss= {avg_recon_loss:.5e} "
                f"KLD_Loss= {avg_kld_loss:.5e} "
                f"Total_Loss= {avg_total_loss:.5e} "
                f"Quality= {avg_quality:.0f}/{self.decoder.maxLength} "
                f"Valid= {avg_valid_rate * 100:.1f}% "
                f"Enc_lr= {Enc_lr:.5e} "
                f"Dec_lr= {Dec_lr:.5e} "
            )

            # 记录指标
            metrics = {
                'recon_loss': avg_recon_loss,
                'kld_loss': avg_kld_loss,
                'total_loss': avg_total_loss,
                'valid_rate': avg_valid_rate
            }
            logger.log_metrics(epoch, metrics)

            # 定期绘图
            if epoch % 50 == 0:
                logger.plot_losses()

            # 手动调整学习率
            if (avg_valid_rate > 0.25) and (scheduler_count == 0):
                encoderOptimizer.param_groups[0]['lr'] = 1e-5
                decoderOptimizer.param_groups[0]['lr'] = 1e-5
                scheduler_count = 1

