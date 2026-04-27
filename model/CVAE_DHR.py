import sys
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
# Add the parent directory to the system path
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import util.utils as utils
import time
from util.enthalpy_predictor import predict_enthalpy
# from util.train_logger import TrainingLogger

class TrainingLogger:
    def __init__(self, base_dir=None):
        if base_dir:
            self.log_dir = Path(base_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            base_dir = "experiments"
            self.exp_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_dir = os.path.join(base_dir, self.exp_time)
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化数据存储
        self.log_data = pd.DataFrame(columns=[
            'epoch', 'recon_loss', 'kld_loss',
            'cond_loss', 'total_loss', 'valid_rate'
        ])

        # 图表样式设置（兼容新旧版matplotlib）
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
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
        for i, col in enumerate(['recon_loss', 'kld_loss', 'cond_loss']):
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

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.activation = nn.ReLU()
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:  # 维度不匹配时使用1x1卷积等效调整
            self.shortcut = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim))# nn.Linear(in_dim, out_dim)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out += residual
        return out


class ResPriorBlock(nn.Module):
    def __init__(self, cond_dim, latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            # 渐进式维度扩展
            nn.Linear(cond_dim, 4),  # 1→4
            nn.ReLU(),
            ResidualBlock(4, 8),  # 4→8
            ResidualBlock(8, 16),  # 8→16
            ResidualBlock(16, 32),
            ResidualBlock(32, 64)
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

    def forward(self, c):
        h = self.mlp(c)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class CondEncoder(nn.Module): # 通过embedding压缩smiles维度，使用rnn实现encoder，decoder采用mlp。未实现多层条件拼接
    def __init__(self, maxLength, num_vocabs, con_dims, fc_dims, latent_dim,
                 state_fname, device, embed_dim=32) -> None:
        super().__init__()
        # self.mix_weight = nn.Parameter(torch.tensor(0.0))
        self.alpha = torch.tensor(0.6)
        self.state_fname = state_fname
        self.device = device
        self.maxLength = maxLength
        self.num_vocabs = num_vocabs
        self.embed = nn.Embedding(num_vocabs, embed_dim, device=self.device)
        self.fc = nn.Sequential(
            nn.Flatten(),
            ResidualBlock(embed_dim * maxLength, fc_dims[0]),
            *[ResidualBlock(fc_dims[i - 1], fc_dims[i])
              for i in range(1, len(fc_dims))],
            nn.ReLU()
        ).to(device)

        self.prior_block = ResPriorBlock(cond_dim=con_dims, latent_dim=latent_dim).to(device)
        self.mu = torch.nn.Linear(fc_dims[-1], latent_dim, device=self.device)
        self.logvar = torch.nn.Linear(
            fc_dims[-1], latent_dim, device=self.device)  #mu和logvar的拟合依靠kld散度损失来更新

    def forward(self, X, enthalpy, alpha=None):
        if alpha is None:
            alpha = self.alpha
        X_embed = self.embed(X)
        X_flatten = torch.flatten(X_embed, start_dim=1)
        X_fc = self.fc(X_flatten)
        mu_x = self.mu(X_fc)
        logvar_x = self.logvar(X_fc)

        mu_prior, logvar_prior = self.prior_block(enthalpy.unsqueeze(1))
        # 动态混合系数
        # alpha = torch.sigmoid(self.mix_weight)  # 可学习参数
        mu = alpha * mu_x + (1 - alpha) * mu_prior
        # var_x = torch.exp(logvar_x)
        # var_prior = torch.exp(logvar_prior)
        # mixed_var = alpha**2 * var_x +(1-alpha)**2 *var_prior
        logvar = 0.5*(alpha * logvar_x + (1 - alpha) * logvar_prior)
        # logvar = 0.5*((torch.log(alpha**2) + logvar_x) + (torch.log((1 - alpha)**2) + logvar_prior))
        # logvar = torch.log(mixed_var + 1e-8)

        return self.reparameterize(mu, logvar), mu, logvar, mu_prior, logvar_prior #logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def loadState(self, seed=1):
        if os.path.isfile(self.state_fname):
            self.load_state_dict(torch.load(self.state_fname))
        else:
            # 初始化前设置种子
            print(f"state file is not found, initializing with fixed seed: {seed}.")
            with utils.temp_seed(seed):  # 需要实现上下文管理器
                for layer in self.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            self.saveState()  # 保存初始化的权重

    def saveState(self):
        dir_name = os.path.dirname(self.state_fname)
        utils.mkdir_multi(dir_name)
        torch.save(self.state_dict(), self.state_fname)


class CondDecoder(torch.nn.Module):
    def __init__(self, maxLength, num_vocabs, con_dims, latent_dim, hidden_dim, num_hidden, state_fname,
                 device, embed_dim=32) -> None:
        super().__init__()
        self.state_fname = state_fname
        self.device = device
        self.maxLength = maxLength
        self.num_vocabs = num_vocabs
        self.con_dims = con_dims
        self.embed = torch.nn.Embedding(num_vocabs, embed_dim, device=self.device)
        # GRU 的输入维度应该是 latent_dim + embed_dim + con_dims
        input_dim = latent_dim + embed_dim + con_dims
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_hidden,
            batch_first=True,
            device=self.device
        )
        self.fc = torch.nn.Linear(hidden_dim, num_vocabs, device=self.device)

    def forward(self, latent_vec, enthalpy, X, freerun=False, randomchoose=True):
        latent_vec = latent_vec.to(self.device)  # (512, 64)
        enthalpy_ori = enthalpy
        enthalpy = enthalpy.unsqueeze(1).unsqueeze(2).expand(-1, self.maxLength, -1)  # (512, 128, 1)

        if not freerun:
            inp_embed = self.embed(X)  # [batch_size, maxLength, embed_dim]
            batch_size = inp_embed.size(0)
            # 检查索引是否超出范围 (使用完整词表大小)
            if torch.max(X) >= self.num_vocabs:
                raise ValueError(f"Input indices exceed vocabulary size: max index = {torch.max(X)}, vocab size = {self.num_vocabs}")
            # 构建初始输入（右移一位）
            start_embed = self.embed(  # <start> 索引为0
                torch.zeros((batch_size, 1), dtype=torch.long, device=self.device))  # [batch, 1, embed_dim]
            # 右移一位
            inp_shifted = torch.cat([start_embed, inp_embed[:, :-1, :]], dim=1)  # [batch, maxLength, embed_dim]
            # 拼接潜在变量、嵌入输入和条件
            latent_expanded = latent_vec.unsqueeze(1).expand(-1, self.maxLength, -1)  # [batch, maxLength, latent_dim]
            X = torch.cat([latent_expanded, inp_shifted, enthalpy], dim=-1)  # [batch, maxLength, latent_dim + embed_dim + con_dims]

            # GRU 前向
            gru_out, _ = self.gru(X)
            logits = self.fc(gru_out)  # [batch, maxLength, num_vocabs]
            return logits
        else:  # 生成模式：自由运行
            batch_size = latent_vec.size(0)
            out = torch.zeros((batch_size, self.maxLength), dtype=torch.float32, device=self.device)  # 添加 device
            cond = enthalpy_ori.unsqueeze(1).unsqueeze(2)  # [batch, 1, con_dims]
            # 初始字符设为 <start>（假设索引为0）
            current_idx = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
            current_embed = self.embed(current_idx)  # [batch, 1, embed_dim]
            # 初始化 GRU 隐藏状态
            # hidden = None
            hidden = torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size, device=self.device)

            # 逐字符生成
            for i in range(self.maxLength):
                # 拼接当前输入
                X = torch.cat([
                    latent_vec.unsqueeze(1).to(self.device),  # [batch, 1, latent_dim]
                    current_embed.to(self.device),  # [batch, 1, embed_dim]
                    cond.to(self.device)  # [batch, 1, con_dims]
                ], dim=-1)
                # GRU 前向传播
                gru_out, hidden = self.gru(X, hidden)  # 使用前一个隐藏状态
                logits = self.fc(gru_out)  # [batch, 1, num_vocabs]
                # 获取下一个字符
                # if randomchoose:
                #     # 数值稳定性处理
                #     logits = logits.squeeze(1) - torch.max(logits.squeeze(1), dim=-1, keepdim=True)[0]
                #     probs = torch.softmax(logits, dim=-1)
                #     # 检查概率值是否合法
                #     if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                #         print("Invalid probabilities detected. Logits:", logits)
                #         # 可以选择使用 argmax 作为替代
                #         next_idx = torch.argmax(logits, dim=-1, keepdim=True)
                #     else:
                #         next_idx = torch.multinomial(probs, 1)  # [batch, 1]
                if randomchoose:
                    probs = torch.softmax(logits.squeeze(1), dim=-1)
                    next_idx = torch.multinomial(probs, 1)  # [batch, 1]
                else:
                    next_idx = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)  # [batch, 1]
                # 更新输出和下一个输入
                out[:, i] = next_idx.squeeze(1)
                current_embed = self.embed(next_idx)  # [batch, 1, embed_dim]

            return out

    def loadState(self, seed=1):
        if os.path.isfile(self.state_fname):
            self.load_state_dict(torch.load(self.state_fname))
        else:
            print(f"state file is not found, initializing with fixed seed: {seed}.")
            with utils.temp_seed(seed):
                for layer in self.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            self.saveState()

    def saveState(self):
        dir_name = os.path.dirname(self.state_fname)
        utils.mkdir_multi(dir_name)
        torch.save(self.state_dict(), self.state_fname)


class EarlyStopper:
    def __init__(self, patience=5, top_n=2):
        self.patience = patience
        self.counter = 0
        self.best_valid = -np.inf
        self.top_models = []  # 保存格式：(valid_rate, encoder_state, decoder_state)
        self.top_n = top_n

    def check(self, current_valid, model, epoch):
        # 更新最佳模型队列
        self.top_models.append((current_valid, model.encoder.state_dict(), model.decoder.state_dict()))
        self.top_models.sort(reverse=True, key=lambda x: x[0])
        if len(self.top_models) > self.top_n:
            self.top_models.pop()

        # 获取当前记录中的最大有效值
        max_valid_rate = max([valid for valid, _, _ in self.top_models])
        # 早停判断，增加新的条件
        if current_valid > self.best_valid:
            self.best_valid = current_valid
            self.counter = 0
            return False  # 不停止
        else:
            self.counter += 1
            if (epoch > 100 and max_valid_rate < 0.4) or (epoch > 800 and self.counter >= self.patience):
                if (max_valid_rate > 0.6) and (epoch < 550):
                    self.counter += -20
                else:
                    return True
            return False

    def save_top_models(self, epoch, logger):
        log_dir = logger.log_dir
        for i, (valid_rate, enc_state, dec_state) in enumerate(self.top_models):
            torch.save(enc_state, os.path.join(log_dir, f"encoder_top{i + 1}_epoch_{epoch}.pt"))
            torch.save(dec_state, os.path.join(log_dir, f"decoder_top{i + 1}_epoch_{epoch}.pt"))

class ConVAE(object):
    def __init__(self, maxLength, num_vocabs, con_dims, fc_dims, latent_dim, hidden_dim,
                 num_hidden, encoder_state_fname, decoder_state_fname, device) -> None:
        self.num_vocabs = num_vocabs
        self.latent_dim = latent_dim
        self.device = device
        self.encoder = CondEncoder(maxLength, num_vocabs, con_dims,
                               fc_dims, latent_dim, encoder_state_fname, device)
        self.decoder = CondDecoder(maxLength, num_vocabs, con_dims, latent_dim,
                               hidden_dim, num_hidden, decoder_state_fname, device)

    def reconstruction_quality_per_sample(self, X, enthalpy):
        self.encoder.eval()
        self.decoder.eval()
        # 1. 通过编码器获取隐空间表示
        latent_vec, mu, logvar,_,_ = self.encoder(X, enthalpy)
        # 2. 通过解码器重构输入（假设X已经是整数索引）
        pred_logits = self.decoder(mu, enthalpy, X)  # [batch, maxLength, num_vocabs]
        # 3. 计算每个位置的最大概率索引
        pred_indices = torch.argmax(pred_logits, dim=-1)  # [batch, maxLength]
        # 4. 计算正确字符数量
        correct_mask = (pred_indices == X)  # [batch, maxLength]
        correct_counts = correct_mask.sum(dim=-1)  # [batch]

        return correct_counts.float()  # 返回浮点数张量

    def sample(self, nSample):
        # 从标准正态分布采样
        latent_vec = torch.randn( # 这里采样生成的是[nSample, maxlength]
            (nSample, self.latent_dim), device=self.device)
        _enthalpy = torch.randn(nSample, device=self.device)  # 这里随机采样的生成焓也随机生成
        # 通过解码器生成分子
        y = self.decoder(latent_vec, _enthalpy, None, freerun=True)  # y [nSample, maxlength]
        numVectors = y
        return numVectors.cpu(), None

    def latent_space_quality(self, nSample, tokenizer=None):
        self.decoder.eval()
        # 从隐空间采样并生成分子
        numVectors, _ = self.sample(nSample)# 采样得到用于表示分子的数字序列
        # 将数字序列转换回SMILES字符串
        smilesStrs = tokenizer.getSmiles(numVectors)
        validSmilesStrs = [sm for sm in smilesStrs if utils.isValidSmiles(sm)]
        return len(validSmilesStrs)

    def calculate_enthalpy_loss(self, gen_smiles, cond_enthalpy, lb, ub):  # 需要数据集中的归一化上下限来反归一化，以和预测数据的大小匹配
        # Calculate the loss for the enthalpy prediction
        gt_enthalpy = cond_enthalpy
        predicted_enthalpy = torch.zeros(len(gen_smiles))
        valid_enthalpy = []
        _mask = torch.zeros(len(gen_smiles))
        _valid_id_mask = []

        for idx, smiles in enumerate(gen_smiles):  # 对于每个smiles，如果有效则使用predict方法，并append有效值；若无效则append 0
            if utils.isValidSmiles(smiles):  # 初步判断有效结构
                predicted_value = predict_enthalpy(smiles)
                if predicted_value != 0:  # 保存预测结果不为0的值
                    predicted_enthalpy[idx] = predicted_value  # 将生成的全0tensor中对应项保存为预测值
                    _mask[idx] = 1  # 将mask对应位置标记为有效
                    _valid_id_mask.append(idx)
                    valid_enthalpy.append(predicted_value) # 单独拎出有效结果，后面用len统计

        # 将有效的预测值和 enthalpy 转换为张量
        valid_enthalpy_tensor = torch.tensor(valid_enthalpy, device=self.device)
        gt_enthalpy_tensor = torch.tensor(gt_enthalpy, device=self.device)
        gt_mask_enthalpy_tensor = torch.tensor([gt_enthalpy_tensor[i] for i in range(len(gt_enthalpy_tensor)) if _mask[i] == 1], device=self.device)

        if len(valid_enthalpy_tensor) > 0:
            normed_valid_enthalpy_tensor = (valid_enthalpy_tensor - lb) / (ub - lb) # Normalization
            cond_loss_mean = torch.nn.functional.mse_loss(normed_valid_enthalpy_tensor, gt_mask_enthalpy_tensor)  # 只计算有效样本的损失
        else:
            cond_loss_mean = torch.tensor(1.0, device=self.device)  # 如果没有有效样本，返回1

        return cond_loss_mean  # loss_per_sample_tensor, predicted_enthalpy_tensor, valid_enthalpy_tensor


    def trainModel(self, dataloader, encoderOptimizer, decoderOptimizer, encoderScheduler, decoderScheduler, KLD_alpha,
                   nepoch, tokenizer, printInterval, lb, ub, seed, log_dir='training_params/CVAE_DHR/default_dir'):
        self.encoder.loadState(seed)
        self.decoder.loadState(seed)
        self.lb = lb
        self.ub = ub
        logger = TrainingLogger(base_dir=log_dir)  # 初始化日志系统
        early_stopper = EarlyStopper(patience=50, top_n=2)  # 初始化早停器
        minloss = None
        numSample = 100  # 训练过程中采样，用于计算valid数量
        num_epochs = utils.config['num_epoch']
        scheduler_count = 0

        for epoch in range(1, nepoch + 1):
            reconstruction_loss_list, kld_loss_list, cond_loss_list, total_loss_list = [], [], [], []
            quality_list, numValid_list = [], []

            for nBatch, (X, enthalpy) in enumerate(dataloader, 1):
                self.encoder.train()
                self.decoder.train()

                # 数据转移到设备
                X = X.to(self.device)
                enthalpy = enthalpy.to(self.device)

                # 前向传播
                # latent_vec, mu, logvar = self.encoder(X, enthalpy)
                latent_vec, mu, logvar, mu_prior, logvar_prior = self.encoder(X, enthalpy)
                # latent_vec, mu, mix_var = self.encoder(X, enthalpy)
                # logvar = torch.log(mix_var)
                pred_y = self.decoder(latent_vec, enthalpy, X)
                predicted_indices = torch.argmax(pred_y, dim=-1)
                predicted_smiles = tokenizer.getSmiles(predicted_indices)

                # 计算损失
                reconstruction_loss = F.cross_entropy(
                    pred_y.view(-1, pred_y.size(-1)), X.view(-1))
                # reconstruction_loss = F.cross_entropy(
                #     pred_y.view(-1, self.decoder.num_vocabs), X.view(-1))
                # kld_loss = 0.5 * torch.sum(logvar_prior - logvar + (torch.exp(logvar) + (mu - mu_prior)**2) / torch.exp(logvar_prior) - 1)
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kld_loss = kld_loss.mean() * KLD_alpha

                cond_loss_mean = self.calculate_enthalpy_loss(predicted_smiles, enthalpy, lb, ub)

                # 总损失
                if cond_loss_mean != 0 and abs(cond_loss_mean) < 10:  # 条件损失有效
                    total_loss = reconstruction_loss + kld_loss + cond_loss_mean
                else:
                    total_loss = reconstruction_loss + kld_loss

                # 反向传播
                encoderOptimizer.zero_grad()
                decoderOptimizer.zero_grad()
                total_loss.backward()
                if not (nBatch == 1 and epoch == 1):  # 跳过第一个batch的梯度裁剪，避免初始化噪声
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1)
                    torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1)
                encoderOptimizer.step()
                decoderOptimizer.step()

                # 记录损失
                reconstruction_loss_list.append(reconstruction_loss.item())
                kld_loss_list.append(kld_loss.item())
                cond_loss_list.append(cond_loss_mean.item())
                total_loss_list.append(total_loss.item())

                # 打印训练信息
                if (nBatch == 1 or nBatch % printInterval == 0):
                    quality = self.reconstruction_quality_per_sample(X, enthalpy).mean()
                    numValid = self.latent_space_quality(numSample, tokenizer)
                    quality_list.append(quality)
                    numValid_list.append(numValid)
                    if minloss is None or total_loss.item() < minloss:
                        self.encoder.saveState()
                        self.decoder.saveState()
                        minloss = total_loss.item()

            # Epoch 结束后的处理
            encoderScheduler.step()
            decoderScheduler.step()

            # 计算 epoch 平均指标
            # 转换为 CPU 张量并转换为 NumPy 数组
            reconstruction_loss_list = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in
                                        reconstruction_loss_list]
            kld_loss_list = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in kld_loss_list]
            cond_loss_list = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in cond_loss_list]
            total_loss_list = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in
                               total_loss_list]
            quality_list = [quality.cpu().numpy() if isinstance(quality, torch.Tensor) else quality for quality in
                            quality_list]
            numValid_list = [numValid.cpu().numpy() if isinstance(numValid, torch.Tensor) else numValid for numValid in
                             numValid_list]
            avg_recon_loss = np.mean(reconstruction_loss_list)
            avg_kld_loss = np.mean(kld_loss_list)
            avg_cond_loss = np.mean(cond_loss_list)
            avg_total_loss = np.mean(total_loss_list)
            avg_quality = np.mean(quality_list)
            avg_valid_rate = np.mean(numValid_list) / numSample

            # 打印 epoch 总结
            Enc_lr = encoderOptimizer.param_groups[0]['lr']
            Dec_lr = decoderOptimizer.param_groups[0]['lr']
            print(
                f"[{time.ctime()}] Epoch {epoch:4d}: "
                f"Reconstruction_Loss= {avg_recon_loss:.5e} "
                f"KLD_Loss= {avg_kld_loss:.5e} "
                f"Condition_Loss= {avg_cond_loss:.5e} "
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
                'cond_loss': avg_cond_loss,
                'total_loss': avg_total_loss,
                'valid_rate': avg_valid_rate
            }
            logger.log_metrics(epoch, metrics)

            # 定期生成图表
            if epoch % 50 == 0:
                logger.plot_losses()

            # 早停与模型保存
            if early_stopper.check(avg_valid_rate, self, epoch):
                print(f"Early stopping at epoch {epoch}")
                early_stopper.save_top_models(epoch, logger)
                break

            if (avg_valid_rate>0.8) and (scheduler_count==0):
                encoderOptimizer.param_groups[0]['lr'] = 1e-5
                decoderOptimizer.param_groups[0]['lr'] = 1e-5
                scheduler_count = 1

        # 训练结束后保存最佳模型
        # early_stopper.save_top_models(num_epochs - 1, logger)

        # 保存最后一个 epoch 的损失图
        # logger.plot_losses()
        # plt.savefig(os.path.join(logger.log_dir, 'final_loss_plot.png'))
        # plt.close()
