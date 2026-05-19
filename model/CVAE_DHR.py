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
            'cond_loss', 'cond_weight', 'kl_weight', 'total_loss', 'valid_rate', 'quality'
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
        """绘制损失曲线并保存（对数坐标，避免初始epoch极端值压扁曲线）"""
        if len(self.log_data) == 0:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
        fig.suptitle(f"Training Progress @ Epoch {int(self.log_data['epoch'].max())}", y=1.02, fontsize=14)

        # ================= 左侧：各损失（对数坐标）=================
        # 过滤掉值为0或负数的点（对数坐标不支持），用绝对值
        for i, col in enumerate(['recon_loss', 'kld_loss', 'cond_loss']):
            data = self.log_data[col].abs()
            # 替换0值为一个小正数，避免log(0)
            data = data.replace(0, np.finfo(float).eps)
            ax1.semilogy(self.log_data['epoch'], data,
                         color=self.colors[i], label=col.replace('_', ' ').title(),
                         linewidth=2.0)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (log scale)', fontsize=12)
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, which="both", ls="--", alpha=0.5)

        # ================= 右侧：Total Loss（对数） + Valid Rate（线性）=================
        total_data = self.log_data['total_loss'].abs().replace(0, np.finfo(float).eps)
        line1, = ax2.semilogy(self.log_data['epoch'], total_data,
                              color=self.colors[3], label='Total Loss', linewidth=2.0)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Total Loss (log scale)', fontsize=12)
        ax2.grid(True, which="both", ls="--", alpha=0.5)

        ax3 = ax2.twinx()
        line2, = ax3.plot(self.log_data['epoch'], self.log_data['valid_rate'] * 100,
                          color='#9467bd', linestyle='--', label='Valid Rate (%)',
                          linewidth=2.0, marker='s', markersize=4, markevery=max(1, len(self.log_data)//20))
        ax3.set_ylabel('Validation Rate (%)', fontsize=12)

        # 合并图例
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, fontsize=11, loc='upper left')

        plt.tight_layout()

        # 保存图片
        plot_path = os.path.join(self.log_dir,
                                 f"loss_plot_epoch_{int(self.log_data['epoch'].max())}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=200)
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


class ResPriorResBlock(nn.Module):
    """ResidualBlock without BatchNorm, for use in ResPriorBlock.

    BatchNorm's running_var can collapse to zero when the prior block
    receives scalar (1-D) inputs with small batches, causing numerical
    explosion at inference time.  This variant keeps only a plain Linear
    shortcut so that no running statistics are needed.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.activation = nn.ReLU()
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)

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
            # 渐进式维度扩展 (no BatchNorm – avoids running_var collapse)
            nn.Linear(cond_dim, 4),  # 1→4
            nn.ReLU(),
            ResPriorResBlock(4, 8),   # 4→8
            ResPriorResBlock(8, 16),  # 8→16
            ResPriorResBlock(16, 32),
            ResPriorResBlock(32, 64)
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
        logvar = 0.5*(alpha * logvar_x + (1 - alpha) * logvar_prior)
        # 防止 logvar 过大导致 exp(logvar) 溢出为 inf (v3: 收紧到[-5,5]，exp(5)≈148)
        logvar = torch.clamp(logvar, min=-5.0, max=5.0)
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

        # 焓值条件门控：训练初期门接近0，强迫z学习条件信息；逐渐打开
        self.h_gate_logit = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2)≈0.12

        # z → 焓值 可微回归头：直接从latent预测焓值，替代纯外部预测器的不可微梯度
        self.z_to_enthalpy = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)

    def forward(self, latent_vec, enthalpy, X, freerun=False, randomchoose=True):
        latent_vec = latent_vec.to(self.device)  # (512, 64)
        enthalpy_ori = enthalpy
        # 焓值门控：控制decoder能看到多少焓值信息
        h_gate = torch.sigmoid(self.h_gate_logit)  # scalar in (0, 1)
        enthalpy = enthalpy * h_gate  # 门控缩放
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
            # 使用门控后的焓值（与训练一致），注意 enthalpy_ori 是门控前的值
            # enthalpy 已经在上面经过门控，但被expand了，需要用原始门控值
            h_gate = torch.sigmoid(self.h_gate_logit)
            cond = (enthalpy_ori * h_gate).unsqueeze(1).unsqueeze(2)  # [batch, 1, con_dims]
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
                    logits_squeezed = logits.squeeze(1)
                    # 数值稳定性处理：减去max防止exp溢出
                    logits_squeezed = logits_squeezed - logits_squeezed.max(dim=-1, keepdim=True)[0]
                    probs = torch.softmax(logits_squeezed, dim=-1)
                    # 检查概率是否合法，不合法则回退到argmax
                    if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                        next_idx = torch.argmax(logits_squeezed, dim=-1, keepdim=True)
                    else:
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

        # === KL Annealing 参数 (v5: 降低KL上限，释放latent capacity给条件控制) ===
        kl_warmup_epochs = 20            # 前20个epoch纯重建学习
        kl_rampup_epochs = 50            # 用50个epoch线性增长 (epoch 21-70)
        kl_weight = 0.0                  # 当前KL权重
        kl_weight_max = 0.1             # v5: 从0.3降到0.1，避免KLD主导total loss (v4: 6.4*0.3=1.92 → v5: 1.28*0.1=0.128)
        free_bits = 0.02                 # v5: 从0.1降到0.02，释放60+维给reconstruction (最小KLD=1.28)

        # === 渐进式条件训练参数 (v5: 可微cond_loss + 更积极的条件训练) ===
        cond_warmup_epochs = 20          # v5: 从30降到20
        cond_weight = 0.0                # 当前cond_loss权重
        cond_weight_start = 0.01         # v5: 从0.1降到0.01
        cond_weight_step = 0.01          # v5: 从0.1降到0.01
        cond_weight_max = 0.3            # v5: 从10.0降到0.3（可微loss更稳定，不需要过大权重）
        cond_growth_interval = 10        # 每隔多少epoch增长一次
        cond_valid_drop_tolerance = 0.15 # valid_rate下降容忍度
        cond_training_started = False
        best_valid_in_cond_phase = 0.0
        epochs_since_last_growth = 0
        cond_weight_update_epoch = 0     # 开始条件训练的epoch

        for epoch in range(1, nepoch + 1):
            # === KL Annealing: 计算当前epoch的kl_weight ===
            if epoch <= kl_warmup_epochs:
                kl_weight = 0.01  # 最小 KL 约束，防止 logvar 完全失控
            elif epoch <= kl_warmup_epochs + kl_rampup_epochs:
                kl_weight = 0.01 + (kl_weight_max - 0.01) * (epoch - kl_warmup_epochs) / kl_rampup_epochs
            else:
                kl_weight = kl_weight_max

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
                # KL Annealing + Free Bits: 每维度最低KL值防止posterior collapse
                # logvar 已被 encoder clamp 到 [-10, 10]，exp(10)≈22026，不会溢出
                kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [batch, latent_dim]
                kld_per_dim = torch.clamp(kld_per_dim, min=free_bits)  # Free Bits
                # 安全计算：再 clamp 一次确保 kld_per_dim 非负且非 nan/inf
                kld_per_dim = torch.clamp(kld_per_dim, min=0.0, max=1e6)
                kld_loss = kld_per_dim.sum(dim=-1).mean() * kl_weight  # KL Annealing
                # 如果 kld_loss 出现 nan/inf，强制归零避免污染总损失
                if torch.isnan(kld_loss) or torch.isinf(kld_loss):
                    kld_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                # v5: 可微 cond_loss — 使用 z→焓值 回归头（替代不可微的外部预测器）
                # 从 latent_vec (重参数化采样) 预测焓值，与真实归一化焓值比较
                z_pred_enthalpy = self.decoder.z_to_enthalpy(latent_vec).squeeze(-1)  # [batch]
                cond_loss_mean = F.mse_loss(z_pred_enthalpy, enthalpy)

                # 总损失（使用渐进式cond_weight）
                if cond_weight > 0 and cond_loss_mean != 0 and abs(cond_loss_mean) < 100:
                    total_loss = reconstruction_loss + kld_loss + cond_weight * cond_loss_mean
                else:
                    total_loss = reconstruction_loss + kld_loss

                # 反向传播
                encoderOptimizer.zero_grad()
                decoderOptimizer.zero_grad()
                total_loss.backward()
                # 每个batch都做梯度裁剪，第一个batch可能梯度最大（随机初始化），不裁剪反而危险
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1)
                encoderOptimizer.step()
                decoderOptimizer.step()

                # 记录损失
                reconstruction_loss_list.append(reconstruction_loss.item())
                kld_loss_list.append(kld_loss.item())
                cond_loss_list.append(cond_loss_mean.item())
                total_loss_list.append(total_loss.item())

                # v3: 每个batch都计算quality（利用已有的predicted_indices，几乎零开销）
                batch_quality = (predicted_indices == X).sum(dim=-1).float().mean()
                quality_list.append(batch_quality.item())

                # 打印训练信息
                if (nBatch == 1 or nBatch % printInterval == 0):
                    numValid = self.latent_space_quality(numSample, tokenizer)
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
                f"Recon= {avg_recon_loss:.5e} "
                f"KLD= {avg_kld_loss:.5e} "
                f"Cond= {avg_cond_loss:.5e} "
                f"Total= {avg_total_loss:.5e} "
                f"Quality= {avg_quality:.0f}/{self.decoder.maxLength} "
                f"Valid= {avg_valid_rate * 100:.1f}% "
                f"kl_w={kl_weight:.2f} "
                f"cond_w={cond_weight:.4f} "
                f"Enc_lr= {Enc_lr:.5e} "
            )

            # 记录指标 (v3: 加入 quality)
            metrics = {
                'recon_loss': avg_recon_loss,
                'kld_loss': avg_kld_loss,
                'cond_loss': avg_cond_loss,
                'cond_weight': cond_weight,
                'kl_weight': kl_weight,
                'total_loss': avg_total_loss,
                'valid_rate': avg_valid_rate,
                'quality': float(avg_quality)
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

            # LR由StepLR和条件训练阶段自动管理，不再手动覆盖

            # === 渐进式cond_weight更新逻辑 ===
            if not cond_training_started and epoch >= cond_warmup_epochs:
                cond_training_started = True
                cond_weight = cond_weight_start
                cond_weight_update_epoch = epoch
                best_valid_in_cond_phase = avg_valid_rate
                epochs_since_last_growth = 0
                # 条件训练开始时，给LR一个合理的值，确保能学习新的cond_loss
                cond_lr = 1e-4
                for param_group in encoderOptimizer.param_groups:
                    param_group['lr'] = cond_lr
                for param_group in decoderOptimizer.param_groups:
                    param_group['lr'] = cond_lr
                # 重建对应的scheduler（让后续StepLR在此基础上继续衰减）
                encoderScheduler = torch.optim.lr_scheduler.StepLR(encoderOptimizer, step_size=15, gamma=0.9)
                decoderScheduler = torch.optim.lr_scheduler.StepLR(decoderOptimizer, step_size=15, gamma=0.9)
                print(f"[Cond] Epoch {epoch}: Starting cond training, cond_weight={cond_weight:.4f}, LR reset to {cond_lr}")

            elif cond_training_started:
                epochs_since_last_growth += 1

                # 更新cond阶段的best valid rate
                if avg_valid_rate > best_valid_in_cond_phase:
                    best_valid_in_cond_phase = avg_valid_rate

                # 定期检查是否增长权重
                if epochs_since_last_growth >= cond_growth_interval and cond_weight < cond_weight_max:
                    # 检查valid是否下降太多
                    valid_threshold = best_valid_in_cond_phase - cond_valid_drop_tolerance
                    if avg_valid_rate >= valid_threshold:
                        # valid稳定，继续增长
                        new_weight = min(cond_weight + cond_weight_step, cond_weight_max)
                        print(f"[Cond] Epoch {epoch}: cond_weight {cond_weight:.4f} -> {new_weight:.4f}, "
                              f"valid={avg_valid_rate*100:.1f}%, best={best_valid_in_cond_phase*100:.1f}%")
                        cond_weight = new_weight
                    else:
                        # valid下降了，暂停增长
                        print(f"[Cond] Epoch {epoch}: valid dropped ({avg_valid_rate*100:.1f}% < "
                              f"{valid_threshold*100:.1f}%), holding cond_weight={cond_weight:.4f}")
                    epochs_since_last_growth = 0

        # 训练结束后保存最佳模型
        # early_stopper.save_top_models(num_epochs - 1, logger)

        # 保存最后一个 epoch 的损失图
        # logger.plot_losses()
        # plt.savefig(os.path.join(logger.log_dir, 'final_loss_plot.png'))
        # plt.close()
