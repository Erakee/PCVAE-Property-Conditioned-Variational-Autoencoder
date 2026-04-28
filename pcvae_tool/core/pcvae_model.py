"""PCVAE (CVAE_DHR) 推理专用实现。

仅保留 forward / 采样所需的网络结构与权重加载逻辑，
训练代码（TrainingLogger / EarlyStopper / trainModel / calculate_enthalpy_loss）
均已剥离。
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn


# ── 基础模块 ──────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.activation = nn.ReLU()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(nn.Linear(in_dim, out_dim),
                                          nn.BatchNorm1d(out_dim))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        return out + residual


class ResPriorBlock(nn.Module):
    """将焓值条件映射为 (mu_prior, logvar_prior)。"""

    def __init__(self, cond_dim: int, latent_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 4),
            nn.ReLU(),
            ResidualBlock(4, 8),
            ResidualBlock(8, 16),
            ResidualBlock(16, 32),
            ResidualBlock(32, 64),
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

    def forward(self, c):
        h = self.mlp(c)
        return self.mu(h), self.logvar(h)


# ── Encoder / Decoder ────────────────────────────────────────────────────────

class CondEncoder(nn.Module):
    def __init__(self, maxLength, num_vocabs, con_dims, fc_dims, latent_dim,
                 state_fname, device, embed_dim=32):
        super().__init__()
        self.alpha = torch.tensor(0.6)
        self.state_fname = state_fname
        self.device = device
        self.maxLength = maxLength
        self.num_vocabs = num_vocabs

        self.embed = nn.Embedding(num_vocabs, embed_dim, device=device)
        self.fc = nn.Sequential(
            nn.Flatten(),
            ResidualBlock(embed_dim * maxLength, fc_dims[0]),
            *[ResidualBlock(fc_dims[i - 1], fc_dims[i]) for i in range(1, len(fc_dims))],
            nn.ReLU(),
        ).to(device)

        self.prior_block = ResPriorBlock(cond_dim=con_dims, latent_dim=latent_dim).to(device)
        self.mu = nn.Linear(fc_dims[-1], latent_dim, device=device)
        self.logvar = nn.Linear(fc_dims[-1], latent_dim, device=device)

    def forward(self, X, enthalpy, alpha=None):
        if alpha is None:
            alpha = self.alpha
        X_embed = self.embed(X)
        X_flat = torch.flatten(X_embed, start_dim=1)
        X_fc = self.fc(X_flat)
        mu_x = self.mu(X_fc)
        logvar_x = self.logvar(X_fc)

        mu_prior, logvar_prior = self.prior_block(enthalpy.unsqueeze(1))
        mu = alpha * mu_x + (1 - alpha) * mu_prior
        logvar = 0.5 * (alpha * logvar_x + (1 - alpha) * logvar_prior)
        return self.reparameterize(mu, logvar), mu, logvar, mu_prior, logvar_prior

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def loadState(self):
        if not os.path.isfile(self.state_fname):
            raise FileNotFoundError(
                f'Encoder weights not found: {self.state_fname}')
        self.load_state_dict(torch.load(self.state_fname, map_location=self.device))


class CondDecoder(nn.Module):
    def __init__(self, maxLength, num_vocabs, con_dims, latent_dim, hidden_dim,
                 num_hidden, state_fname, device, embed_dim=32):
        super().__init__()
        self.state_fname = state_fname
        self.device = device
        self.maxLength = maxLength
        self.num_vocabs = num_vocabs
        self.con_dims = con_dims

        self.embed = nn.Embedding(num_vocabs, embed_dim, device=device)
        input_dim = latent_dim + embed_dim + con_dims
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_hidden,
            batch_first=True,
            device=device,
        )
        self.fc = nn.Linear(hidden_dim, num_vocabs, device=device)

    def forward(self, latent_vec, enthalpy, X=None, freerun=True, randomchoose=True):
        """
        推理时通常 freerun=True：自回归逐步生成。

        Args:
            latent_vec   : [B, latent_dim]
            enthalpy     : [B] 归一化焓值
            X            : 仅在 freerun=False 时用作 teacher forcing
            freerun      : 是否自回归生成
            randomchoose : True 用 multinomial 采样，False 用 argmax
        """
        latent_vec = latent_vec.to(self.device)
        cond = enthalpy.to(self.device).unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        batch_size = latent_vec.size(0)

        if not freerun:
            inp_embed = self.embed(X)
            start_embed = self.embed(
                torch.zeros((batch_size, 1), dtype=torch.long, device=self.device))
            inp_shifted = torch.cat([start_embed, inp_embed[:, :-1, :]], dim=1)
            latent_expanded = latent_vec.unsqueeze(1).expand(-1, self.maxLength, -1)
            cond_expanded = cond.expand(-1, self.maxLength, -1)
            gru_in = torch.cat([latent_expanded, inp_shifted, cond_expanded], dim=-1)
            gru_out, _ = self.gru(gru_in)
            return self.fc(gru_out)

        out = torch.zeros((batch_size, self.maxLength), dtype=torch.long, device=self.device)
        current_idx = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        current_embed = self.embed(current_idx)
        hidden = torch.zeros(self.gru.num_layers, batch_size,
                             self.gru.hidden_size, device=self.device)

        for i in range(self.maxLength):
            step_in = torch.cat([
                latent_vec.unsqueeze(1),
                current_embed,
                cond,
            ], dim=-1)
            gru_out, hidden = self.gru(step_in, hidden)
            logits = self.fc(gru_out).squeeze(1)
            if randomchoose:
                probs = torch.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, 1)
            else:
                next_idx = torch.argmax(logits, dim=-1, keepdim=True)
            out[:, i] = next_idx.squeeze(1)
            current_embed = self.embed(next_idx)
        return out

    def loadState(self):
        if not os.path.isfile(self.state_fname):
            raise FileNotFoundError(
                f'Decoder weights not found: {self.state_fname}')
        self.load_state_dict(torch.load(self.state_fname, map_location=self.device))


# ── 顶层包装 ──────────────────────────────────────────────────────────────────

class PCVAE:
    """PCVAE 推理包装。"""

    def __init__(self, maxLength, num_vocabs, con_dims, fc_dims, latent_dim,
                 hidden_dim, num_hidden, encoder_state_fname, decoder_state_fname,
                 device, embed_dim=32):
        self.num_vocabs = num_vocabs
        self.latent_dim = latent_dim
        self.device = device
        self.encoder = CondEncoder(
            maxLength, num_vocabs, con_dims, fc_dims, latent_dim,
            encoder_state_fname, device, embed_dim=embed_dim)
        self.decoder = CondDecoder(
            maxLength, num_vocabs, con_dims, latent_dim, hidden_dim, num_hidden,
            decoder_state_fname, device, embed_dim=embed_dim)

    def load(self):
        """加载预训练权重并切换到 eval 模式。"""
        self.encoder.loadState()
        self.decoder.loadState()
        self.encoder.eval()
        self.decoder.eval()
        return self
