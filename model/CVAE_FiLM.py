import sys
import os
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import util.utils as utils
from util.config_loader import load_training_config
from dataset.dataset import SmilesDictDataset


# ===================== FiLM Components =====================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation with residual initialization.

    Uses the formulation: output = (1 + gamma) * x + beta
    The final linear layer is zero-initialized so that initially
    gamma=0, beta=0, and the layer acts as identity (residual).
    This ensures stable autoregressive generation from the start.
    """

    def __init__(self, cond_dim: int, feature_dim: int, hidden_dim: int = 16,
                 device=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim, device=device),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim * 2, device=device),
        )
        self.feature_dim = feature_dim
        # Small random init: FiLM starts near-identity but with gradient flow
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [B, feature_dim] or [B, T, feature_dim]
            cond: [B, cond_dim]
        Returns:
            modulated features, same shape as x
        """
        gamma_beta = self.net(cond)  # [B, feature_dim * 2]
        gamma = gamma_beta[:, :self.feature_dim]
        beta = gamma_beta[:, self.feature_dim:]
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        # Residual FiLM: (1+gamma)*x + beta, starts as identity
        return (1.0 + gamma) * x + beta


# ===================== FiLM Encoder =====================

class FiLMEncoder(nn.Module):
    """Encoder identical to PCVAE's DualEncoder architecture.
    
    Uses embedding + concatenation of structure and enthalpy, followed by
    ResNet blocks to produce mu, logvar for the latent space.
    """

    def __init__(self, vocab_size, emb_dim, hidden_dim, latent_dim, maxLength, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.maxLength = maxLength
        self.device = device

        self.embedding = nn.Embedding(vocab_size, emb_dim, device=device)
        self.fc1 = nn.Linear(emb_dim * maxLength + 1, hidden_dim, device=device)
        self.res1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device),
        )
        self.res2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim, device=device)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim, device=device)
        self.prior_block = PriorBlock(cond_dim=1, latent_dim=latent_dim, device=device)

    def forward(self, x, h1):
        """
        Args:
            x: [batch, maxLength] - integer token indices
            h1: [batch] or [batch, 1] - enthalpy (normalized)
        Returns:
            z: [batch, latent_dim]
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
            mu_prior: [batch, latent_dim]
            logvar_prior: [batch, latent_dim]
        """
        if h1.dim() == 1:
            h1 = h1.unsqueeze(1)

        batch_size = x.size(0)
        x_emb = self.embedding(x)                  # [batch, maxLength, emb_dim]
        x_flat = x_emb.reshape(batch_size, -1)     # [batch, maxLength * emb_dim]
        x_cat = torch.cat([x_flat, h1], dim=1)     # [batch, maxLength*emb_dim + 1]

        h = F.silu(self.fc1(x_cat))
        h = F.silu(h + self.res1(h))               # residual block 1
        h = F.silu(h + self.res2(h))               # residual block 2

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        # Prior p(z|h)
        mu_prior, logvar_prior = self.prior_block(h1)
        logvar_prior = torch.clamp(logvar_prior, min=-10.0, max=10.0)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar, mu_prior, logvar_prior

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


class PriorBlock(nn.Module):
    """Conditional prior p(z|h) — maps enthalpy to prior mu/logvar."""

    def __init__(self, cond_dim: int, latent_dim: int, device=None):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, 32, device=device),
            nn.SiLU(),
            nn.Linear(32, latent_dim * 2, device=device),
        )

    def forward(self, h1):
        out = self.fc(h1)
        mu = out[:, :out.size(1) // 2]
        logvar = out[:, out.size(1) // 2:]
        return mu, logvar


# ===================== FiLM Decoder =====================

class FiLMDecoder(nn.Module):
    """Decoder with FiLM conditioning on enthalpy.

    Enthalpy is injected in two ways:
    1. Concatenated with z to initialize GRU hidden state (same as CVAE_DHR)
    2. FiLM layers modulate GRU outputs at each timestep via scale/shift

    This dual injection ensures enthalpy information reaches both the
    recurrent dynamics and the output representation.
    """

    def __init__(self, vocab_size, emb_dim, hidden_dim, latent_dim, maxLength, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.maxLength = maxLength
        self.device = device
        self.num_layers = 4

        self.embedding = nn.Embedding(vocab_size, emb_dim, device=device)
        self.gru = nn.GRU(latent_dim + emb_dim, hidden_dim,
                          num_layers=self.num_layers, batch_first=True, device=device)

        # FiLM modulation after GRU: enthalpy [B,1] -> gamma, beta [B, hidden_dim]
        self.film = FiLMLayer(cond_dim=1, feature_dim=hidden_dim, hidden_dim=16,
                              device=device)

        self.fc = nn.Linear(hidden_dim, vocab_size, device=device)
        # to_latent takes z concatenated with enthalpy (same as CVAE_DHR)
        self.to_latent = nn.Linear(latent_dim + 1, hidden_dim, device=device)

    def forward(self, z, enthalpy, input_x, freerun=False):
        """Forward pass for FiLM decoder.
        
        Args:
            z: latent vector [B, latent_dim]
            enthalpy: enthalpy condition [B] or [B, 1] (normalized)
            input_x: input token indices [B, maxLength]
            freerun: if True, generate autoregressively
        """
        if enthalpy.dim() == 1:
            enthalpy = enthalpy.unsqueeze(1)

        batch_size = z.size(0)

        # Initialize hidden state from z concatenated with enthalpy (same as CVAE_DHR)
        z_h = torch.cat([z, enthalpy], dim=1)  # [B, latent_dim + 1]
        init_h = F.silu(self.to_latent(z_h))   # [B, hidden_dim]
        init_h = init_h.unsqueeze(0).repeat(self.num_layers, 1, 1)

        if not freerun:
            # Teacher forcing: shift right by 1 (prepend <sos>)
            embedded = self.embedding(input_x[:, :-1])  # [B, maxLength-1, emb_dim]
            start_embed = self.embedding(
                torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
            )  # [B, 1, emb_dim]
            inp_shifted = torch.cat([start_embed, embedded], dim=1)  # [B, maxLength, emb_dim]

            z_expanded = z.unsqueeze(1).expand(-1, inp_shifted.size(1), -1)
            gru_input = torch.cat([z_expanded, inp_shifted], dim=2)

            gru_out, _ = self.gru(gru_input, init_h)

            # FiLM modulation via FiLMLayer.forward() — uses residual (1+γ)*x + β
            gru_out = self.film(gru_out, enthalpy)

            logits = self.fc(gru_out)
        else:
            # Free-running (autoregressive)
            outputs = []
            current_input = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            hidden = init_h

            for t in range(self.maxLength):
                embedded = self.embedding(current_input)
                gru_input = torch.cat([z, embedded], dim=1).unsqueeze(1)

                gru_out, hidden = self.gru(gru_input, hidden)
                gru_out = gru_out.squeeze(1)

                # FiLM modulation via FiLMLayer.forward() — uses residual (1+γ)*x + β
                gru_out = self.film(gru_out, enthalpy)

                logit = self.fc(gru_out)
                outputs.append(logit)
                current_input = logit.argmax(dim=1)

            logits = torch.stack(outputs, dim=1)

        return logits


# ===================== FiLM-CVAE Wrapper =====================

class FiLMConVAE(nn.Module):
    """FiLM-conditioned CVAE.

    Compatible with the launcher: accepts **vae_param kwargs from config.yaml.
    The key difference from standard CVAE is that enthalpy is injected via
    FiLM (Feature-wise Linear Modulation) in the decoder rather than simple
    concatenation.
    """

    def __init__(self, *, con_dims=None, fc_dims=None, latent_dim, hidden_dim,
                 num_hidden=None, num_vocabs=None, maxLength, vocab_size=None,
                 emb_dim=None, device,
                 encoder_state_fname=None, decoder_state_fname=None, **kwargs):
        super().__init__()
        # Map from common vae_param keys
        vocab_size = vocab_size or num_vocabs or 19
        emb_dim = emb_dim or 32

        self.encoder = FiLMEncoder(vocab_size, emb_dim, hidden_dim, latent_dim, maxLength, device)
        self.decoder = FiLMDecoder(vocab_size, emb_dim, hidden_dim, latent_dim, maxLength, device)
        self.device = device
        self.latent_dim = latent_dim
        self.maxLength = maxLength
        self.emb_dim = emb_dim

        self.encoder_state_fname = encoder_state_fname
        self.decoder_state_fname = decoder_state_fname
        self._encoder_state_fname = encoder_state_fname
        self._decoder_state_fname = decoder_state_fname

    def forward(self, x, enthalpy):
        """Forward pass.
        
        Args:
            x: [batch, maxLength] token indices
            enthalpy: [batch] or [batch, 1] normalized enthalpy
        """
        z, mu, logvar, mu_prior, logvar_prior = self.encoder(x, enthalpy)
        logits = self.decoder(z, enthalpy, x, freerun=False)
        return logits, mu, logvar, mu_prior, logvar_prior

    # ---- Interface compatible with the launcher ----

    def attach_checkpoint_weights(self):
        """Load pretrained encoder/decoder weights if available."""
        if self._encoder_state_fname and os.path.isfile(self._encoder_state_fname):
            state = torch.load(self._encoder_state_fname, map_location=self.device)
            self.encoder.load_state_dict(state, strict=False)
            print(f"Loaded encoder weights from {self._encoder_state_fname}")
        if self._decoder_state_fname and os.path.isfile(self._decoder_state_fname):
            state = torch.load(self._decoder_state_fname, map_location=self.device)
            self.decoder.load_state_dict(state, strict=False)
            print(f"Loaded decoder weights from {self._decoder_state_fname}")

    def trainModel(self, dataloader, encoderOptimizer, decoderOptimizer,
                   encoderScheduler, decoderScheduler, KLD_alpha,
                   nepoch, tokenizer, printInterval, lb, ub, seed,
                   log_dir='training_params/CVAE_FiLM/default_dir'):
        """Training loop compatible with the launcher.
        
        Signature matches CVAE_DHR.ConVAE.trainModel so the launcher can
        call it without modification. The dataloader yields (X, enthalpy) tuples.
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        max_length = self.maxLength
        pad_idx = tokenizer.getTokensNum('<pad>')

        logger = TrainingLogger(base_dir=str(log_dir))

        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        best_epoch = 0
        grad_clip = 0.5

        # KL annealing (simpler than CVAE_DHR)
        kl_warmup_epochs = 15
        kl_rampup_epochs = 30
        kl_weight_max = 0.2
        free_bits = 0.02

        numSample = 100

        for epoch in range(1, nepoch + 1):
            # KL Annealing
            if epoch <= kl_warmup_epochs:
                kl_weight = 0.01
            elif epoch <= kl_warmup_epochs + kl_rampup_epochs:
                kl_weight = 0.01 + (kl_weight_max - 0.01) * (epoch - kl_warmup_epochs) / kl_rampup_epochs
            else:
                kl_weight = kl_weight_max

            self.train()
            epoch_start = time.time()
            total_loss = 0.0
            total_recon = 0.0
            total_kld = 0.0
            batch_count = 0

            for batch_idx, (X, enthalpy) in enumerate(dataloader):
                X = X.to(self.device)
                enthalpy = enthalpy.to(self.device)

                encoderOptimizer.zero_grad()
                decoderOptimizer.zero_grad()

                # Forward
                z, mu, logvar, mu_prior, logvar_prior = self.encoder(X, enthalpy)
                pred_y = self.decoder(z, enthalpy, X, freerun=False)

                # Reconstruction loss
                recon_loss = F.cross_entropy(
                    pred_y.contiguous().view(-1, pred_y.size(-1)),
                    X.contiguous().view(-1)
                )

                # KL loss with free bits
                kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                kld_per_dim = torch.clamp(kld_per_dim, min=free_bits)
                kld_per_dim = torch.clamp(kld_per_dim, min=0.0, max=1e6)
                kld_loss = kld_per_dim.sum(dim=-1).mean()

                if torch.isnan(kld_loss) or torch.isinf(kld_loss):
                    kld_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                loss = recon_loss + kl_weight * kld_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)
                encoderOptimizer.step()
                decoderOptimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kld += kld_loss.item()
                batch_count += 1

            encoderScheduler.step()
            decoderScheduler.step()

            avg_loss = total_loss / max(batch_count, 1)
            avg_recon = total_recon / max(batch_count, 1)
            avg_kld = total_kld / max(batch_count, 1)

            # ---- Validation ----
            self.eval()
            val_loss_acc = 0.0
            val_correct = 0
            val_total = 0
            val_batch_count = 0

            with torch.no_grad():
                for (X, enthalpy) in dataloader:
                    X = X.to(self.device)
                    enthalpy = enthalpy.to(self.device)

                    z, mu, logvar, mu_prior, logvar_prior = self.encoder(X, enthalpy)
                    pred_y = self.decoder(z, enthalpy, X, freerun=False)

                    recon_loss = F.cross_entropy(
                        pred_y.contiguous().view(-1, pred_y.size(-1)),
                        X.contiguous().view(-1)
                    )
                    kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                    kld_per_dim = torch.clamp(kld_per_dim, min=0.0, max=1e6)
                    kld_loss = kld_per_dim.sum(dim=-1).mean()

                    val_loss_acc += recon_loss.item() + kl_weight * kld_loss.item()
                    preds = pred_y.argmax(dim=-1)
                    mask = X != pad_idx
                    val_correct += (preds == X)[mask].sum().item()
                    val_total += mask.sum().item()
                    val_batch_count += 1

            val_loss = val_loss_acc / max(val_batch_count, 1)
            val_acc = val_correct / val_total if val_total > 0 else 0.0

            # Compute valid rate (every 10 epochs, like CVAE_DHR)
            valid_rate = 0.0
            if epoch % 10 == 0 or epoch == nepoch:
                try:
                    self.eval()
                    with torch.no_grad():
                        latent_vec = torch.randn((numSample, self.latent_dim), device=self.device)
                        _enthalpy = torch.rand(numSample, device=self.device)  # normalized [0,1]
                        y = self.decoder(latent_vec, _enthalpy, None, freerun=True)
                        predicted_indices = y
                        predicted_smiles = tokenizer.getSmiles(predicted_indices)
                        validSmilesStrs = [sm for sm in predicted_smiles if utils.isValidSmiles(sm)]
                        valid_rate = len(validSmilesStrs) / numSample
                except Exception:
                    valid_rate = 0.0

            epoch_time = time.time() - epoch_start

            logger.log_metrics(epoch, {
                'recon_loss': f'{avg_recon:.4f}',
                'kld_loss': f'{avg_kld:.4f}',
                'cond_loss': 0.0,
                'cond_weight': 0.0,
                'kl_weight': f'{kl_weight:.4f}',
                'total_loss': f'{avg_loss:.4f}',
                'valid_rate': f'{valid_rate:.4f}',
                'quality': 0.0,
            })

            if epoch % printInterval == 0 or epoch == nepoch:
                print(f"Epoch {epoch}/{nepoch} | "
                      f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KLD: {avg_kld:.4f} | "
                      f"ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.4f} | "
                      f"ValidRate: {valid_rate:.4f} | KLw: {kl_weight:.4f} | "
                      f"Time: {epoch_time:.1f}s")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.encoder.state_dict(), self._encoder_state_fname)
                torch.save(self.decoder.state_dict(), self._decoder_state_fname)
                if epoch % printInterval != 0:
                    print(f"  -> Saved best model (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best={best_epoch})")
                break

        try:
            logger.plot_training_curves()
        except Exception as e:
            print(f"Warning: Could not plot training curves: {e}")

        print(f"Training complete. Best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f}")


# ===================== Training Logger =====================

class TrainingLogger:
    def __init__(self, base_dir=None):
        if base_dir:
            self.log_dir = Path(base_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            base_dir = "experiments"
            self.exp_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_dir = Path(base_dir) / self.exp_time
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_data = pd.DataFrame(columns=[
            'epoch', 'recon_loss', 'kld_loss',
            'cond_loss', 'cond_weight', 'kl_weight', 'total_loss', 'valid_rate', 'quality'
        ])

        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            plt.style.use('seaborn')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    def log_metrics(self, epoch, metrics_dict):
        new_row = pd.DataFrame([{'epoch': epoch, **metrics_dict}])
        self.log_data = pd.concat([self.log_data, new_row], ignore_index=True)
        self.save()

    def save(self):
        self.log_data.to_csv(self.log_dir / 'train_log.csv', index=False)

    def plot_training_curves(self):
        if len(self.log_data) < 2:
            return
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax = axes[0, 0]
        ax.plot(self.log_data['epoch'], self.log_data['total_loss'],
                color=self.colors[0], label='Total Loss')
        ax.set_title('Total Loss')
        ax.legend()

        ax = axes[0, 1]
        ax.plot(self.log_data['epoch'], self.log_data['recon_loss'],
                color=self.colors[0], label='Recon')
        ax.plot(self.log_data['epoch'], self.log_data['kld_loss'],
                color=self.colors[1], label='KLD')
        ax.set_title('Loss Components')
        ax.legend()

        ax = axes[1, 0]
        ax.plot(self.log_data['epoch'], self.log_data['valid_rate'],
                color=self.colors[2], label='Valid Rate')
        ax.set_title('Valid Rate')
        ax.legend()

        ax = axes[1, 1]
        ax.plot(self.log_data['epoch'], self.log_data['kl_weight'],
                color=self.colors[3], label='KL Weight')
        ax.set_title('KL Weight')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.log_dir / 'train_curves.png', dpi=150)
        plt.close()


# ===================== Testing Entry Point =====================

if __name__ == '__main__':
    yaml_path = 'config.yaml'
    cfg = load_training_config(yaml_path, model='cvae_film')
    tokenizer = utils.get_tokenizer(model='cvae_film')
    dataset = SmilesDictDataset(cfg['fname_dataset'], tokenizer, cfg['maxLength'])
    val_size = int(len(dataset) * 0.1)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FiLMConVAE(
        latent_dim=cfg['vae_param']['latent_dim'],
        hidden_dim=cfg['vae_param']['hidden_dim'],
        maxLength=cfg['maxLength'],
        num_vocabs=cfg['vae_param']['num_vocabs'],
        emb_dim=32,
        device=device,
    ).to(device)

    print(f"FiLM-CVAE Model initialized:")
    print(f"  Vocab size: {cfg['vae_param']['num_vocabs']}")
    print(f"  Latent dim: {cfg['vae_param']['latent_dim']}")
    print(f"  Hidden dim: {cfg['vae_param']['hidden_dim']}")
    print(f"  Device: {device}")