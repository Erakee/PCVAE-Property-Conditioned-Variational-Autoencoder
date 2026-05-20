# EVAE_paper

Official implementation of **Property-Conditioned Variational Autoencoder (PCVAE)** for molecular generation of energetic materials with enthalpy control.

## Overview

This repository provides the training and generation code for the PCVAE model described in our paper. The model generates novel energetic molecules by conditioning on target heats of formation (EoF) through a dual-latent fusion architecture with FiLM-based property conditioning.

**Key features:**
- Conditional VAE with dual latent space (molecular structure + property)
- FiLM-based property conditioning in the decoder
- Gradual property-aware training strategy for stable optimization
- Pretrained GNN-based enthalpy prediction for property supervision

## Requirements

- Python >= 3.8
- PyTorch >= 1.13
- RDKit
- NumPy, Pandas, PyYAML, Matplotlib

## Installation

```bash
git clone https://github.com/Erakee/EVAE_paper.git
cd EVAE_paper
pip install -r requirements.txt
```

**Note:** RDKit is best installed via conda:
```bash
conda install -c rdkit rdkit
```

## Project Structure

```
EVAE_paper/
├── train.py                  # Unified training CLI
├── generate.py               # Unified generation CLI
├── generate_dhr.py           # Generation script for PCVAE (DHR)
├── config.yaml               # Project configuration
├── requirements.txt          # Python dependencies
├── data/
│   ├── em_train.csv          # Training dataset
│   ├── em_test.csv           # Test dataset
│   └── em_train.smi          # Token dictionary
├── model/
│   ├── CVAE_DHR.py           # PCVAE model (main model)
│   ├── CVAE_FiLM.py          # FiLM-variant PCVAE
│   ├── CVAE_HC_SEED.py       # CVAE baseline (enthalpy concat)
│   ├── VAE_H_SEED.py         # VAE baseline
│   └── mpnn.py               # MPNN for enthalpy prediction
├── enthalpy/
│   ├── stateGNN.pt           # Pretrained MPNN enthalpy predictor
│   └── graphdataset.py       # Graph dataset utilities
├── dataset/
│   └── dataset.py            # SMILES dataset and tokenizer
└── util/
    ├── utils.py              # Utility functions
    ├── enthalpy_predictor.py # Enthalpy prediction wrapper
    └── tokens.py             # Token processing
```

## Data Preparation

1. Place your training data in `data/` with columns `smiles` and `heat_of_formation`.
2. Place the SMILES token file (one SMILES per line) at `data/em_train.smi`.
3. The tokenizer file `.tokenizer.pkl` will be generated automatically on first run.

The default dataset used in our experiments is `data/em_train.csv`.

## Training

The unified training CLI is `train.py`:

```bash
# Train the main PCVAE model (CVAE-DHR)
python train.py --model cvae_dhr --seeds 42

# Train with multiple seeds
python train.py --model cvae_dhr --seeds 42 43 44

# Train baseline models for ablation
python train.py --model cvae_hc --seeds 42   # CVAE baseline
python train.py --model vae_h --seeds 42     # VAE baseline

# Override hyperparameters
python train.py --model cvae_dhr --lr 1e-3 --epochs 100 --batch-size 256

# Dry-run to check configuration
python train.py --model cvae_dhr --dry-run
```

Checkpoints are saved under `training_params/<MODEL>/seed_<n>/` by default.

### Key Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 256 | Training batch size |
| `lr` | 5e-3 | Initial learning rate |
| `num_epoch` | 100 | Total training epochs |
| `latent_dim` | 64 | Latent space dimension |
| `hidden_dim` | 256 | GRU hidden dimension |
| `embed_dim` | 32 | Embedding dimension |

## Generation

Generate molecules with target EoF values after training:

```bash
# Generate molecules using PCVAE
python generate_dhr.py --checkpoint training_params/CVAE_DHR/seed_42

# Generate with specific target EoF
python generate.py --model cvae_dhr --enthalpy -50 --num_samples 200
```

Generated SMILES and evaluation results are saved under `generations/`.

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{xxx,
  title={xxx},
  author={xxx},
  journal={xxx},
  year={2026}
}
```

[![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxxx-blue)](https://doi.org/10.xxxx/xxxxx)

## License

This project is released for academic research purposes.