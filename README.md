# EVAE: Enthalpy-guided Variational Autoencoder for Energetic Molecule Generation

This repository contains the implementation of EVAE, a conditional variational autoencoder framework for the design of energetic materials. The model generates novel energetic molecules by combining latent space sampling with enthalpy guidance via a message-passing neural network (MPNN) predictor.

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Generation](#generation)
- [Configuration](#configuration)
- [Pre-trained Enthalpy Predictor](#pre-trained-enthalpy-predictor)
- [Citation](#citation)
- [License](#license)

## Project Structure

```
EVAE_paper/
├── config.yaml                  # Central configuration file
├── train.py                     # Unified training entry point (all models)
├── train_CVAE_HC_SEED.py        # CVAE (HC backbone) training script
├── train_CVAE_DHR.py            # CVAE (DHR backbone) training script
├── train_CVAE_FiLM.py           # CVAE (FiLM backbone) training script
├── train_VAE_H_SEED.py          # VAE baseline training script
├── generate.py                  # Unified generation entry point
├── generate_hc.py               # Generation with HC model
├── generate_dhr.py              # Generation with DHR model
├── generate_film.py             # Generation with FiLM model
├── model/
│   ├── VAE_H_SEED.py            # VAE (encoder + decoder)
│   ├── VAE_H.py                 # Conditional VAE variant
│   ├── CVAE_HC_SEED.py          # CVAE with Hierarchical Context backbone
│   ├── CVAE_DHR.py              # CVAE with Dynamic Hidden Representation backbone
│   ├── CVAE_FiLM.py             # CVAE with FiLM conditioning backbone
│   └── mpnn.py                  # MPNN for enthalpy prediction
├── dataset/
│   └── dataset.py               # Dataset loading and preprocessing
├── training/
│   ├── __init__.py
│   └── launcher.py              # Multi-seed training launcher
├── util/
│   ├── tokens.py                # Tokenizer (SMILES <-> indices)
│   ├── utils.py                 # General utilities
│   ├── config_loader.py         # YAML config loader
│   ├── checkpoint_paths.py      # Checkpoint path resolution
│   ├── enthalpy_predictor.py    # Enthalpy prediction interface
│   ├── train_logger.py          # Training logger
│   ├── similarity_calculation.py
│   ├── generate_fps.py
│   └── csv_to_smi.py
├── data/
│   ├── em_train.csv             # Training dataset
│   ├── em_test.csv              # Test dataset
│   ├── em_train.smi             # SMILES vocabulary
│   └── database_CHON.csv        # Full CHON database
├── enthalpy/
│   ├── stateGNN.pt              # Pre-trained MPNN weights
│   └── graphdataset.py          # Graph dataset for MPNN
├── parameters/                  # Model checkpoint directory (created after training)
└── README.md
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- PyTorch Geometric >= 2.3.0
- RDKit
- See `requirements.txt` for the full list

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Erakee/EVAE_paper.git
cd EVAE_paper

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install torch-scatter and torch-geometric (follow official instructions if needed)
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

## Data Preparation

1. Place your training data CSV file in the `data/` directory. The default expected file is `data/em_train.csv`.
2. The CSV should contain at least the following columns:
   - **SMILES**: molecular SMILES string
   - **enthalpy**: formation enthalpy value
3. Prepare the SMILES vocabulary file (`data/em_train.smi`) — one SMILES per line.
4. The tokenizer will be automatically generated from the SMILES file on first run.

## Training

All model variants share the same `config.yaml` for hyperparameters.

### Using the Unified Entry Point

```bash
# Train all model variants with default settings
python train.py

# Train a specific model variant
python train.py --model cvae_dhr      # CVAE-DHR
python train.py --model cvae_hc       # CVAE-HC
python train.py --model cvae_film     # CVAE-FiLM
python train.py --model vae_h         # VAE baseline (no conditioning)
```

### Using Individual Scripts

```bash
# CVAE-HC
python train_CVAE_HC_SEED.py

# CVAE-DHR
python train_CVAE_DHR.py

# CVAE-FiLM
python train_CVAE_FiLM.py

# VAE baseline
python train_VAE_H_SEED.py
```

### Multi-Seed Training

```bash
# Train with multiple random seeds for reproducibility
python training/launcher.py
```

Training checkpoints are saved to `training_params/<ModelName>/seed_<N>/` by default. You can override the output directory via the `training_runs_root` field in `config.yaml` or the `EVAE_TRAINING_RUNS_ROOT` environment variable.

## Generation

After training, generate novel molecules from the learned latent space:

```bash
# Unified generation
python generate.py --model cvae_dhr --checkpoint training_params/CVAE_DHR/seed_42 --nsamples 1000

# Individual generation scripts
python generate_hc.py
python generate_dhr.py
python generate_film.py
```

Generated molecules are saved to the `generations/` directory by default. Override with `generation_output_root` in `config.yaml` or the `EVAE_GENERATION_OUTPUT_ROOT` environment variable.

## Configuration

All hyperparameters are managed through `config.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `maxLength` | Maximum SMILES length | 64 |
| `batch_size` | Training batch size | 256 |
| `num_epoch` | Number of training epochs | 100 |
| `lr` | Learning rate | 5e-3 |
| `embed_dim` | Embedding dimension | 32 |
| `vae_param.latent_dim` | Latent space dimension | 64 |
| `vae_param.hidden_dim` | GRU hidden dimension | 256 |
| `vae_param.num_hidden` | Number of GRU layers | 4 |
| `vae_param.fc_dims` | FC layer dimensions | [2048, 512, 256, 128] |

## Pre-trained Enthalpy Predictor

A pre-trained MPNN model for enthalpy prediction is provided at `enthalpy/stateGNN.pt`. This model is used during CVAE training to guide the latent space. The predictor can also be used standalone:

```python
from util.enthalpy_predictor import predict_enthalpy

enthalpy = predict_enthalpy("C1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]")
print(f"Predicted enthalpy: {enthalpy}")
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{EVAE2025,
  title={EVAE: Enthalpy-guided Variational Autoencoder for Energetic Molecule Generation},
  author={},
  journal={},
  year={2025}
}
```

<!-- When the paper is published, replace the above with the actual DOI -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.