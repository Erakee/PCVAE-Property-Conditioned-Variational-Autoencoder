# EVAE_paper

Variational autoencoders and conditional VAEs for **SMILES-based molecular generation**, with optional conditioning on **heat of formation (enthalpy)**. The repo includes training code, evaluation utilities, and an optional inference-oriented package (`pcvae_tool`).

## Features

- **Models**
  - **`vae_h`** — vanilla VAE on one-hot SMILES (see `model/VAE_H_SEED.py`).
  - **`cvae_hc`** — conditional VAE with enthalpy concatenated in the encoder (`model/CVAE_HC_SEED.py`).
  - **`cvae_dhr`** — conditional VAE with embedding encoder, GRU decoder, and enthalpy-dependent prior blending (`model/CVAE_DHR.py`).
- **Unified training CLI** — `train.py` selects the model, seeds, output layout, and hyperparameter overrides.
- **Portable config** — `config.yaml` uses `root_path: '.'` so paths resolve relative to the config file (no hard-coded drive letters).

## Requirements

Core dependencies (training & dataset tooling):

- Python 3.8+ recommended  
- [PyTorch](https://pytorch.org/) (CUDA build optional; match your GPU drivers)
- `numpy`, `pandas`, `PyYAML`
- [RDKit](https://www.rdkit.org/)
- `matplotlib` (training logs / plots in some model modules)
- `dscribe`, `ase` (SOAP / descriptor paths in `dataset` and `util`)

For **graph-based enthalpy prediction** and the `pcvae_tool` client, you also need PyTorch Geometric and related wheels; see `pcvae_tool/requirements.txt`.

## Installation

```bash
git clone <your-fork-or-upstream-url> EVAE_paper
cd EVAE_paper
```

Install PyTorch for your platform from the [official install matrix](https://pytorch.org/get-started/locally/), then install the rest, for example:

```bash
pip install numpy pandas pyyaml matplotlib rdkit dscribe ase
# Optional: pcvae_tool extras
pip install -r pcvae_tool/requirements.txt
```

Place your dataset and token list under the project root (or adjust `config.yaml`). Expected default names are under `data/` (see below).

## Data & configuration

- **CSV** — e.g. `data/em_train.csv` with at least `smiles` and `heat_of_formation` (see `config.yaml` → `fname_dataset`).
- **Token dictionary** — e.g. `data/em_train.smi` used to build `.tokenizer.pkl` (see `token_file` / `fname_tokenizer` in `config.yaml`).
- **Main config** — `config.yaml` at the repository root. Key fields: `maxLength`, `batch_size`, `num_epoch`, `lr`, `vae_param` (hidden sizes, `latent_dim`, `num_vocabs` must match the tokenizer for the chosen model).

Training runs resolve `root_path` relative to the YAML file, so moving the repo to another machine only requires valid relative paths inside that root.

## Training

From the repository root:

```bash
# List registered model keys
python train.py --list-models

# Train CVAE-DHR (default config, seed 42)
python train.py --model cvae_dhr

# Multiple seeds and a run tag (output: training_params/CVAE_DHR/<tag>/seed_<n>/)
python train.py --model cvae_dhr --seeds 42 43 --tag baseline

# Override hyperparameters
python train.py --model vae_h --lr 1e-3 --epochs 50 --batch-size 128

# Validate config paths without importing PyTorch training stack
python train.py --model cvae_dhr --dry-run
```

**Where checkpoints are saved** is controlled by `training_runs_root` in `config.yaml` (default: `training_params`, under `root_path`). You can override the whole root with the environment variable `EVAE_TRAINING_RUNS_ROOT`, or a single run with `python train.py --exp-root /path/to/runs ...`. Trained runs look like: `<training_runs_root>/<MODEL_DIR>/[tag/]seed_<seed>/` with `encoder.pt`, `decoder.pt`, and `run_manifest.json`.

**Generation** uses the same `config.yaml` to find weights and to separate output files by model:

- `default_generation_checkpoint` — per-model default folder (relative to `root_path`) containing `encoder.pt` and `decoder.pt`. Update these after training.
- `checkpoint_aliases` — short names (e.g. `dhr_seed42`) that you can pass as `--checkpoint` instead of a long path.
- `generation_output_root` — base directory for generated files; each model writes under `<generation_output_root>/<model_key>/...`. Override with `EVAE_GENERATION_OUTPUT_ROOT` if needed.

```bash
# Same --model as train.py; uses default_generation_checkpoint unless you pass --checkpoint
python generate.py --model cvae_dhr --checkpoint dhr_seed42 --enthalpy -50 --num_samples 200
python generate_dhr.py --checkpoint training_params/CVAE_DHR/seed_42
python generate_hc.py --checkpoint training_params/CVAE_HC/seed_42
```

You do **not** need a `.env` file: use `config.yaml` for project defaults and optional `EVAE_*` environment variables for machine-specific absolute paths.

Legacy entry points `train_CVAE_DHR.py`, `train_CVAE_HC_SEED.py`, and `train_VAE_H_SEED.py` remain as thin wrappers around the same launcher.

## Project layout (overview)

| Path | Purpose |
|------|---------|
| `train.py` | CLI for all training jobs |
| `training/launcher.py` | Model registry, dataloaders, optimizers |
| `util/config_loader.py` | YAML loading without PyTorch (supports `--dry-run`) |
| `model/` | Encoder / decoder definitions |
| `dataset/` | `SmilesDataset`, `SmilesDictDataset`, hybrid/SOAP helpers |
| `util/` | Tokenizer helpers, enthalpy predictor, logging |
| `visualization/` | Analysis and plotting scripts |
| `pcvae_tool/` | Optional inference-oriented utilities |

## Citation

If you use this code in academic work, please cite your associated publication once it is available, and reference this repository URL.

## License

Specify your license here (no `LICENSE` file is included in the repository by default).

---

## 中文简介

本项目用于基于 **SMILES** 的分子生成，结合 **生成焓（条件）** 的变分自编码器（VAE / CVAE）实验代码。统一训练入口为 **`python train.py`**，生成可用 **`python generate.py --model ...`** 或 `generate_dhr.py` / `generate_hc.py`。训练输出目录、生成时加载的默认权重、各模型结果子目录均在 **`config.yaml`** 中配置；也可用环境变量 **`EVAE_TRAINING_RUNS_ROOT`** / **`EVAE_GENERATION_OUTPUT_ROOT`** 覆盖本机绝对路径。详细命令见上文。
