# Local / private tooling (optional)

This folder is **not** part of the main public documentation. It holds a **batch training driver** for multi-seed or small hyperparameter grids on your own machine.

- **Tracked copy (for clone / backup):** run from repo root  
  `python local.example/train_sweep.py --model cvae_dhr --seeds 42,43 --dry-run`
- **Fully private copy:** create a `local/` directory at the repo root (see root `.gitignore`), copy `train_sweep.py` there, and edit as you like — nothing under `local/` is versioned.

If you publish a minimal public fork and do not want this helper in the tree at all, delete the entire `local.example/` directory before pushing.

**Note:** `training_params/` and `generations/` are listed in the root `.gitignore` so local checkpoints and sweep logs are not committed by default.
