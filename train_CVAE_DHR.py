"""Backward-compatible wrapper. Prefer: python train.py --model cvae_dhr"""
import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.launcher import run_training


def main():
    # v7: kl_weight_max=0.2, cond_weight_max=0.5, max_epoch=120
    for seed in [30, 42, 93, 123, 256]:
        print(f"\n{'=' * 40}\nRunning experiment with seed: {seed}\n{'=' * 40}")
        run_training('cvae_dhr', seed, tag='260519test_v7', num_epoch=120)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()