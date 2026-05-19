import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.launcher import run_training


def main():
    seed = 30
    print(f"\n{'=' * 40}\nRunning FiLM-CVAE with seed: {seed}\n{'=' * 40}")
    run_training('cvae_film', seed, tag='film_baseline', num_epoch=40)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()