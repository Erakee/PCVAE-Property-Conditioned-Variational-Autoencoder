#!/usr/bin/env python3
"""
Unified training entry for this project.

Examples
--------
  python train.py --model cvae_dhr
  python train.py --model cvae_dhr --seeds 42 43 --tag baseline
  python train.py --model vae_h --config path/to/config.yaml --lr 1e-3 --epochs 50
  python train.py --list-models
  python train.py --model cvae_dhr --dry-run
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.launcher import list_models, run_training  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train VAE / CVAE models with a single entry point.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--model',
        choices=['cvae_dhr', 'cvae_hc', 'vae_h'],
        help='Model and code path to use (see --list-models).',
    )
    p.add_argument(
        '--config',
        default=None,
        help='YAML config file. Default: <repo>/config.yaml',
    )
    p.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42],
        help='One or more random seeds; each gets its own output folder.',
    )
    p.add_argument(
        '--exp-root',
        default=None,
        help='Override training_runs_root from config (default: config.yaml / EVAE_TRAINING_RUNS_ROOT).',
    )
    p.add_argument(
        '--tag',
        default=None,
        help='Optional subfolder under the model directory (e.g. ablation name).',
    )
    p.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    p.add_argument('--print-interval', type=int, default=50, help='Log every N batches')
    p.add_argument('--kld-alpha', type=float, default=1.0, help='Weight for KL term')
    p.add_argument('--lr', type=float, default=None, help='Override learning rate')
    p.add_argument('--epochs', type=int, default=None, help='Override num_epoch')
    p.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    p.add_argument(
        '--device',
        default=None,
        help='cpu or cuda. Default: cuda if available else cpu.',
    )
    p.add_argument(
        '--dry-run',
        action='store_true',
        help='Load config and print paths; do not train.',
    )
    p.add_argument(
        '--list-models',
        action='store_true',
        help='Show registered models and exit.',
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list_models:
        print(list_models())
        return
    if not args.model:
        print('Error: --model is required (unless using --list-models).', file=sys.stderr)
        sys.exit(2)

    for seed in args.seeds:
        print(f'\n=== Training model={args.model!r} seed={seed} ===\n')
        run_training(
            args.model,
            seed,
            config_path=args.config,
            exp_root=args.exp_root,
            tag=args.tag,
            num_workers=args.num_workers,
            print_interval=args.print_interval,
            kld_alpha=args.kld_alpha,
            lr=args.lr,
            num_epoch=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
