#!/usr/bin/env python3
"""
Unified generation entry: select the same ``--model`` key as ``train.py``,
then forwards to ``generate_dhr`` or ``generate_hc`` (pass-through CLI args).

Examples
--------
  python generate.py --model cvae_dhr --checkpoint dhr_seed42 --enthalpy -50
  python generate.py --model cvae_hc --checkpoint training_params/CVAE_HC/seed_42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Route molecular generation to the right script (see config.yaml).',
    )
    parser.add_argument(
        '--model',
        choices=['cvae_dhr', 'cvae_hc'],
        required=True,
        help='Same model key as train.py; picks encoder/decoder architecture.',
    )
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    if args.model == 'cvae_dhr':
        import generate_dhr
        generate_dhr.main()
    else:
        import generate_hc
        generate_hc.main()


if __name__ == '__main__':
    main()
