#!/usr/bin/env python3
"""
Optional batch training (local / lab use — not described in the main README).

Runs ``training.launcher.run_training`` for a Cartesian grid of seeds × optional
``--lrs`` × optional ``--kld-alphas``. Each run directory still gets ``run_manifest.json``.

Also writes under ``<training_runs_root>/<MODEL>/_sweeps/<sweep_id>/``:
  - ``sweep_config.json`` — planned grid
  - ``sweep_results.jsonl`` — one JSON object per job (ok / error + paths)

Run from repository root::

  python local.example/train_sweep.py --model cvae_dhr --seeds 42,43 --dry-run
  python local.example/train_sweep.py --model cvae_dhr --seeds 42,43 --tag-prefix baseline
  python local.example/train_sweep.py --model cvae_dhr --seeds 42 --lrs 3e-3,5e-3 --tag-prefix lr_grid
  python local.example/train_sweep.py --model cvae_dhr --seeds 42 --kld-alphas 0.5,1.0,2.0
  python local.example/train_sweep.py --model cvae_dhr --seeds 42,43 --print-commands-only
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.config_loader import load_training_config  # noqa: E402
from training.launcher import MODEL_REGISTRY, run_training  # noqa: E402


def _slug_lr(x: float) -> str:
    return f"{x:g}".replace("-", "neg").replace(".", "p")


def _slug_kld(x: float) -> str:
    return _slug_lr(x)


def build_run_tag(
    tag_prefix: Optional[str],
    lr: Optional[float],
    kld_alpha: float,
    *,
    lr_is_default: bool,
    kld_is_default: bool,
) -> Optional[str]:
    parts: List[str] = []
    if tag_prefix:
        parts.append(tag_prefix.strip().replace(" ", "_"))
    if not lr_is_default and lr is not None:
        parts.append(f"lr{_slug_lr(lr)}")
    if not kld_is_default:
        parts.append(f"kld{_slug_kld(kld_alpha)}")
    if not parts:
        return None
    return "__".join(parts)


def _parse_float_list(s: Optional[str]) -> Optional[List[float]]:
    if s is None or not str(s).strip():
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def expand_grid(
    seeds: Sequence[int],
    lrs_opt: Optional[Sequence[float]],
    klds_opt: Optional[Sequence[float]],
) -> List[Tuple[int, Optional[float], float, bool, bool]]:
    lr_list: List[Optional[float]] = [None] if lrs_opt is None else [float(x) for x in lrs_opt]
    kld_list = [1.0] if klds_opt is None else [float(x) for x in klds_opt]
    rows: List[Tuple[int, Optional[float], float, bool, bool]] = []
    for seed in seeds:
        for lr in lr_list:
            for kld in kld_list:
                lr_def = lrs_opt is None
                kld_def = klds_opt is None
                rows.append((seed, lr, kld, lr_def, kld_def))
    return rows


def print_train_commands(
    model: str,
    rows: List[Tuple[int, Optional[float], float, bool, bool]],
    config_path: Optional[str],
    exp_root: Optional[str],
    tag_prefix: Optional[str],
    num_workers: int,
    print_interval: int,
    num_epoch: Optional[int],
    batch_size: Optional[int],
) -> None:
    for seed, lr, kld, lr_def, kld_def in rows:
        tag = build_run_tag(
            tag_prefix, lr, kld, lr_is_default=lr_def, kld_is_default=kld_def
        )
        parts = [
            "python",
            "train.py",
            "--model",
            model,
            "--seeds",
            str(seed),
        ]
        if config_path:
            parts += ["--config", config_path]
        if exp_root:
            parts += ["--exp-root", exp_root]
        if tag:
            parts += ["--tag", tag]
        if not lr_def and lr is not None:
            parts += ["--lr", str(lr)]
        if not kld_def:
            parts += ["--kld-alpha", str(kld)]
        if num_epoch is not None:
            parts += ["--epochs", str(num_epoch)]
        if batch_size is not None:
            parts += ["--batch-size", str(batch_size)]
        parts += ["--num-workers", str(num_workers), "--print-interval", str(print_interval)]
        print(" ".join(parts))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", choices=sorted(MODEL_REGISTRY.keys()), required=True)
    p.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated seeds (default: 42,43,44)",
    )
    p.add_argument(
        "--lrs",
        type=str,
        default=None,
        help="Comma-separated learning rates; omit to use config default for all runs",
    )
    p.add_argument(
        "--kld-alphas",
        type=str,
        default=None,
        help="Comma-separated KL weights; omit to use 1.0 for all runs",
    )
    p.add_argument(
        "--tag-prefix",
        type=str,
        default=None,
        help="Optional prefix for run tags (e.g. baseline_v2)",
    )
    p.add_argument("--sweep-id", type=str, default=None, help="Sweep folder name; default UTC timestamp")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--exp-root", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--print-interval", type=int, default=50)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--print-commands-only",
        action="store_true",
        help="Print equivalent train.py lines and exit",
    )
    args = p.parse_args()

    seeds = _parse_int_list(args.seeds)
    if not seeds:
        print("Error: --seeds must list at least one integer (comma-separated).", file=sys.stderr)
        sys.exit(2)

    lrs_opt = _parse_float_list(args.lrs)
    klds_opt = _parse_float_list(args.kld_alphas)

    meta = MODEL_REGISTRY[args.model]
    cfg = load_training_config(args.config, model=meta["config_model_arg"])

    rows = expand_grid(seeds, lrs_opt, klds_opt)
    if not rows:
        print("Error: empty run grid.", file=sys.stderr)
        sys.exit(2)

    sweep_id = args.sweep_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = Path(cfg["training_runs_root"]) / MODEL_REGISTRY[args.model]["output_subdir"] / "_sweeps" / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    sweep_meta: Dict[str, Any] = {
        "sweep_id": sweep_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "config": args.config,
        "exp_root_cli": args.exp_root,
        "seeds": seeds,
        "lrs_requested": lrs_opt,
        "kld_alphas_requested": klds_opt,
        "base_lr_from_config": float(cfg["lr"]),
        "default_kld_if_omitted": 1.0,
        "tag_prefix": args.tag_prefix,
        "n_runs": len(rows),
        "rows": [
            {
                "seed": r[0],
                "lr": r[1],
                "kld_alpha": r[2],
                "lr_is_default": r[3],
                "kld_is_default": r[4],
                "tag": build_run_tag(
                    args.tag_prefix,
                    r[1],
                    r[2],
                    lr_is_default=r[3],
                    kld_is_default=r[4],
                ),
            }
            for r in rows
        ],
    }
    (sweep_dir / "sweep_config.json").write_text(
        json.dumps(sweep_meta, indent=2, default=str), encoding="utf-8"
    )

    if args.print_commands_only:
        print("# Equivalent commands:\n")
        print_train_commands(
            args.model,
            rows,
            args.config,
            args.exp_root,
            args.tag_prefix,
            args.num_workers,
            args.print_interval,
            args.epochs,
            args.batch_size,
        )
        print(f"\n# Sweep metadata written to: {sweep_dir / 'sweep_config.json'}")
        return

    if args.dry_run:
        print(f"Sweep dir: {sweep_dir}")
        for seed, lr, kld, lr_def, kld_def in rows:
            tag = build_run_tag(
                args.tag_prefix,
                lr,
                kld,
                lr_is_default=lr_def,
                kld_is_default=kld_def,
            )
            run_training(
                args.model,
                seed,
                config_path=args.config,
                exp_root=args.exp_root,
                tag=tag,
                num_workers=args.num_workers,
                print_interval=args.print_interval,
                kld_alpha=kld,
                lr=lr if not lr_def else None,
                num_epoch=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                dry_run=True,
            )
        return

    results_path = sweep_dir / "sweep_results.jsonl"
    with results_path.open("w", encoding="utf-8") as jlog:
        for i, (seed, lr, kld, lr_def, kld_def) in enumerate(rows, 1):
            tag = build_run_tag(
                args.tag_prefix,
                lr,
                kld,
                lr_is_default=lr_def,
                kld_is_default=kld_def,
            )
            print(f"\n[{i}/{len(rows)}] model={args.model} seed={seed} lr={lr!s} kld={kld} tag={tag!r}\n")
            rec: Dict[str, Any] = {
                "index": i,
                "model": args.model,
                "seed": seed,
                "lr": lr,
                "kld_alpha": kld,
                "tag": tag,
                "status": "pending",
            }
            try:
                exp_dir = run_training(
                    args.model,
                    seed,
                    config_path=args.config,
                    exp_root=args.exp_root,
                    tag=tag,
                    num_workers=args.num_workers,
                    print_interval=args.print_interval,
                    kld_alpha=kld,
                    lr=lr if not lr_def else None,
                    num_epoch=args.epochs,
                    batch_size=args.batch_size,
                    device=args.device,
                    dry_run=False,
                )
                rec["status"] = "ok"
                rec["exp_dir"] = str(exp_dir) if exp_dir else None
            except Exception as e:  # noqa: BLE001
                rec["status"] = "error"
                rec["error"] = str(e)
                rec["traceback"] = traceback.format_exc()
                print(f"ERROR: {e}", file=sys.stderr)
            jlog.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            jlog.flush()

    print(f"\nSweep finished. Summary log: {results_path}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
