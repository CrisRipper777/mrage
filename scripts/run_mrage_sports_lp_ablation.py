#!/usr/bin/env python3
"""Run Sports LP MRAGE ablations with an explicit execute gate.

The default matrix is Single Filter plus the four capability variants, seeds
42-44. Training and validation-selected, one-use Final Test stay in the
existing paper benchmark runner.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = "sports-copurchase"
EXPERIMENT_CONFIG = (
    PROJECT_ROOT / "experiments/lp_recommended_20261031/mrage_lp_default_h256.yaml"
)
VARIANTS = (
    "single_filter",
    "physical_only",
    "semantic_only",
    "no_cross_modal_collaboration",
    "feature_only",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/mrage_sports_lp_ablation")
    )
    parser.add_argument("--stage", choices=("validation", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--execute", action="store_true", help="required to start LP training/evaluation"
    )
    args = parser.parse_args()
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute cannot be combined")
    for name, values in (("--variants", args.variants), ("--seeds", args.seeds)):
        if not values:
            parser.error(f"{name} cannot be empty")
        if len(set(values)) != len(values):
            parser.error(f"{name} must not contain duplicates")
    return args


def _command(args: argparse.Namespace, variant: str) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_paper_benchmark.py"),
        "--dataset", DATASET,
        "--task", "lp",
        "--models", "mrage_lp_ablation",
        "--seeds", *(str(seed) for seed in args.seeds),
        "--device", args.device,
        "--experiment-config", str(EXPERIMENT_CONFIG),
        "--override", f"model.ablation_variant={variant}",
        "--output-root", str(args.output_root),
        "--stage", args.stage,
    ]
    if args.resume:
        command.append("--resume")
    return command


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    args.output_root = output_root
    dry_run = not args.execute
    run_count = len(args.variants) * len(args.seeds)
    mode = "DRY RUN (no training will start)" if dry_run else "EXECUTE"
    print(
        f"{mode}: {run_count} seed-runs | dataset={DATASET} | device={args.device} "
        f"| protocol=unified_sampled_lp_v1 | stage={args.stage}",
        flush=True,
    )
    if dry_run:
        print("Add --execute only after reviewing the commands and output paths.", flush=True)

    failures: list[tuple[str, int]] = []
    for variant in args.variants:
        command = _command(args, variant)
        print(f"\n[{DATASET} / {variant}]\n$ {shlex.join(command)}", flush=True)
        delegated = [*command, "--dry-run"] if dry_run else command
        result = subprocess.run(delegated, cwd=PROJECT_ROOT, check=False)
        if result.returncode:
            failures.append((variant, result.returncode))
            print(f"FAILED variant={variant} exit={result.returncode}", file=sys.stderr, flush=True)

    if failures:
        print("\nFailed variants:", file=sys.stderr)
        for variant, code in failures:
            print(f"  {variant}: exit={code}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
