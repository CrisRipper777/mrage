#!/usr/bin/env python3
"""Safely schedule the 5-dataset, 4-variant, 3-seed capability ablation."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADOPTED_ROOT = PROJECT_ROOT / "experiments/nc5_test_exploration_20261031/adopted"
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
VARIANTS = (
    "physical_only",
    "semantic_only",
    "no_cross_modal_collaboration",
    "feature_only",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/mrage_capability_nc5")
    )
    parser.add_argument("--stage", choices=("validation", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands and resolved run directories"
    )
    parser.add_argument(
        "--execute", action="store_true", help="required to launch real training/evaluation"
    )
    args = parser.parse_args()
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute cannot be combined")
    for name, values in (
        ("--datasets", args.datasets),
        ("--variants", args.variants),
        ("--seeds", args.seeds),
    ):
        if not values:
            parser.error(f"{name} cannot be empty")
        if len(set(values)) != len(values):
            parser.error(f"{name} must not contain duplicates")
    return args


def _command(args: argparse.Namespace, dataset: str, variant: str) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_paper_benchmark.py"),
        "--dataset", dataset,
        "--task", "nc",
        "--models", "mrage_capability_ablation",
        "--seeds", *(str(seed) for seed in args.seeds),
        "--split-seed", "42",
        "--device", args.device,
        "--experiment-config", str(ADOPTED_ROOT / f"{dataset}.yaml"),
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
    run_count = len(args.datasets) * len(args.variants) * len(args.seeds)
    print(
        f"{'DRY RUN (no training will start)' if dry_run else 'EXECUTE'}: "
        f"{run_count} seed-runs | device={args.device} | stage={args.stage}",
        flush=True,
    )
    if dry_run:
        print("Add --execute only after reviewing the commands and paths.", flush=True)

    failures: list[tuple[str, str, int]] = []
    for dataset in args.datasets:
        for variant in args.variants:
            command = _command(args, dataset, variant)
            print(f"\n[{dataset} / {variant}]\n$ {shlex.join(command)}", flush=True)
            delegated = [*command, "--dry-run"] if dry_run else command
            result = subprocess.run(delegated, cwd=PROJECT_ROOT, check=False)
            if result.returncode:
                failures.append((dataset, variant, result.returncode))
                print(
                    f"FAILED dataset={dataset} variant={variant} exit={result.returncode}",
                    file=sys.stderr,
                    flush=True,
                )
    if failures:
        print("\nFailed matrix cells:", file=sys.stderr)
        for dataset, variant, code in failures:
            print(f"  {dataset} / {variant}: exit={code}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
