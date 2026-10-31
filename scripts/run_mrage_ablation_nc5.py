#!/usr/bin/env python3
"""Safely schedule the 5x5x3 MRAGE NC ablation matrix.

Training is disabled unless ``--execute`` is explicitly provided. The actual
training/evaluation remains delegated to run_paper_benchmark.py.
"""

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
    "single_filter",
    "unshared_experts",
    "no_semantic_propagation",
    "no_topological_collaboration",
    "no_semantic_refinement",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output-root", type=Path, default=Path("outputs/mrage_ablation_nc5"))
    parser.add_argument("--stage", choices=("validation", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print commands and exact run directories")
    parser.add_argument("--execute", action="store_true", help="required to launch training/evaluation")
    args = parser.parse_args()
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute cannot be combined")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets must not contain duplicates")
    if len(set(args.variants)) != len(args.variants):
        parser.error("--variants must not contain duplicates")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if not args.seeds:
        parser.error("--seeds cannot be empty")
    return args


def _command(args: argparse.Namespace, dataset: str, variant: str) -> list[str]:
    adopted = ADOPTED_ROOT / f"{dataset}.yaml"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_paper_benchmark.py"),
        "--dataset", dataset,
        "--task", "nc",
        "--models", "mrage_ablation",
        "--seeds", *(str(seed) for seed in args.seeds),
        "--split-seed", "42",
        "--device", args.device,
        "--experiment-config", str(adopted),
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
    cells = len(args.datasets) * len(args.variants) * len(args.seeds)
    mode = "DRY RUN (no training will start)" if dry_run else "EXECUTE"
    print(f"{mode}: {cells} seed-runs | device={args.device} | stage={args.stage}", flush=True)
    if dry_run:
        print("Add --execute to launch these commands.", flush=True)

    failures: list[tuple[str, str, int]] = []
    for dataset in args.datasets:
        for variant in args.variants:
            command = _command(args, dataset, variant)
            print(f"\n[{dataset} / {variant}]\n$ {shlex.join(command)}", flush=True)
            run_command = [*command, "--dry-run"] if dry_run else command
            completed = subprocess.run(run_command, cwd=PROJECT_ROOT, check=False)
            if completed.returncode:
                failures.append((dataset, variant, completed.returncode))
                print(
                    f"FAILED dataset={dataset} variant={variant} exit={completed.returncode}",
                    file=sys.stderr,
                    flush=True,
                )
                if not dry_run:
                    continue
    if failures:
        print("\nFailed cells:", file=sys.stderr)
        for dataset, variant, code in failures:
            print(f"  {dataset} / {variant}: exit={code}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
