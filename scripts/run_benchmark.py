#!/usr/bin/env python3
"""Run an explicitly selected dataset/task/model/seed matrix sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", choices=("nc", "lp"), required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Hydra override; repeat for each setting, e.g. --override task.epochs=300",
    )
    args = parser.parse_args()

    failures = []
    for seed in args.seeds:
        for model in args.models:
            command = [
                sys.executable, "-m", "src.main",
                f"dataset={args.dataset}", f"task={args.task}",
                f"model={model}", f"seed={seed}", "num_runs=1",
                f"device={args.device}", *args.override,
            ]
            print("+", " ".join(command), flush=True)
            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            if result.returncode:
                failures.append((seed, model, result.returncode))

    if failures:
        for seed, model, code in failures:
            print(f"FAILED seed={seed} model={model} exit={code}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
