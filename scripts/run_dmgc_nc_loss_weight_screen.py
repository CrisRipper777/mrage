#!/usr/bin/env python3
"""Validation-only DMGC cross-modal loss-weight screen on three NC datasets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from run_nc_baseline_benchmark import PROTOCOL_OVERRIDES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Movies", "Grocery", "Toys", "Reddit-S", "ele-fashion")
SEEDS = (42, 43, 44)
LAMBDA_CM_VALUES = (0.3, 0.1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/dmgc_nc_lambda_cm_screen_20261031"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "validation_screen.log"

    base_overrides = [
        *PROTOCOL_OVERRIDES,
        "model.lr=0.005",
        "model.weight_decay=0.00001",
    ]
    jobs = []
    for lambda_cm in LAMBDA_CM_VALUES:
        candidate_root = output_root / f"lambda_cm_{lambda_cm:g}"
        for dataset in DATASETS:
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_paper_benchmark.py"),
                "--dataset", dataset,
                "--task", "nc",
                "--models", "dmgc",
                "--seeds", *(str(seed) for seed in SEEDS),
                "--split-seed", "42",
                "--device", args.device,
                "--output-root", str(candidate_root),
                "--stage", "validation",
                "--override", f"model.lambda_cm={lambda_cm}",
            ]
            for override in base_overrides:
                command.extend(["--override", override])
            if args.resume:
                command.append("--resume")
            jobs.append((lambda_cm, dataset, command))

    manifest = {
        "experiment": "dmgc_nc_lambda_cm_validation_screen",
        "selection_metric": "Validation Accuracy",
        "datasets": list(DATASETS),
        "models": ["dmgc"],
        "seeds": list(SEEDS),
        "lambda_cm_values": list(LAMBDA_CM_VALUES),
        "fixed_model_optimizer": {"optimizer": "AdamW", "lr": 0.005, "weight_decay": 1e-5},
        "test_evaluation": False,
        "device": args.device,
        "output_root": str(output_root),
        "started_at_unix": time.time(),
        "commands": [command for _, _, command in jobs],
    }
    (output_root / "_manifests").mkdir(exist_ok=True)
    manifest_path = output_root / "_manifests" / "validation_screen.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    failures = []
    with log_path.open("a", encoding="utf-8") as log:
        for lambda_cm, dataset, command in jobs:
            header = f"\n[lambda_cm={lambda_cm:g}] dataset={dataset}\n$ {' '.join(command)}\n"
            print(header, end="", flush=True)
            log.write(header)
            log.flush()
            process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = int(process.wait())
            if code:
                failures.append({"lambda_cm": lambda_cm, "dataset": dataset, "exit_code": code})
                print(f"FAILED lambda_cm={lambda_cm:g} dataset={dataset} exit={code}", flush=True)
                log.write(f"FAILED exit={code}\n")
            else:
                print(f"VALIDATION_COMPLETE lambda_cm={lambda_cm:g} dataset={dataset}", flush=True)
                log.write("VALIDATION_COMPLETE\n")
            log.flush()

    manifest["finished_at_unix"] = time.time()
    manifest["failed_jobs"] = failures
    manifest["status"] = "failed" if failures else "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Validation screen manifest: {manifest_path}")
    print(f"Validation screen log: {log_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
