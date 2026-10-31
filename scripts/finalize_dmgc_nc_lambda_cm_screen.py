#!/usr/bin/env python3
"""Run the single final Test phase for the validation-selected DMGC setting."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from run_nc_baseline_benchmark import PROTOCOL_OVERRIDES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Movies", "Grocery", "Toys", "Reddit-S", "ele-fashion")
SEEDS = (42, 43, 44)
SELECTED_LAMBDA_CM = 0.3


def main() -> int:
    output_root = PROJECT_ROOT / "outputs" / "dmgc_nc_lambda_cm_screen_20261031"
    candidate_root = output_root / f"lambda_cm_{SELECTED_LAMBDA_CM:g}"
    log_dir = output_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "selected_lambda_cm_final_test.log"
    manifest_path = output_root / "_manifests" / "selected_lambda_cm_final_test.json"

    base_overrides = [
        *PROTOCOL_OVERRIDES,
        "model.lr=0.005",
        "model.weight_decay=0.00001",
        f"model.lambda_cm={SELECTED_LAMBDA_CM:g}",
    ]
    manifest = {
        "experiment": "dmgc_nc_selected_lambda_cm_final_test",
        "selected_lambda_cm": SELECTED_LAMBDA_CM,
        "selection_basis": "Highest cross-dataset validation Accuracy with reduced mean seed standard deviation; Test was not used for selection.",
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "device": "cuda:1",
        "candidate_output_root": str(candidate_root),
        "started_at_unix": time.time(),
        "commands": [],
    }
    failures = []
    with log_path.open("a", encoding="utf-8") as log:
        for dataset in DATASETS:
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_paper_benchmark.py"),
                "--dataset", dataset,
                "--task", "nc",
                "--models", "dmgc",
                "--seeds", *(str(seed) for seed in SEEDS),
                "--split-seed", "42",
                "--device", "cuda:1",
                "--output-root", str(candidate_root),
                "--stage", "all",
                "--resume",
            ]
            for override in base_overrides:
                command.extend(["--override", override])
            manifest["commands"].append(command)
            header = f"\n[final-test] dataset={dataset}\n$ {' '.join(command)}\n"
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
                failures.append({"dataset": dataset, "exit_code": code})
            log.write(f"{'FAILED' if code else 'COMPLETE'} exit={code}\n")
            log.flush()

    manifest["finished_at_unix"] = time.time()
    manifest["failed_datasets"] = failures
    manifest["status"] = "failed" if failures else "complete"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Final-test manifest: {manifest_path}")
    print(f"Final-test log: {log_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
