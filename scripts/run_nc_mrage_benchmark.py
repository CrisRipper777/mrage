#!/usr/bin/env python3
"""Run MRAGE on the five official NC splits with the selected tuning config."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_baseline_matrix import PROJECT_ROOT, SEEDS, _write_json_atomic
from run_nc_baseline_benchmark import DATASETS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/mrage_nc_benchmark_20261031"))
    parser.add_argument("--experiment-config", type=Path,
                        default=Path("experiments/nc5_tuning_20261030/selected_common.yaml"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    source = args.experiment_config.resolve()
    config_bytes = source.read_bytes()
    config_path = output / "_manifests" / "mrage_selected.yaml"
    commands = []
    for dataset in DATASETS:
        command = [sys.executable, str(PROJECT_ROOT / "scripts/run_paper_benchmark.py"),
                   "--dataset", dataset, "--task", "nc", "--models", "mrage",
                   "--seeds", *map(str, SEEDS), "--split-seed", "42",
                   "--device", args.device, "--stage", "all",
                   "--experiment-config", str(config_path), "--output-root", str(output)]
        if args.resume:
            command.append("--resume")
        commands.append(command)
    if args.dry_run:
        print(f"Selected config: {source}")
        for command in commands:
            print("+", " ".join(command))
        return 0

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(config_bytes)
    manifest_path = output / "_manifests/nc_mrage_benchmark.json"
    manifest = {"benchmark": "mrage_nc_selected_config", "datasets": list(DATASETS),
                "models": ["mrage"], "seeds": list(SEEDS), "split_seed": 42,
                "device": args.device, "source_config": str(source),
                "experiment_config": str(config_path),
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "protocol_version": "unified_full_graph_nc_v1",
                "selection": "Configuration frozen by prior Validation tuning; best Validation Accuracy checkpoint per run.",
                "commands": commands, "started_at_unix": time.time(),
                "status": "running", "completed_datasets": [], "failed_datasets": []}
    _write_json_atomic(manifest_path, manifest)
    log_path = output / "_logs/nc_mrage_benchmark.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(manifest, ensure_ascii=False) + "\n")
        log.flush()
        for dataset, command in zip(DATASETS, commands):
            print(f"START MRAGE NC {dataset} seeds={SEEDS} device={args.device}", flush=True)
            log.write("\n$ " + " ".join(command) + "\n")
            log.flush()
            result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, check=False,
                                    env={**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})
            key = "failed_datasets" if result.returncode else "completed_datasets"
            manifest[key].append(dataset)
            _write_json_atomic(manifest_path, manifest)
            print(f"DONE {dataset} exit={result.returncode}", flush=True)
    manifest.update(status="failed" if manifest["failed_datasets"] else "complete",
                    finished_at_unix=time.time())
    _write_json_atomic(manifest_path, manifest)
    print(f"Benchmark {manifest['status']}. Log: {log_path}", flush=True)
    return int(bool(manifest["failed_datasets"]))


if __name__ == "__main__":
    raise SystemExit(main())
