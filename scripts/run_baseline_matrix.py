#!/usr/bin/env python3
"""Shared launcher for the fixed benchmark matrices."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODELS = (
    "mlp",
    "gcn",
    "sage",
    "mmgcn",
    "mgat",
    "dip",
    "dgf",
    "dmgc",
    "rolemag",
    "lgmrec",
    "graphmore",
    "ntsformer",
)
LP_SUPPORTED_MODELS = tuple(model for model in BASELINE_MODELS if model != "ntsformer")
SEEDS = (42, 43, 44)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _parse_common_args(description: str, *, default_device: str = "cuda:0"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--device", default=default_device,
        help=f"Training device (default: {default_device}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/baseline_benchmark"),
        help="Root for checkpoints, per-seed logs, metrics, and sweep records.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only runs that have a saved validation checkpoint awaiting final Test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact dataset commands without starting training or writing files.",
    )
    return parser.parse_args()


def _launch_matrix(
    *,
    task: str,
    datasets: tuple[str, ...],
    models: tuple[str, ...],
    protocol_version: str,
    protocol_overrides: tuple[str, ...],
    protocol_description: dict,
    args,
    manifest_filename: str | None = None,
    log_filename: str | None = None,
    benchmark_name: str | None = None,
) -> int:
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()

    commands: list[list[str]] = []
    for dataset in datasets:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_paper_benchmark.py"),
            "--dataset",
            dataset,
            "--task",
            task,
            "--models",
            *models,
            "--seeds",
            *(str(seed) for seed in SEEDS),
            "--device",
            args.device,
            "--output-root",
            str(output_root),
            "--stage",
            "all",
        ]
        if task == "nc":
            command.extend(["--split-seed", "42"])
        if args.resume:
            command.append("--resume")
        for override in protocol_overrides:
            command.extend(["--override", override])
        commands.append(command)

    manifest_filename = manifest_filename or f"{task}_baseline_benchmark.json"
    log_filename = log_filename or f"{task}_baseline_benchmark.log"
    manifest_path = output_root / "_manifests" / manifest_filename
    manifest = {
        "benchmark": benchmark_name or f"non_mrage_{task}_baseline_matrix",
        "protocol_version": protocol_version,
        "datasets": list(datasets),
        "models": list(models),
        "seeds": list(SEEDS),
        "runs_per_model_dataset": len(SEEDS),
        "device": args.device,
        "output_root": str(output_root),
        "protocol": protocol_description,
        "overrides": list(protocol_overrides),
        "resume_requested": bool(args.resume),
        "started_at_unix": time.time(),
        "commands": [
            [
                part if part != str(output_root) else "<output-root>"
                for part in command
            ]
            for command in commands
        ],
    }

    print(f"Task: {task} | protocol: {protocol_version}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Models: {', '.join(models)}")
    print(f"Seeds: {', '.join(str(seed) for seed in SEEDS)}")
    print(f"Output root: {output_root}")
    if args.dry_run:
        for command in commands:
            print("+", " ".join(command), flush=True)
        return 0

    _write_json_atomic(manifest_path, manifest)
    log_path = output_root / "_logs" / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, int]] = []
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 80 + "\n")
        log.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        log.flush()
        for dataset, command in zip(datasets, commands):
            rendered_command = " ".join(command)
            print(f"\n[{task}] dataset={dataset}", flush=True)
            print("+", rendered_command, flush=True)
            log.write(f"\n[{task}] dataset={dataset}\n$ {rendered_command}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return_code = int(process.wait())
            if return_code:
                failures.append((dataset, return_code))
                log.write(f"\nFAILED dataset={dataset} exit={return_code}\n")
                print(f"FAILED dataset={dataset} exit={return_code}", file=sys.stderr, flush=True)
            else:
                log.write(f"\nCOMPLETE dataset={dataset}\n")
                print(f"COMPLETE dataset={dataset}", flush=True)
            log.flush()

    manifest["finished_at_unix"] = time.time()
    manifest["failed_datasets"] = [
        {"dataset": dataset, "exit_code": code} for dataset, code in failures
    ]
    manifest["status"] = "failed" if failures else "complete"
    _write_json_atomic(manifest_path, manifest)
    if failures:
        print(f"Some datasets failed. See {log_path} and per-run status.json files.", file=sys.stderr)
        return 1
    print(f"All runs completed. Sweep log: {log_path}")
    print(f"Sweep manifest: {manifest_path}")
    return 0
