#!/usr/bin/env python3
"""Run recorded MRAGE NC5 parameter trials through the existing paper runner."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import time

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.run_artifacts import write_json_atomic


def snapshot_base(study: Path) -> None:
    for source, name in (
        (ROOT / "configs/model/mrage.yaml", "base_model.yaml"),
        (ROOT / "configs/task/nc.yaml", "base_task_nc.yaml"),
    ):
        target = study / name
        if not target.exists():
            target.write_bytes(source.read_bytes())
    path = study / "study_metadata.json"
    if not path.exists():
        write_json_atomic(path, {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "created_at_unix": time.time(),
            "original_mrage_sha256": hashlib.sha256(
                (ROOT / "configs/model/mrage.yaml").read_bytes()
            ).hexdigest(),
            "selection": "Validation Accuracy; Test only after frozen selection",
        })


def candidate_file(study: Path, manifest, candidate: str) -> Path:
    cfg = OmegaConf.create({
        "model": OmegaConf.to_container(OmegaConf.load(study / "base_model.yaml")),
        "task": OmegaConf.to_container(OmegaConf.load(study / "base_task_nc.yaml")),
    })
    cfg = OmegaConf.merge(
        cfg, manifest.get("common_overrides", {}),
        manifest.candidates[candidate].get("overrides", {}),
    )
    path = study / "candidates" / f"{candidate}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = OmegaConf.to_yaml(cfg)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"candidate changed; use a new candidate ID: {candidate}")
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    return path


def check_idle_gpu(device: str) -> None:
    index = device.split(":")[-1]
    result = subprocess.run(
        ["nvidia-smi", "-i", index, "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(f"{device} has an existing compute process; stopping before launch: {result.stdout.strip()}")


def write_ledger(study: Path, rows: list[dict]) -> None:
    write_json_atomic(study / "trials.json", {"runs": rows})
    columns = [
        "candidate", "dataset", "seed", "device", "physical_device", "cuda_visible_devices", "phase", "status", "val_acc", "val_macro_f1",
        "test_acc", "test_macro_f1", "best_epoch", "wall_seconds", "model_parameters",
        "config_hash", "split_id", "run_dir", "error",
    ]
    with (study / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    datasets = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
    representatives = ["Movies", "Grocery", "ele-fashion"]
    lines = [
        "# Validation search progress", "",
        "Generated from completed Validation runs. Each cell reports Accuracy percent and the number of available training seeds. Test is excluded.", "",
        "| Candidate | Movies | Toys | Grocery | ele-fashion | Reddit-S | Representative mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in dict.fromkeys(row["candidate"] for row in rows):
        values = {dataset: [row["val_acc"] for row in rows
                           if row["candidate"] == name and row["dataset"] == dataset
                           and row["phase"] == "validation" and row["status"] == "validation_complete"]
                  for dataset in datasets}
        means = {dataset: sum(items) / len(items) for dataset, items in values.items() if items}
        cells = [f"{means[dataset] * 100:.3f} (n={len(values[dataset])})" if dataset in means else "pending"
                 for dataset in datasets]
        aggregate = (f"{sum(means[d] for d in representatives) / len(representatives) * 100:.3f}"
                     if all(d in means for d in representatives) else "pending")
        lines.append("| " + name + " | " + " | ".join(cells) + " | " + aggregate + " |")
    (study / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def record_trial(study: Path, record: dict) -> None:
    """Merge a completed trial when two GPU workers finish concurrently."""
    with (study / ".ledger.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = study / "trials.json"
        rows = json.loads(path.read_text(encoding="utf-8"))["runs"] if path.exists() else []
        key = tuple(record[name] for name in ("candidate", "dataset", "seed", "phase"))
        rows = [r for r in rows if tuple(r[name] for name in ("candidate", "dataset", "seed", "phase")) != key]
        write_ledger(study, [*rows, record])
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def collect_record(output_root: Path, candidate: str, dataset: str, seed: int, phase: str) -> dict:
    statuses = list((output_root / candidate / dataset / "nc" / "mrage").glob(f"*/seed-{seed}/status.json"))
    if not statuses:
        raise FileNotFoundError("runner did not produce status.json")
    if len(statuses) != 1:
        raise RuntimeError(f"ambiguous configs for {candidate}/{dataset}/seed{seed}")
    run_dir = statuses[0].parent
    status = json.loads(statuses[0].read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metric_path = run_dir / ("validation_metrics.json" if phase == "validation" else "final_test_metrics.json")
    payload = json.loads(metric_path.read_text(encoding="utf-8")) if metric_path.exists() else {}
    return {
        "status": status["status"], "run_dir": str(run_dir),
        "device": metadata.get("device"),
        "best_epoch": metadata.get("best_epoch"), "config_hash": metadata.get("config_hash"),
        "split_id": metadata.get("split_id"), "model_parameters": metadata.get("model_parameters"),
        **payload.get("metrics", {}), "error": status.get("error", ""),
    }


def run_job(study: Path, output_root: Path, phase: str, device: str, job: tuple) -> int:
    candidate, dataset, seed, overlay = job
    runtime_device = device
    environment = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
    if phase == "final-test":
        saved_device = collect_record(output_root, candidate, dataset, seed, phase).get("device") or device
        if saved_device != device:
            # Resume reads the saved logical device. Map old cuda:0 checkpoints
            # to the requested physical GPU without editing their saved config.
            if saved_device != "cuda:0" or not device.startswith("cuda:"):
                raise RuntimeError(f"unsupported device remap: {saved_device} -> {device}")
            environment["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
            runtime_device = "cuda:0"
    check_idle_gpu(device)
    command = [
        sys.executable, "scripts/run_paper_benchmark.py", "--dataset", dataset,
        "--task", "nc", "--models", "mrage", "--seeds", str(seed),
        "--device", runtime_device, "--stage", phase, "--resume",
        "--experiment-config", str(overlay), "--output-root", str(output_root / candidate),
    ]
    start = time.monotonic()
    print(f"START {phase} {candidate} {dataset} seed={seed} device={device}", flush=True)
    log_dir = output_root / "launchers" / candidate / dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"seed-{seed}-{phase}.log"
    record = {"candidate": candidate, "dataset": dataset, "seed": seed, "phase": phase,
              "device": device, "physical_device": device,
              "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES")}
    failure = 0
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"command": command, "physical_device": device,
                                  "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
                                  "at_unix": time.time()}) + "\n")
            log.flush()
            completed = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        record.update(collect_record(output_root, candidate, dataset, seed, phase))
        if completed.returncode:
            raise RuntimeError(f"runner exit={completed.returncode}; see {log_path}")
    except Exception as exc:
        failure = 1
        record.update(status="failed", error=str(exc))
    record["wall_seconds"] = time.monotonic() - start
    record_trial(study, record)
    print(f"DONE {record}", flush=True)
    return failure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=ROOT / "experiments/nc5_tuning_20261030")
    parser.add_argument("--candidates", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--phase", choices=("validation", "final-test"), default="validation")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    study = args.study_dir.resolve()
    manifest = OmegaConf.load(study / "study.yaml")
    datasets = args.datasets or list(manifest.get("representatives", manifest.datasets))
    if set(datasets) - set(manifest.datasets):
        parser.error("this study only permits the declared five NC datasets")
    candidates = args.candidates or list(manifest.candidates)
    if set(candidates) - set(manifest.candidates):
        parser.error("unknown candidate")
    if args.phase == "final-test" and not (study / "selection.json").is_file():
        parser.error("record selection.json before final Test")
    snapshot_base(study)
    output_root = ROOT / "outputs" / str(manifest.study_id)
    old = study / "trials.json"
    rows = json.loads(old.read_text(encoding="utf-8"))["runs"] if old.exists() else []
    devices = args.devices or [args.device]
    jobs = Queue()
    for candidate in candidates:
        overlay = candidate_file(study, manifest, candidate)
        for dataset in datasets:
            for seed in args.seeds:
                key = (candidate, dataset, seed, args.phase)
                previous = next((r for r in rows if (r["candidate"], r["dataset"], r["seed"], r["phase"]) == key), None)
                if previous and previous["status"] in {"validation_complete", "final_test_complete"}:
                    print(f"SKIP {key}", flush=True)
                    continue
                if args.dry_run:
                    print(f"{args.phase}: {candidate} {dataset} seed={seed} overlay={overlay}", flush=True)
                    continue
                jobs.put((candidate, dataset, seed, overlay))
    if args.dry_run:
        return 0

    def worker(device: str) -> int:
        failures = 0
        while not (study / "STOP_AFTER_CURRENT").exists():
            try:
                job = jobs.get_nowait()
            except Empty:
                break
            failures += run_job(study, output_root, args.phase, device, job)
        return failures

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        failures = sum(pool.map(worker, devices))
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
