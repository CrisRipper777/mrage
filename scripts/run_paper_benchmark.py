#!/usr/bin/env python3
"""Run isolated paper experiments with validation-selected, single-use Test phases."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.run_artifacts import write_json_atomic


def _safe_component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"invalid path component: {value!r}")
    return value


def _compose_config(args, seed: int):
    base_overrides = [
        f"dataset={args.dataset}", f"task={args.task}", f"model={args.model}",
        f"seed={seed}", "num_runs=1", f"device={args.device}",
    ]
    with initialize_config_dir(
        version_base=None, config_dir=str(PROJECT_ROOT / "configs")
    ):
        cfg = compose(config_name="config", overrides=base_overrides)
    if args.experiment_config:
        experiment = OmegaConf.load(Path(args.experiment_config).expanduser())
        if "defaults" in experiment:
            experiment.pop("defaults")
        cfg = OmegaConf.merge(cfg, experiment)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))

    cfg.dataset.name = args.dataset
    cfg.task.name = args.task
    cfg.model.name = args.model
    cfg.seed = int(seed)
    cfg.num_runs = 1
    cfg.device = args.device
    if args.task == "nc":
        cfg.dataset_split_seed = int(args.split_seed)
    return cfg


def _config_hash(cfg) -> str:
    payload = OmegaConf.to_container(cfg, resolve=True)
    payload = copy.deepcopy(payload)
    payload.pop("seed", None)
    payload.pop("num_runs", None)
    payload.pop("device", None)
    payload.get("paths", {}).pop("output_root", None)
    payload.get("paths", {}).pop("semantic_cache_root", None)
    dataset = payload.get("dataset", {})
    dataset.pop("nc_split_path", None)
    dataset.pop("lp_split_path", None)
    task = payload.get("task", {})
    for key in (
        "phase", "evaluate_test", "mask_test_labels", "save_ckpt_path",
        "run_metrics_path", "checkpoint_path", "validation_metrics_path",
        "final_test_metrics_path", "paper_run_status_path",
    ):
        task.pop(key, None)
    semantic = payload.get("semantic_graph", {})
    for key in (
        "cache_dir", "resolved_cache_path", "identity_fingerprint",
        "feature_fingerprint", "candidate_fingerprint", "v7b_fingerprint",
    ):
        semantic.pop(key, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _source_tree_fingerprint() -> str:
    digest = hashlib.sha256()
    for folder in (PROJECT_ROOT / "src", PROJECT_ROOT / "configs", PROJECT_ROOT / "scripts"):
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml"}:
                continue
            relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _run_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process holds run lock {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _persist_config(cfg, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config_resolved.yaml"
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return path


def _run_main(config_path: Path, run_dir: Path, phase: str, log_path: Path) -> int:
    hydra_dir = run_dir / "hydra" / phase
    hydra_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "src/main.py",
        "--config-path", str(config_path.parent),
        "--config-name", config_path.stem,
        f"task.phase={phase}",
        f"hydra.run.dir={json.dumps(str(hydra_dir))}",
    ]
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
            check=False,
        )
    return int(result.returncode)


def _metric_payload(checkpoint: dict, args, seed: int, split_id: str) -> dict:
    metrics = {
        key: float(value) for key, value in checkpoint.get("metrics", {}).items()
        if key.startswith("val_")
    }
    if not metrics:
        raise ValueError("best checkpoint does not contain Validation metrics")
    return {
        "dataset": args.dataset,
        "task": args.task,
        "model": args.model,
        "seed": int(seed),
        "split_id": split_id,
        "protocol_version": str(checkpoint["protocol_version"]),
        "best_epoch": int(checkpoint["epoch"]),
        "selection_metric": checkpoint["selection"],
        "metrics": metrics,
        "test_evaluation_count": 0,
    }


def _load_completed_test_metrics(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"run is marked complete but Final Test results are missing: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"completed Final Test results cannot be read: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"completed Final Test results must be a JSON object: {path}")
    return payload


def _prepare_run(args, seed: int, run_dir: Path, config_hash: str, source_commit: str):
    cfg = _compose_config(args, seed)
    cfg.task.phase = "train"
    cfg.task.evaluate_test = False
    if args.task == "nc":
        cfg.task.mask_test_labels = True
        cfg.task.run_metrics_path = str(run_dir / "training_metrics.json")
    cfg.task.save_ckpt_path = str(run_dir / "best_validation.pt")
    cfg.task.checkpoint_path = str(run_dir / "best_validation.pt")
    cfg.task.validation_metrics_path = str(run_dir / "validation_metrics.json")
    cfg.task.final_test_metrics_path = str(run_dir / "final_test_metrics.json")
    cfg.task.paper_run_status_path = str(run_dir / "status.json")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = _persist_config(cfg, run_dir)
    metadata = {
        "dataset": args.dataset,
        "task": args.task,
        "model": args.model,
        "training_seed": int(seed),
        "expected_seeds": [int(value) for value in args.seeds],
        "split_seed": (
            int(args.split_seed)
            if args.task == "nc" and str(cfg.dataset.get("source", "")) == "magb"
            else None
        ),
        "protocol_version": str(cfg.task.protocol_version),
        "config_hash": config_hash,
        "source_commit": source_commit,
        "source_tree_sha256": _source_tree_fingerprint(),
        "dataset_config": OmegaConf.to_container(cfg.dataset, resolve=True),
        "model_config": OmegaConf.to_container(cfg.model, resolve=True),
        "device": str(cfg.device),
        "created_at_unix": time.time(),
    }
    write_json_atomic(run_dir / "run_metadata.json", metadata)
    return cfg, config_path, metadata


def _finish_training(args, seed: int, run_dir: Path, cfg, metadata: dict) -> None:
    checkpoint_path = run_dir / "best_validation.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError("training exited successfully without best_validation.pt")
    checkpoint = __import__("torch").load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    run_metadata = checkpoint.get("run_metadata", {})
    split_id = run_metadata.get("split_sha256")
    if not split_id:
        raise ValueError("checkpoint metadata does not contain a split fingerprint")
    validation = _metric_payload(checkpoint, args, seed, str(split_id))
    write_json_atomic(run_dir / "validation_metrics.json", validation)
    optimizer = run_metadata.get("effective_optimizer", {})
    groups = optimizer.get("parameter_groups", [])
    metadata.update(
        split_id=str(split_id),
        split_file=run_metadata.get("split_path"),
        best_epoch=int(checkpoint["epoch"]),
        selection_metric=checkpoint["selection"],
        effective_optimizer=optimizer,
        effective_learning_rate=groups[0]["learning_rate"] if groups else None,
        effective_weight_decay=groups[0]["weight_decay"] if groups else None,
        model_parameters=run_metadata.get("model_parameters"),
        head_parameters=run_metadata.get("head_parameters", run_metadata.get("predictor_parameters")),
        projection_parameters=run_metadata.get("projection_parameters", 0),
        checkpoint_sha256=_sha256(checkpoint_path),
        checkpoint_run_metadata=run_metadata,
        test_evaluation_count=0,
    )
    write_json_atomic(run_dir / "run_metadata.json", metadata)
    write_json_atomic(
        run_dir / "status.json",
        {
            "status": "validation_complete",
            "stage": "validation",
            "training_seed": int(seed),
            "test_evaluation_count": 0,
            "updated_at_unix": time.time(),
        },
    )


def _run_one(args, seed: int, model: str, output_root: Path, source_commit: str) -> str:
    args = copy.copy(args)
    args.model = model
    cfg = _compose_config(args, seed)
    config_hash = _config_hash(cfg)
    run_dir = output_root / args.dataset / args.task / model / config_hash / f"seed-{seed}"
    lock_path = run_dir.parent / f".seed-{seed}.lock"
    if args.dry_run:
        print(
            f"{args.stage}: dataset={args.dataset} task={args.task} model={model} "
            f"seed={seed} config={config_hash} output={run_dir}"
        )
        return "dry_run"

    with _run_lock(lock_path):
        status_path = run_dir / "status.json"
        if run_dir.exists():
            if not args.resume:
                raise FileExistsError(
                    f"run directory already exists; use --resume only for a saved validation checkpoint: {run_dir}"
                )
            if not status_path.is_file():
                raise RuntimeError(f"existing run has no status.json and cannot be resumed: {run_dir}")
            status = json.loads(status_path.read_text(encoding="utf-8"))
            final_path = run_dir / "final_test_metrics.json"
            if status.get("status") == "final_test_running" and final_path.is_file():
                final_payload = json.loads(final_path.read_text(encoding="utf-8"))
                if int(final_payload.get("test_evaluation_count", 0)) == 1:
                    status.update(status="final_test_complete", test_evaluation_count=1)
                    write_json_atomic(status_path, status)
            if status.get("status") == "final_test_complete":
                if status.get("test_evaluation_count") != 1:
                    raise RuntimeError("completed run has an invalid Test evaluation count")
                final_payload = _load_completed_test_metrics(final_path)
                metadata_path = run_dir / "run_metadata.json"
                if metadata_path.is_file():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata.get("test_evaluation_count") != 1:
                        metadata["test_evaluation_count"] = 1
                        metadata["final_test_metrics"] = final_payload.get("metrics", {})
                        write_json_atomic(metadata_path, metadata)
                print(f"SKIP complete: {run_dir}")
                return "complete"
            if status.get("status") != "validation_complete":
                raise RuntimeError(
                    "resume is allowed only from validation_complete; "
                    f"found status={status.get('status')!r}"
                )
            old_metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            if old_metadata.get("config_hash") != config_hash:
                raise RuntimeError("resolved config hash differs from the existing run")
            if args.stage == "validation":
                print(f"SKIP validation_complete: {run_dir}")
                return "validation_complete"
            config_path = run_dir / "config_resolved.yaml"
            metadata = old_metadata
            cfg = OmegaConf.load(config_path)
            if _config_hash(cfg) != config_hash:
                raise RuntimeError("saved resolved config does not match its recorded config hash")
        else:
            if args.stage == "final-test":
                raise FileNotFoundError(f"final-test requires an existing validation run: {run_dir}")
            cfg, config_path, metadata = _prepare_run(
                args, seed, run_dir, config_hash, source_commit
            )
            write_json_atomic(
                status_path,
                {
                    "status": "training",
                    "stage": "validation",
                    "training_seed": int(seed),
                    "test_evaluation_count": 0,
                    "started_at_unix": time.time(),
                },
            )
            try:
                code = _run_main(config_path, run_dir, "train", run_dir / "train.log")
                if code != 0:
                    raise RuntimeError(f"training process exited with code {code}")
                _finish_training(args, seed, run_dir, cfg, metadata)
            except Exception as exc:
                write_json_atomic(
                    status_path,
                    {
                        "status": "failed",
                        "stage": "validation",
                        "training_seed": int(seed),
                        "test_evaluation_count": 0,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "updated_at_unix": time.time(),
                    },
                )
                raise
            if args.stage == "validation":
                return "validation_complete"

        if args.stage == "validation":
            return "validation_complete"
        metadata_path = run_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint_path = run_dir / "best_validation.pt"
        expected_checkpoint_hash = metadata.get("checkpoint_sha256")
        if not expected_checkpoint_hash or _sha256(checkpoint_path) != expected_checkpoint_hash:
            raise RuntimeError("best Validation checkpoint SHA256 changed after training")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("test_evaluation_count") != 0:
            raise RuntimeError("Test evaluation count is not zero before final phase")
        status.update(status="final_test_running", stage="final-test", updated_at_unix=time.time())
        write_json_atomic(status_path, status)
        try:
            code = _run_main(config_path, run_dir, "final_test", run_dir / "train.log")
            if code != 0:
                raise RuntimeError(f"final-test process exited with code {code}")
            final_path = run_dir / "final_test_metrics.json"
            final_payload = json.loads(final_path.read_text(encoding="utf-8"))
            if int(final_payload.get("test_evaluation_count", 0)) != 1:
                raise RuntimeError("final-test output did not record exactly one Test evaluation")
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") != "final_test_complete" or status.get("test_evaluation_count") != 1:
                raise RuntimeError("final-test evaluator did not persist its completion status")
            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            metadata["test_evaluation_count"] = 1
            metadata["final_test_metrics"] = final_payload["metrics"]
            write_json_atomic(run_dir / "run_metadata.json", metadata)
            return "final_test_complete"
        except Exception as exc:
            # If the evaluator wrote its atomic result before a later interruption,
            # preserve the single-use marker and never launch Test again.
            final_path = run_dir / "final_test_metrics.json"
            if final_path.is_file():
                payload = json.loads(final_path.read_text(encoding="utf-8"))
                status_value = "final_test_complete" if payload.get("test_evaluation_count") == 1 else "final_test_outcome_unknown"
                count = int(payload.get("test_evaluation_count", 0))
            else:
                status_value, count = "final_test_outcome_unknown", 0
            write_json_atomic(
                status_path,
                {
                    "status": status_value,
                    "stage": "final-test",
                    "training_seed": int(seed),
                    "test_evaluation_count": count,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "updated_at_unix": time.time(),
                },
            )
            raise


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", choices=("nc", "lp"), required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experiment-config", type=Path, default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--stage", choices=("validation", "final-test", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.dataset = _safe_component(args.dataset)
    args.models = [_safe_component(value) for value in args.models]
    if args.seeds is None:
        if args.task == "nc":
            args.seeds = [42, 43, 44]
        else:
            parser.error("--seeds is required for LP runs")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if args.experiment_config and not args.experiment_config.is_file():
        parser.error(f"experiment config not found: {args.experiment_config}")
    return args


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    source_commit = _source_commit()
    failures = []
    for model in args.models:
        for seed in args.seeds:
            try:
                outcome = _run_one(args, seed, model, output_root, source_commit)
                print(f"{outcome}: model={model} seed={seed}", flush=True)
            except Exception as exc:
                failures.append((model, seed, str(exc)))
                print(f"FAILED model={model} seed={seed}: {exc}", file=sys.stderr, flush=True)
                if args.stage != "all":
                    continue
    if failures:
        for model, seed, message in failures:
            print(f"ERROR {model} seed={seed}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
