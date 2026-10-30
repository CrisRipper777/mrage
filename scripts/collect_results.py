#!/usr/bin/env python3
"""Collect structured per-run metrics and completed Test aggregates."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


RUN_COLUMNS = [
    "dataset", "task", "model", "seed", "split_id", "protocol_version",
    "config_hash", "best_epoch", "metric_name", "metric_value",
]
SUMMARY_COLUMNS = [
    "dataset", "task", "model", "split_id", "protocol_version", "config_hash",
    "metric_name", "mean", "std", "num_valid_runs", "expected_seeds",
    "missing_seeds",
]


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    run_rows: list[dict] = []
    status_rows: list[dict] = []
    grouped: dict[tuple, list[dict]] = {}
    expected_by_group: dict[tuple, set[int]] = {}

    for status_path in sorted(root.rglob("status.json")):
        run_dir = status_path.parent
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            status_rows.append({"run_dir": str(run_dir), "status": "invalid_status", "error": str(exc)})
            continue
        metadata_path = run_dir / "run_metadata.json"
        metadata = {}
        try:
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            status_rows.append({"run_dir": str(run_dir), "status": "invalid_metadata", "error": str(exc)})
            continue

        record = {
            "run_dir": str(run_dir),
            "dataset": metadata.get("dataset"),
            "task": metadata.get("task"),
            "model": metadata.get("model"),
            "seed": metadata.get("training_seed", status.get("training_seed")),
            "status": status.get("status", "unknown"),
            "test_evaluation_count": status.get("test_evaluation_count", 0),
            "split_id": metadata.get("split_id"),
            "protocol_version": metadata.get("protocol_version"),
            "config_hash": metadata.get("config_hash"),
            "best_epoch": metadata.get("best_epoch"),
            "error": status.get("error"),
            "expected_seeds": metadata.get("expected_seeds", []),
        }
        status_rows.append(record)
        group_base = (
            record.get("dataset"), record.get("task"), record.get("model"),
            record.get("split_id"), record.get("protocol_version"),
            record.get("config_hash"),
        )
        expected_by_group.setdefault(group_base, set()).update(
            int(seed) for seed in record.get("expected_seeds", [])
        )

        # Only an explicitly completed run with exactly one persisted Test
        # evaluation can contribute to the paper's Test result table.
        if status.get("status") != "final_test_complete" or int(status.get("test_evaluation_count", 0)) != 1:
            continue
        final_path = run_dir / "final_test_metrics.json"
        if not final_path.is_file():
            continue
        try:
            final = json.loads(final_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if int(final.get("test_evaluation_count", 0)) != 1:
            continue
        metrics = final.get("metrics", {})
        run_fields = {
            "dataset": final.get("dataset", metadata.get("dataset")),
            "task": final.get("task", metadata.get("task")),
            "model": final.get("model", metadata.get("model")),
            "seed": final.get("seed", metadata.get("training_seed")),
            "split_id": final.get("split_id", metadata.get("split_id")),
            "protocol_version": final.get("protocol_version", metadata.get("protocol_version")),
            "config_hash": metadata.get("config_hash"),
            "best_epoch": final.get("best_epoch", metadata.get("best_epoch")),
        }
        for metric_name, metric_value in metrics.items():
            row = {**run_fields, "metric_name": metric_name, "metric_value": float(metric_value)}
            run_rows.append(row)
            if metric_name.startswith("test_"):
                key = (
                    row["dataset"], row["task"], row["model"], row["split_id"],
                    row["protocol_version"], row["config_hash"], metric_name,
                )
                grouped.setdefault(key, []).append(
                    {**row, "expected_seeds": metadata.get("expected_seeds", [])}
                )

    summary_rows = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        values = [row["metric_value"] for row in rows]
        expected = sorted(expected_by_group.get(key[:-1], set()))
        observed = {int(row["seed"]) for row in rows}
        missing = [seed for seed in expected if seed not in observed]
        summary_rows.append(
            {
                "dataset": key[0], "task": key[1], "model": key[2],
                "split_id": key[3], "protocol_version": key[4], "config_hash": key[5],
                "metric_name": key[6], "mean": statistics.fmean(values),
                "std": statistics.pstdev(values), "num_valid_runs": len(values),
                "expected_seeds": ";".join(map(str, expected)),
                "missing_seeds": ";".join(map(str, missing)),
            }
        )
    return run_rows, summary_rows, status_rows


def _write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "| Dataset | Task | Model | Split | Protocol | Config | Metric | Mean | Std | N | Missing seeds |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {task} | {model} | {split_id} | {protocol_version} | "
            "{config_hash} | {metric_name} | {mean:.6g} | {std:.6g} | {num_valid_runs} | {missing_seeds} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output_dir = (args.output_dir or root).expanduser().resolve()
    runs, summaries, statuses = collect(root)
    _write_csv(output_dir / "run_metrics.csv", RUN_COLUMNS, runs)
    _write_csv(output_dir / "summary.csv", SUMMARY_COLUMNS, summaries)
    _write_csv(
        output_dir / "run_status.csv",
        [
            "run_dir", "dataset", "task", "model", "seed", "status",
            "test_evaluation_count", "split_id", "protocol_version", "config_hash",
            "best_epoch", "error", "expected_seeds",
        ],
        statuses,
    )
    _write_markdown(output_dir / "summary.md", summaries)
    print(f"Collected {len(runs)} metric rows from {len(statuses)} runs; {len(summaries)} Test aggregates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
