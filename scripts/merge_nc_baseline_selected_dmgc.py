#!/usr/bin/env python3
"""Build the updated NC baseline report with validation-selected DMGC lambda_cm."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "outputs" / "baseline_benchmark_full_method_nc_20261031"
SELECTED_DMGC_ROOT = PROJECT_ROOT / "outputs" / "dmgc_nc_lambda_cm_screen_20261031" / "lambda_cm_0.3"
CLASS_COUNTS = {"Movies": 20, "Grocery": 20, "Toys": 18, "Reddit-S": 20, "ele-fashion": 11}
DATASETS = ("Movies", "Grocery", "Toys", "Reddit-S", "ele-fashion")


def main() -> int:
    source_path = BASELINE_ROOT / "nc_baseline_per_seed_metrics_observed_classes.csv"
    with source_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    for row in rows:
        if row["model"] != "dmgc":
            continue
        dataset, seed = row["dataset"], int(row["seed"])
        result_path = next(
            (SELECTED_DMGC_ROOT / dataset / "nc" / "dmgc").glob(
                f"*/seed-{seed}/final_test_metrics.json"
            )
        )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        run_dir = result_path.parent
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        if payload.get("test_evaluation_count") != 1 or status.get("test_evaluation_count") != 1:
            raise RuntimeError(f"selected DMGC Test count is not exactly one: {run_dir}")
        if status.get("status") != "final_test_complete":
            raise RuntimeError(f"selected DMGC run is incomplete: {run_dir}")
        metrics = payload["metrics"]
        row.update(
            test_acc_pct=f"{metrics['test_acc'] * 100:.12f}",
            test_macro_f1_pct=f"{metrics['test_macro_f1'] * 100:.12f}",
            macro_f1_classes=str(CLASS_COUNTS[dataset]),
            best_epoch=str(payload["best_epoch"]),
            run_dir=str(run_dir),
            checkpoint=str(run_dir / "best_validation.pt"),
            train_log=str(run_dir / "train.log"),
            final_test_metrics=str(result_path),
            effective_learning_rate=str(metadata["effective_learning_rate"]),
            effective_weight_decay=str(metadata["effective_weight_decay"]),
            lambda_cm="0.3",
        )

    output_per_seed = BASELINE_ROOT / "nc_baseline_per_seed_selected_dmgc_lambda_cm03.csv"
    per_seed_fields = list(rows[0]) + ["effective_learning_rate", "effective_weight_decay", "lambda_cm"]
    with output_per_seed.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_seed_fields)
        writer.writeheader()
        writer.writerows(rows)

    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["model"]), []).append(row)
    summary = []
    for dataset in DATASETS:
        for model in dict.fromkeys(row["model"] for row in rows):
            values = groups.get((dataset, model), [])
            if not values:
                continue
            acc = [float(row["test_acc_pct"]) for row in values]
            f1 = [float(row["test_macro_f1_pct"]) for row in values]
            summary.append({
                "dataset": dataset,
                "model": model,
                "seeds": ",".join(row["seed"] for row in values),
                "macro_f1_classes": CLASS_COUNTS[dataset],
                "test_acc_mean_pct": statistics.fmean(acc),
                "test_acc_std_pct": statistics.pstdev(acc),
                "test_macro_f1_mean_pct": statistics.fmean(f1),
                "test_macro_f1_std_pct": statistics.pstdev(f1),
                "dmgc_lambda_cm": "0.3 (selected on Validation Accuracy)" if model == "dmgc" else "",
            })

    summary_path = BASELINE_ROOT / "nc_baseline_summary_selected_dmgc_lambda_cm03.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# NC baseline results with selected DMGC setting",
        "",
        "- Ten non-MRAGE baselines; datasets Movies, Grocery, Toys, Reddit-S, ele-fashion; seeds 42–44.",
        "- Accuracy remains the primary metric and the checkpoint selection metric is best Validation Accuracy.",
        "- Values are mean ± population standard deviation (`ddof=0`), in percent.",
        "- ele-fashion Macro-F1 uses its 11 observed classes; class 5 has zero labeled nodes.",
        "- DMGC uses lambda_cm=0.3, selected using validation Accuracy only; its final Test was then evaluated once. Other baseline rows retain their original completed runs.",
        "",
    ]
    for dataset in DATASETS:
        lines += [f"## {dataset}", "", "| Model | Test Accuracy (%) | Test Macro-F1 (%) |", "|---|---:|---:|"]
        for row in summary:
            if row["dataset"] == dataset:
                lines.append(
                    f"| {row['model']} | {row['test_acc_mean_pct']:.2f} ± {row['test_acc_std_pct']:.2f} "
                    f"| {row['test_macro_f1_mean_pct']:.2f} ± {row['test_macro_f1_std_pct']:.2f} |"
                )
        lines.append("")
    summary_md = BASELINE_ROOT / "nc_baseline_summary_selected_dmgc_lambda_cm03.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary: {summary_md}")
    print(f"Summary CSV: {summary_path}")
    print(f"Per-seed CSV: {output_per_seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
