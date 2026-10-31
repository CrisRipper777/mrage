#!/usr/bin/env python3
"""Recalculate the saved NC summary over dataset classes that actually exist.

For ele-fashion, class 5 is absent from every label. Its prior 12-class score
therefore equals the 11-class score multiplied by 11/12; this script reverses
that factor without running another Test inference. Other dataset metrics are
copied unchanged.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


CLASS_COUNTS = {
    "Movies": 20,
    "Grocery": 20,
    "Toys": 18,
    "Reddit-S": 20,
    "ele-fashion": 11,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/baseline_benchmark_full_method_nc_20261031/nc_baseline_per_seed_metrics.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()
    output_root = (args.output_root or input_path.parent).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))

    rows = []
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for source in source_rows:
        row = dict(source)
        dataset = row["dataset"]
        if dataset == "ele-fashion":
            row["test_macro_f1_pct"] = str(float(row["test_macro_f1_pct"]) * 12.0 / 11.0)
        row["macro_f1_classes"] = str(CLASS_COUNTS[dataset])
        rows.append(row)
        groups[(dataset, row["model"])].append(row)

    per_seed_path = output_root / "nc_baseline_per_seed_metrics_observed_classes.csv"
    fields = ["dataset", "model", "seed", "test_acc_pct", "test_macro_f1_pct", "macro_f1_classes",
              "best_epoch", "run_dir", "checkpoint", "train_log", "final_test_metrics"]
    with per_seed_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for (dataset, model), values in groups.items():
        accuracy = [float(row["test_acc_pct"]) for row in values]
        macro_f1 = [float(row["test_macro_f1_pct"]) for row in values]
        summary.append({
            "dataset": dataset,
            "model": model,
            "seeds": ",".join(row["seed"] for row in values),
            "macro_f1_classes": CLASS_COUNTS[dataset],
            "test_acc_mean_pct": statistics.fmean(accuracy),
            "test_acc_std_pct": statistics.pstdev(accuracy),
            "test_macro_f1_mean_pct": statistics.fmean(macro_f1),
            "test_macro_f1_std_pct": statistics.pstdev(macro_f1),
        })

    summary_path = output_root / "nc_baseline_summary_observed_classes.csv"
    summary_fields = list(summary[0])
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    markdown_path = output_root / "nc_baseline_summary_observed_classes.md"
    lines = [
        "# NC baseline results: Macro-F1 over observed dataset classes",
        "",
        "- Accuracy and checkpoint selection are unchanged: best Validation Accuracy.",
        "- Seeds 42–44; values are mean ± population standard deviation (`ddof=0`), in percent.",
        "- ele-fashion Macro-F1 excludes class 5, which has zero labeled nodes. Its saved 12-class Macro-F1 was rescaled by `12/11`; no new Test inference was run.",
        "",
    ]
    datasets = list(dict.fromkeys(row["dataset"] for row in summary))
    for dataset in datasets:
        lines += [f"## {dataset}", "", "| Model | Test Accuracy (%) | Test Macro-F1 (%) |", "|---|---:|---:|"]
        for row in summary:
            if row["dataset"] == dataset:
                lines.append(
                    f"| {row['model']} | {row['test_acc_mean_pct']:.2f} ± {row['test_acc_std_pct']:.2f} "
                    f"| {row['test_macro_f1_mean_pct']:.2f} ± {row['test_macro_f1_std_pct']:.2f} |"
                )
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Per-seed corrected metrics: {per_seed_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Summary Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
