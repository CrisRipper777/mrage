#!/usr/bin/env python3
"""Promote the corrected NC summary with selected DMGC runs to canonical files."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "baseline_benchmark_full_method_nc_20261031"
SELECTED_ROOT = OUTPUT_ROOT
ARCHIVE_ROOT = OUTPUT_ROOT / "_archives" / "original_nc_summary_before_dmgc_lambda_cm03_20261031"

CANONICAL_FILES = {
    "nc_baseline_summary.md": SELECTED_ROOT / "nc_baseline_summary_selected_dmgc_lambda_cm03.md",
    "nc_baseline_summary.csv": SELECTED_ROOT / "nc_baseline_summary_selected_dmgc_lambda_cm03.csv",
    "nc_baseline_per_seed_metrics.csv": SELECTED_ROOT / "nc_baseline_per_seed_selected_dmgc_lambda_cm03.csv",
}
SUMMARY_FIELDS = [
    "dataset", "model", "seeds", "test_acc_mean_pct", "test_acc_std_pct",
    "test_macro_f1_mean_pct", "test_macro_f1_std_pct",
]
PER_SEED_FIELDS = [
    "dataset", "model", "seed", "test_acc_pct", "test_macro_f1_pct", "best_epoch",
    "run_dir", "checkpoint", "train_log", "final_test_metrics",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    originals = [
        "nc_baseline_summary.md",
        "nc_baseline_summary.csv",
        "nc_baseline_per_seed_metrics.csv",
    ]
    archived = {}
    for name in originals:
        source = OUTPUT_ROOT / name
        destination = ARCHIVE_ROOT / name
        if not destination.exists():
            shutil.copy2(source, destination)
        archived[name] = {"path": str(destination), "sha256": sha256(destination)}

    selected_summary = CANONICAL_FILES["nc_baseline_summary.csv"]
    with selected_summary.open("r", encoding="utf-8-sig", newline="") as stream:
        summary_rows = list(csv.DictReader(stream))
    if len(summary_rows) != 50:
        raise RuntimeError(f"expected 50 aggregate rows, found {len(summary_rows)}")
    summary_path = OUTPUT_ROOT / "nc_baseline_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in SUMMARY_FIELDS} for row in summary_rows)

    selected_per_seed = CANONICAL_FILES["nc_baseline_per_seed_metrics.csv"]
    with selected_per_seed.open("r", encoding="utf-8-sig", newline="") as stream:
        per_seed_rows = list(csv.DictReader(stream))
    if len(per_seed_rows) != 150:
        raise RuntimeError(f"expected 150 per-seed rows, found {len(per_seed_rows)}")
    per_seed_path = OUTPUT_ROOT / "nc_baseline_per_seed_metrics.csv"
    with per_seed_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PER_SEED_FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in PER_SEED_FIELDS} for row in per_seed_rows)

    summary_md_source = CANONICAL_FILES["nc_baseline_summary.md"]
    summary_md_target = OUTPUT_ROOT / "nc_baseline_summary.md"
    shutil.copy2(summary_md_source, summary_md_target)

    manifest = {
        "summary_version": "selected_dmgc_lambda_cm_0.3_observed_classes_v1",
        "updated_date": date.today().isoformat(),
        "canonical_summary_markdown": str(summary_md_target),
        "canonical_summary_csv": str(summary_path),
        "canonical_per_seed_csv": str(per_seed_path),
        "sources": {
            "non_dmgc_runs": str(OUTPUT_ROOT),
            "selected_dmgc_runs": str(ROOT / "outputs" / "dmgc_nc_lambda_cm_screen_20261031" / "lambda_cm_0.3"),
            "selected_dmgc_lambda_cm": 0.3,
            "dmgc_selection": "selected by validation Accuracy after a validation-only five-dataset screen",
            "ele_fashion_macro_f1": "11 observed classes; class 5 has zero labeled nodes",
            "checkpoint_selection": "best validation Accuracy",
            "test_evaluation": "one final Test evaluation per selected run",
        },
        "archived_previous_canonical_files": archived,
    }
    manifest_path = OUTPUT_ROOT / "nc_baseline_summary_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Promoted summary: {summary_md_target}")
    print(f"Promoted CSV: {summary_path}")
    print(f"Promoted per-seed metrics: {per_seed_path}")
    print(f"Archived previous files: {ARCHIVE_ROOT}")
    print(f"Provenance: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
