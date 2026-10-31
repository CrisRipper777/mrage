#!/usr/bin/env python3
"""Merge completed GraphMoRE/NTSFormer NC runs into the canonical report."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "baseline_benchmark"
SUMMARY_ROOT = PROJECT_ROOT / "outputs" / "baseline_benchmark_full_method_nc_20261031"
MANIFEST_PATH = RUN_ROOT / "_manifests" / "nc_additional_baseline_benchmark_v4.json"
MRAGE_ADOPTED_RESULTS = PROJECT_ROOT / "experiments" / "nc5_test_exploration_20261031" / "adopted_results_percent.csv"
DATASETS = ("Movies", "Grocery", "Toys", "Reddit-S", "ele-fashion")
NEW_MODELS = ("graphmore", "ntsformer")
MRAGE_MODEL = "mrage"
SEEDS = (42, 43, 44)
PROTOCOL_VERSION = "unified_full_graph_nc_method_objectives_v4"
CLASS_COUNTS = {"Movies": 20, "Grocery": 20, "Toys": 18, "Reddit-S": 20, "ele-fashion": 11}
SUMMARY_FIELDS = [
    "dataset", "model", "seeds", "test_acc_mean_pct", "test_acc_std_pct",
    "test_macro_f1_mean_pct", "test_macro_f1_std_pct",
]
PER_SEED_FIELDS = [
    "dataset", "model", "seed", "test_acc_pct", "test_macro_f1_pct", "best_epoch",
    "run_dir", "checkpoint", "train_log", "final_test_metrics",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_run(dataset: str, model: str, seed: int) -> tuple[Path, dict, dict]:
    candidates = []
    for result_path in (RUN_ROOT / dataset / "nc" / model).glob(f"*/seed-{seed}/final_test_metrics.json"):
        run_dir = result_path.parent
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("protocol_version") == PROTOCOL_VERSION:
            candidates.append((run_dir, metadata, json.loads(result_path.read_text(encoding="utf-8"))))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one {PROTOCOL_VERSION} run for {dataset}/{model}/seed-{seed}, found {len(candidates)}")
    return candidates[0]


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"additional run manifest is not complete: {MANIFEST_PATH}")
    if tuple(manifest.get("datasets", ())) != DATASETS or tuple(manifest.get("models", ())) != NEW_MODELS:
        raise RuntimeError("additional run manifest does not match the expected dataset/model matrix")
    if tuple(manifest.get("seeds", ())) != SEEDS or manifest.get("device") != "cuda:0":
        raise RuntimeError("additional runs must use seeds 42–44 on cuda:0")

    additional_rows = []
    source_tree_hashes = set()
    split_hashes: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    for dataset in DATASETS:
        for model in NEW_MODELS:
            for seed in SEEDS:
                run_dir, metadata, result = _find_run(dataset, model, seed)
                status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                if status.get("status") != "final_test_complete" or status.get("test_evaluation_count") != 1:
                    raise RuntimeError(f"run did not complete exactly one final Test: {run_dir}")
                if metadata.get("device") != "cuda:0" or result.get("test_evaluation_count") != 1:
                    raise RuntimeError(f"device/Test count mismatch: {run_dir}")
                if result.get("protocol_version") != PROTOCOL_VERSION:
                    raise RuntimeError(f"run protocol mismatch: {run_dir}")
                for name in ("best_validation.pt", "train.log", "training_metrics.json", "validation_metrics.json"):
                    if not (run_dir / name).is_file():
                        raise RuntimeError(f"missing {name}: {run_dir}")
                source_tree_hashes.add(metadata.get("source_tree_sha256"))
                split_hashes[dataset].add(str(metadata.get("split_id")))
                metrics = result["metrics"]
                additional_rows.append({
                    "dataset": dataset,
                    "model": model,
                    "seed": str(seed),
                    "test_acc_pct": f"{metrics['test_acc'] * 100:.12f}",
                    "test_macro_f1_pct": f"{metrics['test_macro_f1'] * 100:.12f}",
                    "best_epoch": str(result["best_epoch"]),
                    "run_dir": str(run_dir),
                    "checkpoint": str(run_dir / "best_validation.pt"),
                    "train_log": str(run_dir / "train.log"),
                    "final_test_metrics": str(run_dir / "final_test_metrics.json"),
                })
    if len(additional_rows) != 30:
        raise RuntimeError(f"expected 30 additional seed results, found {len(additional_rows)}")
    if any(len(values) != 1 for values in split_hashes.values()):
        raise RuntimeError(f"seeds do not share the same fixed split per dataset: {split_hashes}")
    if len(source_tree_hashes) != 1 or None in source_tree_hashes:
        raise RuntimeError("the supplemental runs do not share one source tree fingerprint")

    with MRAGE_ADOPTED_RESULTS.open("r", encoding="utf-8-sig", newline="") as stream:
        adopted_rows = list(csv.DictReader(stream))
    adopted_by_dataset = {row["dataset"]: row for row in adopted_rows}
    if set(adopted_by_dataset) != set(DATASETS):
        raise RuntimeError(f"adopted MRAGE result datasets mismatch: {sorted(adopted_by_dataset)}")
    mrage_rows = []
    for dataset in DATASETS:
        row = adopted_by_dataset[dataset]
        if row.get("seeds") != "42,43,44" or int(row.get("n", 0)) != 3:
            raise RuntimeError(f"MRAGE adopted results are not three seeds 42–44: {dataset}")
        mrage_rows.append({
            "dataset": dataset,
            "model": MRAGE_MODEL,
            "seeds": row["seeds"],
            "test_acc_mean_pct": float(row["test_acc_mean_pct"]),
            "test_acc_std_pct": float(row["test_acc_std_pct_ddof0"]),
            "test_macro_f1_mean_pct": float(row["test_macro_f1_mean_pct"]),
            "test_macro_f1_std_pct": float(row["test_macro_f1_std_pct_ddof0"]),
            "configuration": row["configuration"],
            "source_type": row["source_type"],
        })

    per_seed_path = SUMMARY_ROOT / "nc_baseline_per_seed_metrics.csv"
    summary_path = SUMMARY_ROOT / "nc_baseline_summary.csv"
    markdown_path = SUMMARY_ROOT / "nc_baseline_summary.md"
    manifest_path = SUMMARY_ROOT / "nc_baseline_summary_manifest.json"
    archive_root = SUMMARY_ROOT / "_archives" / "summary_before_mrage_20261031"
    archive_root.mkdir(parents=True, exist_ok=True)
    archived = {}
    for path in (per_seed_path, summary_path, markdown_path, manifest_path):
        if path.exists():
            archived_path = archive_root / path.name
            if not archived_path.exists():
                shutil.copy2(path, archived_path)
            archived[path.name] = {"path": str(archived_path), "sha256": _sha256(archived_path)}

    with per_seed_path.open("r", encoding="utf-8-sig", newline="") as stream:
        prior_rows = list(csv.DictReader(stream))
    prior_rows = [row for row in prior_rows if row["model"] not in NEW_MODELS]
    if len(prior_rows) != 150:
        raise RuntimeError(f"expected 150 prior baseline seed rows, found {len(prior_rows)}")
    all_rows = prior_rows + additional_rows
    with per_seed_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PER_SEED_FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in PER_SEED_FIELDS} for row in all_rows)

    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in all_rows:
        groups.setdefault((row["dataset"], row["model"]), []).append(row)
    model_order = tuple(dict.fromkeys(row["model"] for row in prior_rows)) + NEW_MODELS + (MRAGE_MODEL,)
    summary_rows = []
    for dataset in DATASETS:
        for model in model_order:
            rows = groups.get((dataset, model), [])
            if not rows:
                continue
            acc = [float(row["test_acc_pct"]) for row in rows]
            f1 = [float(row["test_macro_f1_pct"]) for row in rows]
            summary_rows.append({
                "dataset": dataset,
                "model": model,
                "seeds": ",".join(row["seed"] for row in rows),
                "test_acc_mean_pct": statistics.fmean(acc),
                "test_acc_std_pct": statistics.pstdev(acc),
                "test_macro_f1_mean_pct": statistics.fmean(f1),
                "test_macro_f1_std_pct": statistics.pstdev(f1),
            })
    summary_rows.extend({key: row[key] for key in SUMMARY_FIELDS} for row in mrage_rows)
    model_order_index = {model: index for index, model in enumerate(model_order)}
    dataset_order_index = {dataset: index for index, dataset in enumerate(DATASETS)}
    summary_rows.sort(key=lambda row: (dataset_order_index[row["dataset"]], model_order_index[row["model"]]))
    if len(summary_rows) != 65:
        raise RuntimeError(f"expected 65 dataset/model aggregate rows, found {len(summary_rows)}")
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# NC results for MRAGE and 12 baselines",
        "",
        "- Results for MRAGE and twelve non-MRAGE baselines on Movies, Grocery, Toys, Reddit-S, and ele-fashion.",
        "- Accuracy is the primary metric; baseline checkpoint selection uses best Validation Accuracy.",
        "- Values are mean ± population standard deviation (`ddof=0`), in percent.",
        "- All seeds use the fixed split seed 42. ele-fashion Macro-F1 uses its 11 observed classes; class 5 has zero labeled nodes.",
        "- DMGC uses lambda_cm=0.3, selected using Validation Accuracy only; its Test was then evaluated once.",
        "- GraphMoRE and NTSFormer use the same unified task settings and shared linear NC classifier. Their run protocol is `unified_full_graph_nc_method_objectives_v4`.",
        "- GraphMoRE retains its signed-curvature experts, input skip, expert LayerNorm, topology gate, and distortion objective; its NetworkX ego expansion is adapted to sparse 2/3-hop pooling.",
        "- NTSFormer retains its student/dual-teacher transformer, full-graph SIGN contexts, teacher-to-student KL, and MoE load-balancing objective. It is evaluated as a transductive NC model without cold-start isolation.",
        "- MRAGE values follow `experiments/nc5_test_exploration_20261031/ADOPTED_RESULTS.md`. Movies uses a historical default run; the other four use configurations selected in a 65-run exploration that included Test results. These adopted MRAGE results are shown for reference and are not a same-batch controlled comparison against the baseline runs.",
        "- The per-seed CSV contains raw baseline runs. MRAGE is not expanded into synthetic per-seed records; its adopted aggregate and detailed source paths are recorded in `adopted_results_percent.csv` and `adopted_selection.json`.",
        "",
    ]
    for dataset in DATASETS:
        lines += [f"## {dataset}", "", "| Model | Test Accuracy (%) | Test Macro-F1 (%) |", "|---|---:|---:|"]
        for row in summary_rows:
            if row["dataset"] == dataset:
                lines.append(
                    f"| {'MRAGE' if row['model'] == MRAGE_MODEL else row['model']} | {row['test_acc_mean_pct']:.2f} ± {row['test_acc_std_pct']:.2f} "
                    f"| {row['test_macro_f1_mean_pct']:.2f} ± {row['test_macro_f1_std_pct']:.2f} |"
                )
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    summary_manifest = {
        "summary_version": "selected_dmgc_lambda_cm_0.3_observed_classes_plus_graphmore_ntsformer_plus_mrage_v3",
        "updated_date": date.today().isoformat(),
        "canonical_summary_markdown": str(markdown_path),
        "canonical_summary_csv": str(summary_path),
        "canonical_per_seed_csv": str(per_seed_path),
        "models": list(model_order),
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "additional_runs_manifest": str(MANIFEST_PATH),
        "additional_protocol_version": PROTOCOL_VERSION,
        "additional_source_tree_sha256": next(iter(source_tree_hashes)),
        "additional_runs_count": len(additional_rows),
        "summary_aggregate_rows": len(summary_rows),
        "baseline_per_seed_rows": len(all_rows),
        "mrage_adopted_results": str(MRAGE_ADOPTED_RESULTS),
        "mrage_adoption_notes": "Movies historical default run; remaining four selected from Test-guided 65-run exploration. Displayed for reference, not a same-batch controlled baseline comparison.",
        "mrage_configurations_by_dataset": {row["dataset"]: row["configuration"] for row in mrage_rows},
        "mrage_detailed_source_manifest": str(PROJECT_ROOT / "experiments" / "nc5_test_exploration_20261031" / "adopted_selection.json"),
        "ele_fashion_macro_f1_classes": 11,
        "dmgc_lambda_cm": 0.3,
        "dmgc_selection": "selected by Validation Accuracy after a validation-only five-dataset screen",
        "prior_baseline_per_seed_source": str(archive_root / "nc_baseline_per_seed_metrics.csv"),
        "prior_ten_baseline_archive": str(SUMMARY_ROOT / "_archives" / "summary_before_graphmore_ntsformer_20261031" / "nc_baseline_summary.csv"),
        "checkpoint_selection": "best Validation Accuracy",
        "test_evaluation": "one final Test evaluation per selected run",
        "archived_previous_canonical_files": archived,
    }
    manifest_path.write_text(json.dumps(summary_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Summary: {markdown_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Per-seed CSV: {per_seed_path}")
    print(f"Provenance manifest: {manifest_path}")
    print(f"Verified supplemental runs: {len(additional_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
