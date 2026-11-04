#!/usr/bin/env python3
"""Figure 3(b): modality-relation routing patterns of the shared expert pool.

Only frozen Full MRAGE checkpoints are read.  The six router calls are labelled
by actual forward order, not by modality/relation IDs alone.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mrage_figure3_common import (  # noqa: E402
    COLLABORATIVE_PHYSICAL,
    CONTEXT_ORDER,
    DATASETS,
    DEFAULT_CACHE_ROOT,
    ContextCapture,
    forward_full,
    jensen_shannon_divergence,
    load_model_bundle,
    primary_subset,
    resolve_checkpoint_records,
    routing_rows,
    set_device,
    validation_metrics,
    write_csv,
    write_json,
)


PATTERN_FIELDS = [
    "dataset",
    "training_seed",
    "checkpoint",
    "context",
    "subset",
    "expert_id",
    "mean_top2_weight",
    "top2_selection_frequency",
    "active_node_count",
    "active_node_ratio",
    "cross_nonzero_count",
    "cross_nonzero_ratio",
    "mean_node_route_std",
]
DIAGNOSTIC_FIELDS = [
    "dataset",
    "training_seed",
    "checkpoint",
    "context",
    "subset",
    "active_node_count",
    "active_node_ratio",
    "n_top2_combinations",
    "most_common_top2_combo",
    "most_common_top2_combo_label",
    "most_common_share",
    "mean_node_route_std",
    "cross_nonzero_count",
    "cross_nonzero_ratio",
] + [f"jsd_vs_{context}" for context in CONTEXT_ORDER]
PAIRWISE_FIELDS = [
    "dataset",
    "training_seed",
    "checkpoint",
    "context_a",
    "context_b",
    "subset_a",
    "subset_b",
    "comparison",
    "jsd",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--all", action="store_true", help="all available adopted datasets")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/mrage_figure3_mechanism/data",
    )
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--cache-mode", choices=("existing", "auto"), default="existing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = resolve_checkpoint_records(
        dataset=args.dataset,
        seeds=args.seeds,
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        all_datasets=args.all,
    )
    if not records:
        raise SystemExit("no checkpoints found for the requested selection")
    device = set_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern_rows_all: list[dict] = []
    diagnostic_rows_all: list[dict] = []
    pairwise_rows_all: list[dict] = []
    manifest: list[dict] = []
    failures: list[dict] = []
    for record in records:
        print(f"[router] {record.dataset} seed={record.training_seed} | {record.checkpoint}", flush=True)
        try:
            bundle = load_model_bundle(
                record,
                device=device,
                cache_root=args.cache_root,
                cache_mode=args.cache_mode,
            )
            with torch.inference_mode():
                embeddings = forward_full(bundle)
                metrics, _ = validation_metrics(embeddings, bundle.head, bundle.data, device)
                del embeddings
            capture = ContextCapture(
                bundle.model,
                capture_experts=False,
                capture_routing=True,
            ).install()
            capture.begin_forward()
            with torch.inference_mode():
                embeddings = forward_full(bundle)
                del embeddings
            capture.end_forward()
            capture.remove()
            if set(capture.routing_records) != set(CONTEXT_ORDER):
                raise RuntimeError(
                    "router capture did not see all six contexts: "
                    f"{sorted(capture.routing_records)}"
                )
            pattern_rows, diagnostic_rows, mean_distributions = routing_rows(
                dataset=record.dataset,
                training_seed=record.training_seed,
                checkpoint=str(record.checkpoint),
                records=capture.routing_records,
            )
            pattern_rows_all.extend(pattern_rows)
            diagnostic_rows_all.extend(diagnostic_rows)
            for first_index, first in enumerate(CONTEXT_ORDER):
                for second in CONTEXT_ORDER[first_index + 1:]:
                    first_subset = primary_subset(first)
                    second_subset = primary_subset(second)
                    key_first = f"{first}:{first_subset}"
                    key_second = f"{second}:{second_subset}"
                    if key_first in mean_distributions and key_second in mean_distributions:
                        pairwise_rows_all.append(
                            {
                                "dataset": record.dataset,
                                "training_seed": int(record.training_seed),
                                "checkpoint": str(record.checkpoint),
                                "context_a": first,
                                "context_b": second,
                                "subset_a": first_subset,
                                "subset_b": second_subset,
                                "comparison": "between_contexts",
                                "jsd": jensen_shannon_divergence(
                                    mean_distributions[key_first],
                                    mean_distributions[key_second],
                                ),
                            }
                        )
            for context in COLLABORATIVE_PHYSICAL:
                key_all = f"{context}:{primary_subset(context)}"
                key_cross = f"{context}:cross_nonzero"
                if key_all in mean_distributions and key_cross in mean_distributions:
                    pairwise_rows_all.append(
                        {
                            "dataset": record.dataset,
                            "training_seed": int(record.training_seed),
                            "checkpoint": str(record.checkpoint),
                            "context_a": context,
                            "context_b": context,
                            "subset_a": "physical_support",
                            "subset_b": "cross_nonzero",
                            "comparison": "within_context_support_vs_cross",
                            "jsd": jensen_shannon_divergence(
                                mean_distributions[key_all],
                                mean_distributions[key_cross],
                            ),
                        }
                    )
            manifest.append(
                {
                    "dataset": record.dataset,
                    "training_seed": int(record.training_seed),
                    "checkpoint": str(record.checkpoint),
                    "config": str(record.config),
                    "best_epoch": int(bundle.checkpoint.get("epoch", -1)),
                    "recomputed_val_ce": metrics["val_ce"],
                    "recomputed_val_acc": metrics["val_acc"],
                    "recomputed_val_macro_f1": metrics["val_macro_f1"],
                    "device": str(device),
                }
            )
            print(
                "[router]   "
                + " | ".join(
                    f"{context}:{capture.routing_records[context]['n_active']}"
                    for context in CONTEXT_ORDER
                ),
                flush=True,
            )
            del bundle
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "dataset": record.dataset,
                    "training_seed": int(record.training_seed),
                    "checkpoint": str(record.checkpoint),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[router] FAILED {record.dataset} seed={record.training_seed}: {exc}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_csv(output_dir / "routing_patterns.csv", PATTERN_FIELDS, pattern_rows_all)
    write_csv(output_dir / "routing_diagnostics.csv", DIAGNOSTIC_FIELDS, diagnostic_rows_all)
    write_csv(output_dir / "routing_pairwise_jsd.csv", PAIRWISE_FIELDS, pairwise_rows_all)
    write_json(output_dir / "manifest_router.json", {"runs": manifest, "failures": failures})
    print(
        f"[router] wrote {len(pattern_rows_all)} pattern rows, "
        f"{len(diagnostic_rows_all)} diagnostic rows, {len(pairwise_rows_all)} pairwise rows; "
        f"{len(failures)} failures",
        flush=True,
    )
    return 1 if failures and not manifest else 0


if __name__ == "__main__":
    raise SystemExit(main())
