#!/usr/bin/env python3
"""Figure 3(a): learned graph-filter coefficients and expert output diversity.

Reads frozen Full MRAGE checkpoints only.  No training, Test evaluation, or
model/data mutation is performed.
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
    CONTEXT_ORDER,
    DATASETS,
    DEFAULT_CACHE_ROOT,
    ContextCapture,
    coefficient_diversity_rows,
    coefficient_rows,
    forward_full,
    load_model_bundle,
    resolve_checkpoint_records,
    set_device,
    validation_metrics,
    write_csv,
    write_json,
)


COEFFICIENT_FIELDS = [
    "dataset",
    "training_seed",
    "checkpoint",
    "relation",
    "expert_id",
    "order",
    "initialized_coefficient",
    "trained_coefficient",
    "coefficient_delta",
]
DIVERSITY_FIELDS = [
    "dataset",
    "training_seed",
    "checkpoint",
    "metric",
    "relation",
    "expert_id",
    "expert_j",
    "context",
    "value",
    "n_active",
    "n_sampled",
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
    parser.add_argument("--max-nodes", type=int, default=4096)
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
    coefficient_rows_all: list[dict] = []
    diversity_rows_all: list[dict] = []
    manifest: list[dict] = []
    failures: list[dict] = []
    for record in records:
        print(f"[experts] {record.dataset} seed={record.training_seed} | {record.checkpoint}", flush=True)
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
            theta = bundle.model.filter_bank.theta.detach().cpu().clone()
            coefficient_rows_all.extend(
                coefficient_rows(
                    dataset=record.dataset,
                    training_seed=record.training_seed,
                    checkpoint=str(record.checkpoint),
                    theta=theta,
                )
            )
            diversity_rows_all.extend(
                coefficient_diversity_rows(
                    dataset=record.dataset,
                    training_seed=record.training_seed,
                    checkpoint=str(record.checkpoint),
                    theta=theta,
                )
            )
            capture = ContextCapture(
                bundle.model,
                capture_experts=True,
                capture_routing=False,
                max_nodes=int(args.max_nodes),
            ).install()
            capture.begin_forward()
            with torch.inference_mode():
                embeddings = forward_full(bundle)
                del embeddings
            capture.end_forward()
            capture.remove()
            for context in CONTEXT_ORDER:
                expert_record = capture.expert_records.get(context)
                if expert_record is None:
                    continue
                n_active = int(expert_record["n_active"])
                n_sampled = int(expert_record["n_sampled"])
                for (first, second), value in expert_record["cosine"].items():
                    diversity_rows_all.append(
                        {
                            "dataset": record.dataset,
                            "training_seed": int(record.training_seed),
                            "checkpoint": str(record.checkpoint),
                            "metric": "expert_output_cosine",
                            "relation": "physical" if "physical" in context else "semantic",
                            "expert_id": first + 1,
                            "expert_j": second + 1,
                            "context": context,
                            "value": float(value),
                            "n_active": n_active,
                            "n_sampled": n_sampled,
                        }
                    )
                for expert_id, value in expert_record["mean_norm"].items():
                    diversity_rows_all.append(
                        {
                            "dataset": record.dataset,
                            "training_seed": int(record.training_seed),
                            "checkpoint": str(record.checkpoint),
                            "metric": "expert_output_mean_l2",
                            "relation": "physical" if "physical" in context else "semantic",
                            "expert_id": expert_id + 1,
                            "expert_j": "",
                            "context": context,
                            "value": float(value),
                            "n_active": n_active,
                            "n_sampled": n_sampled,
                        }
                    )
            manifest.append(
                {
                    "dataset": record.dataset,
                    "training_seed": int(record.training_seed),
                    "checkpoint": str(record.checkpoint),
                    "config": str(record.config),
                    "best_epoch": int(bundle.checkpoint.get("epoch", -1)),
                    "saved_val_acc": float(
                        bundle.checkpoint.get("metrics", {}).get("val_acc", float("nan"))
                    ),
                    "saved_val_macro_f1": float(
                        bundle.checkpoint.get("metrics", {}).get("val_macro_f1", float("nan"))
                    ),
                    "recomputed_val_ce": metrics["val_ce"],
                    "recomputed_val_acc": metrics["val_acc"],
                    "recomputed_val_macro_f1": metrics["val_macro_f1"],
                    "device": str(device),
                }
            )
            print(
                f"[experts]   val_acc={metrics['val_acc']:.4f} "
                f"val_f1={metrics['val_macro_f1']:.4f} "
                f"contexts={len(capture.expert_records)}",
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
            print(f"[experts] FAILED {record.dataset} seed={record.training_seed}: {exc}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_csv(output_dir / "expert_coefficients.csv", COEFFICIENT_FIELDS, coefficient_rows_all)
    write_csv(output_dir / "expert_diversity.csv", DIVERSITY_FIELDS, diversity_rows_all)
    write_json(output_dir / "manifest_experts.json", {"runs": manifest, "failures": failures})
    print(
        f"[experts] wrote {len(coefficient_rows_all)} coefficient rows, "
        f"{len(diversity_rows_all)} diversity rows; {len(failures)} failures",
        flush=True,
    )
    return 1 if failures and not manifest else 0


if __name__ == "__main__":
    raise SystemExit(main())
