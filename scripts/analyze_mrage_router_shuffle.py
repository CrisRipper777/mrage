#!/usr/bin/env python3
"""Figure 3(c): frozen Validation router-shuffle sensitivity.

The intervention permutes each node's complete four-dimensional router-logit
vector within the active node set of selected contexts.  Features, graphs,
experts, router parameters, and the NC head are unchanged.  Only Validation
labels are used for scoring; the official Test path is never called.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
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
    INTERVENTION_GROUPS,
    ContextCapture,
    forward_full,
    load_model_bundle,
    resolve_checkpoint_records,
    set_device,
    validation_metrics,
    write_csv,
    write_json,
)


TRIAL_FIELDS = [
    "dataset",
    "training_seed",
    "checkpoint",
    "intervention_group",
    "shuffle_seed",
    "repeat",
    "baseline_val_ce",
    "shuffled_val_ce",
    "delta_val_ce",
    "baseline_val_acc",
    "shuffled_val_acc",
    "delta_val_acc",
    "baseline_val_macro_f1",
    "shuffled_val_macro_f1",
    "delta_val_macro_f1",
    "prediction_disagreement",
    "n_active_total",
    "n_changed_total",
    "all_metrics_finite",
    "parameter_checksum_unchanged",
    "edge_index_checksum_unchanged",
]
SUMMARY_FIELDS = [
    "dataset",
    "training_seed",
    "checkpoint",
    "intervention_group",
    "n_repeats",
    "baseline_val_ce",
    "baseline_val_acc",
    "baseline_val_macro_f1",
    "mean_shuffled_val_ce",
    "std_shuffled_val_ce",
    "mean_delta_val_ce",
    "std_delta_val_ce",
    "mean_shuffled_val_acc",
    "std_shuffled_val_acc",
    "mean_delta_val_acc",
    "std_delta_val_acc",
    "mean_prediction_disagreement",
    "std_prediction_disagreement",
    "mean_n_active_total",
    "mean_n_changed_total",
]
DATASET_SUMMARY_FIELDS = [
    "dataset",
    "intervention_group",
    "n_checkpoints",
    "baseline_val_ce",
    "baseline_val_acc",
    "baseline_val_macro_f1",
    "mean_delta_val_ce",
    "std_delta_val_ce",
    "mean_delta_val_acc",
    "std_delta_val_acc",
    "mean_prediction_disagreement",
    "std_prediction_disagreement",
    "mean_n_active_total",
    "mean_n_changed_total",
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
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--shuffle-seed-base", type=int, default=1000)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=tuple(INTERVENTION_GROUPS),
        default=list(INTERVENTION_GROUPS),
    )
    parser.add_argument(
        "--include-single-contexts",
        action="store_true",
        help="also run each of the six contexts separately",
    )
    parser.add_argument(
        "--no-identity-check",
        action="store_false",
        dest="identity_check",
        help="skip the identity-permutation equality check",
    )
    parser.set_defaults(identity_check=True)
    return parser.parse_args()


def _checksum_tensor(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _parameter_checksum(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _evaluate(
    bundle,
    *,
    shuffle_group: tuple[str, ...] | None = None,
    shuffle_seed: int | None = None,
    shuffle_mode: str = "none",
) -> tuple[dict[str, float], torch.Tensor, list[dict], torch.Tensor]:
    capture.begin_forward(
        shuffle_group=shuffle_group,
        shuffle_seed=shuffle_seed,
        shuffle_mode=shuffle_mode,
    )
    with torch.inference_mode():
        embeddings = forward_full(bundle)
        metrics, predictions = validation_metrics(
            embeddings, bundle.head, bundle.data, bundle.device
        )
    capture.end_forward()
    metadata = list(capture.last_shuffle_metadata)
    return metrics, embeddings.detach().cpu(), metadata, predictions


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(statistics.fmean(values)), float(statistics.pstdev(values))


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
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
    trial_rows: list[dict] = []
    summary_rows: list[dict] = []
    dataset_summary_rows: list[dict] = []
    manifest: list[dict] = []
    failures: list[dict] = []
    checks: list[dict] = []
    group_specs: list[tuple[str, tuple[str, ...]]] = [
        (name, tuple(INTERVENTION_GROUPS[name])) for name in args.groups
    ]
    if args.include_single_contexts:
        group_specs.extend((context, (context,)) for context in CONTEXT_ORDER)
    for record in records:
        print(f"[shuffle] {record.dataset} seed={record.training_seed} | {record.checkpoint}", flush=True)
        global capture  # used by _evaluate
        try:
            bundle = load_model_bundle(
                record,
                device=device,
                cache_root=args.cache_root,
                cache_mode=args.cache_mode,
            )
            parameter_before = _parameter_checksum(bundle.model)
            edge_before = _checksum_tensor(bundle.edge_index)
            capture = ContextCapture(
                bundle.model,
                capture_experts=False,
                capture_routing=False,
                shuffle_capable=True,
            ).install()
            baseline_metrics, baseline_embeddings, _, baseline_predictions = _evaluate(bundle)
            identity_max_abs_diff = None
            if args.identity_check:
                identity_metrics, identity_embeddings, identity_metadata, identity_predictions = _evaluate(
                    bundle,
                    shuffle_group=CONTEXT_ORDER,
                    shuffle_seed=int(args.shuffle_seed_base),
                    shuffle_mode="identity",
                )
                identity_max_abs_diff = float(
                    (identity_embeddings - baseline_embeddings).abs().max().item()
                )
                identity_prediction_mismatch = float(
                    (identity_predictions != baseline_predictions).float().mean().item()
                )
                identity_metric_deltas = {
                    "val_ce": float(identity_metrics["val_ce"] - baseline_metrics["val_ce"]),
                    "val_acc": float(identity_metrics["val_acc"] - baseline_metrics["val_acc"]),
                    "val_macro_f1": float(
                        identity_metrics["val_macro_f1"] - baseline_metrics["val_macro_f1"]
                    ),
                }
                # CUDA sparse reductions can introduce tiny run-to-run noise even
                # without an intervention.  Predictions must be unchanged; allow
                # only sub-1e-3 embedding drift for the identity check.
                if identity_prediction_mismatch != 0.0 or identity_max_abs_diff > 1.0e-3:
                    raise RuntimeError(
                        "identity permutation changed the frozen forward: "
                        f"max_abs_diff={identity_max_abs_diff}, "
                        f"prediction_mismatch={identity_prediction_mismatch}"
                    )
                checks.append(
                    {
                        "dataset": record.dataset,
                        "training_seed": int(record.training_seed),
                        "identity_max_abs_diff": identity_max_abs_diff,
                        "identity_prediction_mismatch": identity_prediction_mismatch,
                        "identity_metric_deltas": identity_metric_deltas,
                        "identity_n_changed": int(
                            sum(item["n_changed"] for item in identity_metadata)
                        ),
                    }
                )
                del identity_metrics, identity_embeddings, identity_predictions
            per_group_trials: dict[str, list[dict]] = {}
            for group_name, group_contexts in group_specs:
                group_trials: list[dict] = []
                for repeat in range(int(args.repeats)):
                    shuffle_seed = int(args.shuffle_seed_base) + repeat
                    metrics, embeddings, metadata, predictions = _evaluate(
                        bundle,
                        shuffle_group=group_contexts,
                        shuffle_seed=shuffle_seed,
                        shuffle_mode="shuffle",
                    )
                    del embeddings
                    disagreement = float(
                        (predictions != baseline_predictions).float().mean().item()
                    )
                    n_active_total = int(sum(item["n_active"] for item in metadata))
                    n_changed_total = int(sum(item["n_changed"] for item in metadata))
                    all_finite = all(
                        torch.isfinite(torch.tensor(value))
                        for value in (
                            metrics["val_ce"],
                            metrics["val_acc"],
                            metrics["val_macro_f1"],
                        )
                    )
                    row = {
                        "dataset": record.dataset,
                        "training_seed": int(record.training_seed),
                        "checkpoint": str(record.checkpoint),
                        "intervention_group": group_name,
                        "shuffle_seed": shuffle_seed,
                        "repeat": repeat,
                        "baseline_val_ce": baseline_metrics["val_ce"],
                        "shuffled_val_ce": metrics["val_ce"],
                        "delta_val_ce": metrics["val_ce"] - baseline_metrics["val_ce"],
                        "baseline_val_acc": baseline_metrics["val_acc"],
                        "shuffled_val_acc": metrics["val_acc"],
                        "delta_val_acc": baseline_metrics["val_acc"] - metrics["val_acc"],
                        "baseline_val_macro_f1": baseline_metrics["val_macro_f1"],
                        "shuffled_val_macro_f1": metrics["val_macro_f1"],
                        "delta_val_macro_f1": baseline_metrics["val_macro_f1"]
                        - metrics["val_macro_f1"],
                        "prediction_disagreement": disagreement,
                        "n_active_total": n_active_total,
                        "n_changed_total": n_changed_total,
                        "all_metrics_finite": bool(all_finite),
                        "parameter_checksum_unchanged": True,
                        "edge_index_checksum_unchanged": True,
                    }
                    trial_rows.append(row)
                    group_trials.append(row)
                per_group_trials[group_name] = group_trials
                delta_ce = [row["delta_val_ce"] for row in group_trials]
                delta_acc = [row["delta_val_acc"] for row in group_trials]
                disagreement = [row["prediction_disagreement"] for row in group_trials]
                mean_delta_ce, std_delta_ce = _mean_std(delta_ce)
                mean_delta_acc, std_delta_acc = _mean_std(delta_acc)
                mean_disagreement, std_disagreement = _mean_std(disagreement)
                summary_rows.append(
                    {
                        "dataset": record.dataset,
                        "training_seed": int(record.training_seed),
                        "checkpoint": str(record.checkpoint),
                        "intervention_group": group_name,
                        "n_repeats": len(group_trials),
                        "baseline_val_ce": baseline_metrics["val_ce"],
                        "baseline_val_acc": baseline_metrics["val_acc"],
                        "baseline_val_macro_f1": baseline_metrics["val_macro_f1"],
                        "mean_shuffled_val_ce": statistics.fmean(
                            row["shuffled_val_ce"] for row in group_trials
                        ),
                        "std_shuffled_val_ce": statistics.pstdev(
                            row["shuffled_val_ce"] for row in group_trials
                        ),
                        "mean_delta_val_ce": mean_delta_ce,
                        "std_delta_val_ce": std_delta_ce,
                        "mean_shuffled_val_acc": statistics.fmean(
                            row["shuffled_val_acc"] for row in group_trials
                        ),
                        "std_shuffled_val_acc": statistics.pstdev(
                            row["shuffled_val_acc"] for row in group_trials
                        ),
                        "mean_delta_val_acc": mean_delta_acc,
                        "std_delta_val_acc": std_delta_acc,
                        "mean_prediction_disagreement": mean_disagreement,
                        "std_prediction_disagreement": std_disagreement,
                        "mean_n_active_total": statistics.fmean(
                            row["n_active_total"] for row in group_trials
                        ),
                        "mean_n_changed_total": statistics.fmean(
                            row["n_changed_total"] for row in group_trials
                        ),
                    }
                )
                print(
                    f"[shuffle]   {group_name}: delta_ce={mean_delta_ce:+.4f} "
                    f"delta_acc={mean_delta_acc:+.4f} "
                    f"disagree={mean_disagreement:.4f} "
                    f"active={int(statistics.fmean(row['n_active_total'] for row in group_trials))}",
                    flush=True,
                )
            parameter_after = _parameter_checksum(bundle.model)
            edge_after = _checksum_tensor(bundle.edge_index)
            if parameter_before != parameter_after:
                raise RuntimeError("model parameters changed during frozen analysis")
            if edge_before != edge_after:
                raise RuntimeError("edge_index changed during frozen analysis")
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
                    "recomputed_val_ce": baseline_metrics["val_ce"],
                    "recomputed_val_acc": baseline_metrics["val_acc"],
                    "recomputed_val_macro_f1": baseline_metrics["val_macro_f1"],
                    "identity_max_abs_diff": identity_max_abs_diff,
                    "parameter_checksum_before": parameter_before,
                    "parameter_checksum_after": parameter_after,
                    "edge_index_checksum_before": edge_before,
                    "edge_index_checksum_after": edge_after,
                    "device": str(device),
                }
            )
            capture.remove()
            del bundle, baseline_embeddings, baseline_predictions
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
            print(f"[shuffle] FAILED {record.dataset} seed={record.training_seed}: {exc}", flush=True)
            if "capture" in globals():
                try:
                    capture.remove()
                except Exception:  # noqa: BLE001
                    pass
            if device.type == "cuda":
                torch.cuda.empty_cache()
    # Dataset-level pooling: average the per-seed summary means.
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in summary_rows:
        grouped.setdefault((row["dataset"], row["intervention_group"]), []).append(row)
    for (dataset, group_name), rows in sorted(grouped.items()):
        delta_ce = [float(row["mean_delta_val_ce"]) for row in rows]
        delta_acc = [float(row["mean_delta_val_acc"]) for row in rows]
        disagreement = [float(row["mean_prediction_disagreement"]) for row in rows]
        mean_delta_ce, std_delta_ce = _mean_std(delta_ce)
        mean_delta_acc, std_delta_acc = _mean_std(delta_acc)
        mean_disagreement, std_disagreement = _mean_std(disagreement)
        dataset_summary_rows.append(
            {
                "dataset": dataset,
                "intervention_group": group_name,
                "n_checkpoints": len(rows),
                "baseline_val_ce": statistics.fmean(
                    float(row["baseline_val_ce"]) for row in rows
                ),
                "baseline_val_acc": statistics.fmean(
                    float(row["baseline_val_acc"]) for row in rows
                ),
                "baseline_val_macro_f1": statistics.fmean(
                    float(row["baseline_val_macro_f1"]) for row in rows
                ),
                "mean_delta_val_ce": mean_delta_ce,
                "std_delta_val_ce": std_delta_ce,
                "mean_delta_val_acc": mean_delta_acc,
                "std_delta_val_acc": std_delta_acc,
                "mean_prediction_disagreement": mean_disagreement,
                "std_prediction_disagreement": std_disagreement,
                "mean_n_active_total": statistics.fmean(
                    float(row["mean_n_active_total"]) for row in rows
                ),
                "mean_n_changed_total": statistics.fmean(
                    float(row["mean_n_changed_total"]) for row in rows
                ),
            }
        )
    write_csv(output_dir / "router_shuffle_trials.csv", TRIAL_FIELDS, trial_rows)
    write_csv(output_dir / "router_shuffle_summary.csv", SUMMARY_FIELDS, summary_rows)
    write_csv(
        output_dir / "router_shuffle_dataset_summary.csv",
        DATASET_SUMMARY_FIELDS,
        dataset_summary_rows,
    )
    write_json(
        output_dir / "manifest_shuffle.json",
        {"runs": manifest, "checks": checks, "failures": failures},
    )
    print(
        f"[shuffle] wrote {len(trial_rows)} trial rows, {len(summary_rows)} seed summaries, "
        f"{len(dataset_summary_rows)} dataset summaries; {len(failures)} failures",
        flush=True,
    )
    return 1 if failures and not manifest else 0


if __name__ == "__main__":
    raise SystemExit(main())
