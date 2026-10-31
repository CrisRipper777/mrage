#!/usr/bin/env python3
"""Add GraphMoRE and NTSFormer to the fixed five-dataset NC benchmark."""

from __future__ import annotations

from run_baseline_matrix import _launch_matrix, _parse_common_args


DATASETS = ("Movies", "Grocery", "Toys", "Reddit-S", "ele-fashion")
MODELS = ("graphmore", "ntsformer")
PROTOCOL_OVERRIDES = (
    "task.protocol_version=unified_full_graph_nc_method_objectives_v4",
    "task.training_mode=full_graph",
    "task.optimizer=adamw",
    "task.epochs=300",
    "task.lr=0.001",
    "task.weight_decay=0.0001",
    "task.eval_every=1",
    "task.patience=30",
    "task.early_stop_min_epoch=30",
    "task.early_stop_min_delta=0.0001",
    "task.grad_clip=1.0",
    "task.loss.aux_weight=1.0",
    "task.loss.unimodal_weight=0.0",
    "task.loss.class_balanced_main_ce=false",
    "task.scheduler=null",
)
PROTOCOL_DESCRIPTION = {
    "training": "Full-graph transductive NC; one run per seed; maximum 300 epochs; early stop on validation accuracy.",
    "selection": "Best validation accuracy; Macro-F1 follows each dataset's declared evaluation class set.",
    "test": "Test labels are masked during training; selected checkpoints are evaluated once in a separate final-Test phase.",
    "split": "Fixed NC split seed 42 for all three training seeds and datasets; Books-NC is excluded.",
    "heads": "Both models use the shared linear NC classifier and shared cross-entropy task loss.",
    "graphmore": "Five trainable signed-curvature experts share one curvature per expert; outputs receive LayerNorm and are gate-weighted then concatenated with raw features before the shared linear classifier. The original per-node NetworkX ego expansion is replaced by sparse 2/3-hop neighborhood pooling; the native distortion objective is retained.",
    "ntsformer": "Shared classifier is applied to the self student and two contextual teacher views. The averaged teacher representation receives shared NC cross-entropy; teacher-to-student KL and MoE load balancing are retained. Full-graph SIGN contexts are precomputed once, with node-batched gradient accumulation and batched inference. KL and MoE auxiliary objectives cover all graph nodes without using validation/test labels; evaluation uses the contextual teacher view.",
}


def main() -> int:
    args = _parse_common_args(__doc__ or "Run additional NC baseline models.")
    return _launch_matrix(
        task="nc",
        datasets=DATASETS,
        models=MODELS,
        protocol_version="unified_full_graph_nc_method_objectives_v4",
        protocol_overrides=PROTOCOL_OVERRIDES,
        protocol_description=PROTOCOL_DESCRIPTION,
        args=args,
        manifest_filename="nc_additional_baseline_benchmark_v4.json",
        log_filename="nc_additional_baseline_benchmark_v4.log",
    )


if __name__ == "__main__":
    raise SystemExit(main())
