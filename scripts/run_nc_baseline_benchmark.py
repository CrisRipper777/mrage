#!/usr/bin/env python3
"""Run the 10 non-MRAGE baselines on the five fixed NC datasets, seeds 42-44."""

from __future__ import annotations

from run_baseline_matrix import BASELINE_MODELS, _launch_matrix, _parse_common_args


DATASETS = ("Movies", "Grocery", "Toys", "Reddit-S", "ele-fashion")
PROTOCOL_OVERRIDES = (
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
    "task.macro_f1_labels=all_classes",
    "task.scheduler=null",
)
PROTOCOL_DESCRIPTION = {
    "training": "Full-graph NC; one run per seed; maximum 300 epochs; early stop on validation accuracy.",
    "selection": "Best validation accuracy; validation macro-F1 uses all configured classes.",
    "test": "Test labels are masked during training; the selected checkpoint is evaluated once in a separate final-test phase.",
    "split": "Fixed NC split seed 42 for all training seeds and all models; Books-NC is excluded.",
    "optimizer": "AdamW; shared task LR/weight decay are 1e-3/1e-4 unless a model config defines its own LR or weight decay.",
    "model_hyperparameters": "Architecture and model-specific settings come from configs/model/<model>.yaml.",
}


def main() -> int:
    args = _parse_common_args(__doc__ or "Run the NC baseline benchmark.")
    return _launch_matrix(
        task="nc",
        datasets=DATASETS,
        models=BASELINE_MODELS,
        protocol_version="unified_full_graph_nc_v1",
        protocol_overrides=PROTOCOL_OVERRIDES,
        protocol_description=PROTOCOL_DESCRIPTION,
        args=args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
