#!/usr/bin/env python3
"""Run the 12 non-MRAGE baselines on the five fixed NC datasets, seeds 42-44."""

from __future__ import annotations

from run_baseline_matrix import BASELINE_MODELS, _launch_matrix, _parse_common_args


DATASETS = ("Movies", "Grocery", "Toys", "Reddit-S", "ele-fashion")
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
    "training": "Full-graph NC; one run per seed; maximum 300 epochs; early stop on validation accuracy.",
    "selection": "Best validation accuracy; Macro-F1 uses dataset-declared observed classes (all configured classes by default).",
    "test": "Test labels are masked during training; the selected checkpoint is evaluated once in a separate final-test phase.",
    "split": "Fixed NC split seed 42 for all training seeds and all models; Books-NC is excluded.",
    "optimizer": "AdamW; shared task LR/weight decay are 1e-3/1e-4 unless a model config defines its own LR or weight decay. DMGC retains the OpenMAG reference preset 5e-3/1e-5.",
    "objective": (
        "Shared task cross-entropy plus selected model-native objectives: DGF cross-modal, "
        "random-walk graph, and community losses; DMGC dual-frequency and cross-modal "
        "contrastive losses (NC lambda_cm=0.3, selected by Validation Accuracy across five datasets); "
        "RoleMAG paper EVI, role-balance, and QCA losses; GraphMoRE embedding distortion; "
        "and NTSFormer teacher-to-student KL plus MoE load balancing. Remaining baselines "
        "use task loss only. OpenMAG's generic visual/text InfoNCE is excluded."
    ),
    "auxiliary_memory_adapter": (
        "DGF and RoleMAG QCA use 1024-node contrastive subsets; DMGC uses its OpenMAG "
        "2048-node contrastive cap. This bounds full-graph NC memory while preserving "
        "the original sampled contrastive objective."
    ),
    "model_hyperparameters": (
        "Architecture and model-specific settings come from configs/model/<model>.yaml; "
        "GCN/SAGE/MMGCN normalization is aligned with OpenMAG (no hidden BatchNorm), "
        "and SAGE dropout is 0.02."
    ),
}


def main() -> int:
    args = _parse_common_args(__doc__ or "Run the NC baseline benchmark.")
    return _launch_matrix(
        task="nc",
        datasets=DATASETS,
        models=BASELINE_MODELS,
        protocol_version="unified_full_graph_nc_method_objectives_v4",
        protocol_overrides=PROTOCOL_OVERRIDES,
        protocol_description=PROTOCOL_DESCRIPTION,
        args=args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
