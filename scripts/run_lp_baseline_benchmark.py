#!/usr/bin/env python3
"""Run default MRAGE first, then 10 LP baselines on Sports and Cloth, seeds 42-44.

MLP and the NC-only NTSFormer are excluded. Both MRAGE datasets must finish
successfully before any baseline starts. No NC tuning overlays are loaded.
"""

from __future__ import annotations

from run_baseline_matrix import LP_SUPPORTED_MODELS, _launch_matrix, _parse_common_args


DATASETS = ("sports-copurchase", "cloth-copurchase")
BASELINE_MODELS = tuple(model for model in LP_SUPPORTED_MODELS if model != "mlp")
PROTOCOL_OVERRIDES = (
    "task.training_mode=sampled",
    "task.optimizer=adam",
    "task.epochs=150",
    "task.lr=0.001",
    "task.weight_decay=0.00001",
    "task.patience=10",
    "task.early_stop_min_epoch=20",
    "task.early_stop_min_delta=0.0001",
    "task.grad_clip=1.0",
    "task.batch_size=2048",
    "task.num_neighbors=[5,5,5]",
    "task.subgraph_type=bidirectional",
    "task.num_train_neg=1",
    "task.positive_edge_mask_backend=global_eid",
    "task.loader_num_workers=4",
    "task.loader_prefetch_factor=2",
    "task.torch_threads=8",
    "task.negative_seed_offset=10000",
    "task.shuffle_seed_offset=20000",
    "task.neighbor_seed_offset=30000",
    "task.inference_mode=full",
    "task.inference_batch_size=4096",
    "task.eval_preload_node_emb=true",
    "task.eval_every=2",
    "task.max_train_batches=null",
    "task.train_pos_per_epoch=null",
    "task.eval_edge_batch_size=512",
    "task.profile_timing=false",
    "task.loss.aux_weight=1.0",
    "task.decoder.name=mlp",
    "task.decoder.hidden_dim=256",
    "task.decoder.num_layers=3",
    "task.decoder.dropout=0.02",
    "task.decoder.proj_dim=128",
)
PROTOCOL_DESCRIPTION = {
    "training": "Sampled LP with LinkNeighborLoader; one run per seed; maximum 150 epochs; all graph encoders use the same three fanouts [5,5,5]. DGF's 10 filter iterations operate within those sampled subgraphs and do not increase sampler depth.",
    "selection": "Best validation MRR; validation every 2 epochs with patience 10.",
    "test": "The selected checkpoint is evaluated once in a separate final-test phase.",
    "split": "Each dataset uses its official fixed LP split (lp-edge-split.pt), shared across training seeds 42, 43, 44 and all models; seeds vary training randomness, not the split.",
    "negative_sampling": "One filtered train negative per positive; validation/test use the fixed negatives in the dataset split files.",
    "optimizer": "Adam; shared task LR/weight decay are 1e-3/1e-5 unless a model config defines its own LR or weight decay.",
    "decoder": "Shared MLP decoder with hidden dimension 256, 3 layers, dropout 0.02 and projection dimension 128.",
    "model_hyperparameters": "Architecture and model-specific settings come from configs/model/<model>.yaml.",
    "sampled_model_context": "GraphMoRE curvature experts, distortion, and sparse 2/3-hop gate pooling operate on each [5,5,5] sampled message subgraph; DiP pseudo-node aggregation, RoleMAG routing, and LGMRec hypergraph aggregation are likewise computed on each sampled message subgraph. Validation and final testing use the full train graph.",
}


def main() -> int:
    args = _parse_common_args(__doc__ or "Run the LP benchmark.", default_device="cuda:1")
    print("Phase 1/2: default MRAGE on Sports, then Cloth (three seeds each).", flush=True)
    return_code = _launch_matrix(
        task="lp",
        datasets=DATASETS,
        models=("mrage",),
        protocol_version="unified_sampled_lp_v1",
        protocol_overrides=PROTOCOL_OVERRIDES,
        protocol_description=PROTOCOL_DESCRIPTION,
        args=args,
        manifest_filename="lp_mrage_benchmark.json",
        log_filename="lp_mrage_benchmark.log",
        benchmark_name="mrage_lp_matrix",
    )
    if return_code:
        print("MRAGE phase failed; baseline phase was not started.", flush=True)
        return return_code

    print("Phase 2/2: 10 baselines on Sports, then Cloth (three seeds each).", flush=True)
    return _launch_matrix(
        task="lp",
        datasets=DATASETS,
        models=BASELINE_MODELS,
        protocol_version="unified_sampled_lp_v1",
        protocol_overrides=PROTOCOL_OVERRIDES,
        protocol_description=PROTOCOL_DESCRIPTION,
        args=args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
