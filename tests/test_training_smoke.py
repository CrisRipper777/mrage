from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.data.semantic_graph_manager import ensure_semantic_graph_cache
from src.data.types import EdgeSplit, MAGData
from src.models.factory import build_model
from src.tasks.lp import run_lp
from src.tasks.lp import evaluate_lp_checkpoint
from src.tasks.nc import evaluate_nc_checkpoint, run_nc
from src.utils.run_artifacts import write_json_atomic


def _model_config(name: str) -> dict:
    if name == "mrage":
        return {
            "name": "mrage", "hidden_dim": 256, "physical_hops": 3,
            "semantic_hops": 1, "num_experts": 4, "top_k": 2,
            "expert_bottleneck": 64, "expert_eta": 0.1, "dropout": 0.0,
            "late_cross_dropout": 0.0, "fusion_dropout": 0.0,
            "balance_weight": 0.01, "rho_init": 0.25,
            "gamma_init": 0.08, "beta_init": 0.25,
            "same_node_enabled": True, "semantic_knn_k": 8,
            "nc_unimodal_weight": 0.3, "eps": 1.0e-8,
            "physical_chunk_size": 128, "semantic_chunk_size": 128,
            "cross_edge_chunk_size": 64, "attention_dim": 32,
            "router_hidden_dim": 256, "router_embed_dim": 8,
            "channel_gate_hidden": 128,
            "semantic_context_physical_fanout": [1, 1, 1],
        }
    return {
        "name": "gcn", "hidden_dim": 16, "num_layers": 2,
        "dropout": 0.0, "activation": "relu", "norm": "none",
    }


def _features() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1907)
    text = torch.randn(24, 7, generator=generator)
    visual = torch.randn(24, 5, generator=generator)
    return text, visual, torch.cat((text, visual), dim=1)


def _nc_cfg(model_name: str, *, checkpoint: Path | None = None):
    cfg = OmegaConf.create(
        {
            "seed": 37,
            "num_runs": 1,
            "dataset": {"name": "Smoke", "make_undirected": True},
            "model": _model_config(model_name),
            "task": {
                "name": "nc", "protocol_version": "smoke_nc",
                "training_mode": "full_graph", "optimizer": "adamw",
                "epochs": 1, "lr": 1.0e-3, "weight_decay": 0.0,
                "batch_size": 8, "inference_mode": "full",
                "inference_batch_size": 24, "eval_every": 1,
                "patience": 1, "early_stop_min_epoch": 1,
                "early_stop_min_delta": 0.0, "grad_clip": 1.0,
                "max_train_batches": None, "save_ckpt_path": str(checkpoint) if checkpoint else None,
                "run_metrics_path": None, "scheduler": None,
                "scheduler_warmup_epochs": 1, "scheduler_min_lr": 1.0e-5,
                "loss": {
                    "aux_weight": 1.0,
                    "unimodal_weight": 0.3 if model_name == "mrage" else 0.0,
                    "class_balanced_main_ce": False,
                },
                "evaluate_test": True,
            },
        }
    )
    if model_name == "mrage":
        cfg.semantic_graph = {
            "resolved_cache_path": None,
            "identity_fingerprint": None,
            "feature_fingerprint": None,
            "candidate_fingerprint": None,
            "v7b_fingerprint": None,
        }
    return cfg


def _lp_cfg(cache, checkpoint: Path):
    return OmegaConf.create(
        {
            "seed": 41,
            "num_runs": 1,
            "dataset": {"name": "Smoke", "make_undirected": True},
            "model": _model_config("mrage"),
            "semantic_graph": {
                "resolved_cache_path": str(cache.path),
                "identity_fingerprint": cache.identity_fingerprint,
                "feature_fingerprint": cache.metadata["feature_fingerprint"],
                "candidate_fingerprint": cache.metadata["candidate_fingerprint"],
                "v7b_fingerprint": cache.metadata["v7b_fingerprint"],
            },
            "task": {
                "name": "lp", "protocol_version": "smoke_sampled_lp",
                "training_mode": "sampled", "optimizer": "adam",
                "epochs": 1, "lr": 1.0e-3, "weight_decay": 0.0,
                "patience": 1, "early_stop_min_epoch": 1,
                "early_stop_min_delta": 0.0, "grad_clip": 1.0,
                "batch_size": 4, "num_neighbors": [2, 2, 2],
                "subgraph_type": "bidirectional", "num_train_neg": 1,
                "positive_edge_mask_backend": "global_eid",
                "loader_num_workers": 0, "negative_seed_offset": 10000,
                "shuffle_seed_offset": 20000, "neighbor_seed_offset": 30000,
                "inference_mode": "full", "inference_batch_size": 24,
                "eval_preload_node_emb": True, "eval_every": 1,
                "max_train_batches": 1, "train_pos_per_epoch": 4,
                "eval_edge_batch_size": 2, "torch_threads": 1,
                "loss": {"aux_weight": 1.0},
                "decoder": {
                    "name": "mlp", "hidden_dim": 8, "num_layers": 1,
                    "dropout": 0.0, "proj_dim": 8,
                },
                "save_ckpt_path": str(checkpoint), "evaluate_test": True,
            },
        }
    )


def _synthetic_data(text: torch.Tensor, visual: torch.Tensor, x: torch.Tensor) -> tuple[MAGData, MAGData]:
    n = x.size(0)
    nodes = torch.arange(n)
    ring = torch.stack((nodes, (nodes + 1) % n))
    skip = torch.stack((nodes, (nodes + 4) % n))
    nc_edges = torch.cat((ring, skip, ring.flip(0), skip.flip(0)), dim=1)
    labels = nodes % 3
    nc_data = MAGData(
        name="Smoke", source="synthetic", task="nc", x=x,
        x_i=visual, x_t=text, edge_index=nc_edges, y=labels,
        train_idx=nodes[:16], val_idx=nodes[16:20], test_idx=nodes[20:24],
        num_nodes=n, num_classes=3,
        info={"nc_split_path": "synthetic-nc-split.pt", "nc_split_sha256": "synthetic-nc-split-sha"},
    )

    train_src = torch.arange(12)
    valid_src = torch.arange(12, 16)
    test_src = torch.arange(16, 20)
    neg_offsets = torch.tensor([5, 8, 10])

    def section(src: torch.Tensor, offset: int) -> dict[str, torch.Tensor]:
        return {
            "source_node": src,
            "target_node": (src + 1) % n,
            "target_node_neg": (src[:, None] + neg_offsets[None, :] + offset) % n,
        }

    edge_split = EdgeSplit(
        train={"source_node": train_src, "target_node": train_src + 1},
        valid=section(valid_src, 0),
        test=section(test_src, 1),
        metadata={"undirected": True},
    )
    train_edges = edge_split.train
    train_graph = torch.stack(
        (
            torch.cat((train_edges["source_node"], train_edges["target_node"])),
            torch.cat((train_edges["target_node"], train_edges["source_node"])),
        )
    )
    lp_data = MAGData(
        name="Smoke", source="synthetic", task="lp", x=x,
        x_i=visual, x_t=text, edge_index=train_graph,
        edge_split=edge_split, num_nodes=n,
        info={"lp_split_path": "synthetic-lp-split.pt", "lp_split_sha256": "synthetic-lp-split-sha"},
    )
    return nc_data, lp_data


def test_cpu_nc_lp_and_baseline_one_epoch_smokes(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    text, visual, x = _features()
    nc_data, lp_data = _synthetic_data(text, visual, x)
    logger = logging.getLogger("mrage-smoke")
    logger.addHandler(logging.NullHandler())

    nc_cache = ensure_semantic_graph_cache(
        dataset="Smoke", task="nc", text_features=text,
        visual_features=visual, physical_edge_index=nc_data.edge_index,
        config={
            "mode": "auto", "knn_k": 8, "cache_dir": str(tmp_path / "semantic"),
            "validate": True, "faiss_seed": 2026, "faiss_threads": 1,
        },
        project_root=tmp_path,
    )
    nc_cfg = _nc_cfg("mrage", checkpoint=tmp_path / "nc.pt")
    nc_cfg.semantic_graph.update(
        {
            "resolved_cache_path": str(nc_cache.path),
            "identity_fingerprint": nc_cache.identity_fingerprint,
            "feature_fingerprint": nc_cache.metadata["feature_fingerprint"],
            "candidate_fingerprint": nc_cache.metadata["candidate_fingerprint"],
            "v7b_fingerprint": nc_cache.metadata["v7b_fingerprint"],
        }
    )
    nc_cfg.task.evaluate_test = False
    nc_cfg.task.mask_test_labels = True
    nc_cfg.task.checkpoint_path = str(tmp_path / "nc.pt")
    nc_cfg.task.validation_metrics_path = str(tmp_path / "nc_validation.json")
    nc_cfg.task.final_test_metrics_path = str(tmp_path / "nc_final.json")
    nc_cfg.task.paper_run_status_path = str(tmp_path / "nc_status.json")
    nc_result = run_nc(nc_cfg, nc_data, torch.device("cpu"), logger)
    assert "val_acc" in nc_result and (tmp_path / "nc.pt").is_file()
    assert int(nc_data.y[nc_data.test_idx].min()) >= 0
    nc_checkpoint = torch.load(tmp_path / "nc.pt", map_location="cpu", weights_only=False)
    write_json_atomic(
        nc_cfg.task.validation_metrics_path,
        {"metrics": {key: value for key, value in nc_checkpoint["metrics"].items() if key.startswith("val_")}},
    )
    write_json_atomic(
        nc_cfg.task.paper_run_status_path,
        {"status": "validation_complete", "test_evaluation_count": 0},
    )
    nc_final = evaluate_nc_checkpoint(nc_cfg, nc_data, torch.device("cpu"), logger)
    assert "test_acc" in nc_final
    assert __import__("json").loads((tmp_path / "nc_final.json").read_text())["test_evaluation_count"] == 1
    nc_restored = build_model(
        nc_cfg,
        {"input_dim": 12, "num_nodes": 24, "num_classes": 3, "text_dim": 7, "visual_dim": 5},
    )
    nc_restored.load_semantic_cache(str(nc_cache.path))
    nc_restored.load_state_dict(nc_checkpoint["model_state"])

    lp_cache = ensure_semantic_graph_cache(
        dataset="Smoke", task="lp", text_features=text,
        visual_features=visual, physical_edge_index=lp_data.edge_index,
        config={
            "mode": "auto", "knn_k": 8, "cache_dir": str(tmp_path / "semantic"),
            "validate": True, "faiss_seed": 2026, "faiss_threads": 1,
        },
        project_root=tmp_path,
    )
    lp_cfg = _lp_cfg(lp_cache, tmp_path / "lp.pt")
    lp_cfg.task.evaluate_test = False
    lp_cfg.task.checkpoint_path = str(tmp_path / "lp.pt")
    lp_cfg.task.validation_metrics_path = str(tmp_path / "lp_validation.json")
    lp_cfg.task.final_test_metrics_path = str(tmp_path / "lp_final.json")
    lp_cfg.task.paper_run_status_path = str(tmp_path / "lp_status.json")
    lp_result = run_lp(
        lp_cfg,
        lp_data,
        torch.device("cpu"),
        logger,
    )
    assert "val_mrr" in lp_result and (tmp_path / "lp.pt").is_file()
    lp_checkpoint = torch.load(tmp_path / "lp.pt", map_location="cpu", weights_only=False)
    write_json_atomic(
        lp_cfg.task.validation_metrics_path,
        {"metrics": {key: value for key, value in lp_checkpoint["metrics"].items() if key.startswith("val_")}},
    )
    write_json_atomic(
        lp_cfg.task.paper_run_status_path,
        {"status": "validation_complete", "test_evaluation_count": 0},
    )
    lp_final = evaluate_lp_checkpoint(lp_cfg, lp_data, torch.device("cpu"), logger)
    assert "test_mrr" in lp_final
    assert __import__("json").loads((tmp_path / "lp_final.json").read_text())["test_evaluation_count"] == 1
    lp_restored = build_model(
        lp_cfg,
        {"input_dim": 12, "num_nodes": 24, "num_classes": None, "text_dim": 7, "visual_dim": 5},
    )
    lp_restored.load_semantic_cache(str(lp_cache.path))
    lp_restored.load_state_dict(lp_checkpoint["model_state"])

    baseline_result = run_nc(
        _nc_cfg("gcn"), nc_data, torch.device("cpu"), logger
    )
    assert "val_acc" in baseline_result
