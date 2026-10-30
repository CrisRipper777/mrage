from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.data.semantic_graph_manager import ensure_semantic_graph_cache
from src.models import LinkPredictor, build_model


def _inputs():
    generator = torch.Generator().manual_seed(3701)
    n = 24
    text = torch.randn(n, 7, generator=generator)
    visual = torch.randn(n, 5, generator=generator)
    x = torch.cat((text, visual), dim=1)
    nodes = torch.arange(n)
    one = torch.stack((nodes, (nodes + 1) % n))
    two = torch.stack((nodes, (nodes + 4) % n))
    edge_index = torch.cat((one, two, one.flip(0), two.flip(0)), dim=1)
    return n, text, visual, x, edge_index


def _cfg(task: str, hidden_dim: int) -> OmegaConf:
    return OmegaConf.create(
        {
            "dataset": {"name": "Dimensions"},
            "task": {"name": task},
            "model": {
                "name": "mrage", "hidden_dim": hidden_dim,
                "physical_hops": 3, "semantic_hops": 1,
                "num_experts": 4, "top_k": 2, "semantic_knn_k": 8,
                "expert_bottleneck": 8, "expert_eta": 0.1,
                "dropout": 0.0, "late_cross_dropout": 0.0,
                "fusion_dropout": 0.0, "balance_weight": 0.01,
                "rho_init": 0.25, "gamma_init": 0.08, "beta_init": 0.25,
                "same_node_enabled": True, "eps": 1e-8,
                "physical_chunk_size": 32, "semantic_chunk_size": 32,
                "cross_edge_chunk_size": 32, "attention_dim": 8,
                "router_hidden_dim": 16, "router_embed_dim": 4,
                "channel_gate_hidden": 16,
                "semantic_context_physical_fanout": [1, 1, 1],
            },
        }
    )


def test_mrage_hidden_dimensions_train_nc_and_lp_adapters(tmp_path):
    torch.set_num_threads(1)
    n, text, visual, x, edge_index = _inputs()
    cache_args = {
        "text_features": text, "visual_features": visual,
        "config": {
            "mode": "auto", "knn_k": 8,
            "cache_dir": str(tmp_path / "semantic"),
            "validate": True, "faiss_seed": 2026, "faiss_threads": 1,
        },
        "project_root": tmp_path,
    }
    caches = {
        task: ensure_semantic_graph_cache(
            dataset="Dimensions", task=task, physical_edge_index=edge_index,
            **cache_args,
        )
        for task in ("nc", "lp")
    }
    labels = torch.arange(n) % 3
    data_info_nc = {
        "input_dim": 12, "num_nodes": n, "num_classes": 3,
        "text_dim": 7, "visual_dim": 5,
    }
    data_info_lp = {**data_info_nc, "num_classes": None}

    for hidden_dim in (128, 256, 512):
        nc_cfg = _cfg("nc", hidden_dim)
        nc_model = build_model(nc_cfg, data_info_nc)
        nc_model.load_semantic_cache(str(caches["nc"].path))
        classifier = nn.Linear(hidden_dim, 3)
        nc_optimizer = torch.optim.Adam(list(nc_model.parameters()) + list(classifier.parameters()), lr=1e-3)
        nc_model.train()
        z, _, _, aux_loss, _ = nc_model(x, edge_index)
        nc_loss = F.cross_entropy(classifier(z[:18]), labels[:18]) + aux_loss
        assert z.shape == (n, hidden_dim)
        assert torch.isfinite(nc_loss)
        nc_loss.backward()
        nc_optimizer.step()
        assert all(torch.isfinite(parameter).all() for parameter in nc_model.parameters())

        lp_cfg = _cfg("lp", hidden_dim)
        lp_model = build_model(lp_cfg, data_info_lp)
        lp_model.load_semantic_cache(str(caches["lp"].path))
        lp_model.bind_lp_graph(x, edge_index)
        predictor = LinkPredictor(in_dim=hidden_dim, hidden_dim=16, num_layers=1, dropout=0.0)
        lp_optimizer = torch.optim.Adam(list(lp_model.parameters()) + list(predictor.parameters()), lr=1e-3)
        lp_model.train()
        lp_z, _, _, lp_aux, _ = lp_model(x, edge_index)
        source = torch.arange(8)
        target = (source + 1) % n
        logits = predictor.score_pairs(lp_z[source], lp_z[target])
        lp_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)) + lp_aux
        assert lp_z.shape == (n, hidden_dim)
        assert torch.isfinite(lp_loss)
        lp_loss.backward()
        lp_optimizer.step()
        assert all(torch.isfinite(parameter).all() for parameter in lp_model.parameters())


def test_mrage_keeps_expert_router_and_graph_contracts():
    info = {"input_dim": 12, "num_nodes": 24, "num_classes": 3, "text_dim": 7, "visual_dim": 5}
    cfg = _cfg("nc", 128)
    cfg.model.num_experts = 3
    try:
        build_model(cfg, info)
    except ValueError as exc:
        assert "four experts and Top-2" in str(exc)
    else:
        raise AssertionError("MRAGE accepted a non-contract expert count")

    cfg = _cfg("nc", 128)
    cfg.model.physical_hops = 2
    try:
        build_model(cfg, info)
    except ValueError as exc:
        assert "physical_hops=3" in str(exc)
    else:
        raise AssertionError("MRAGE accepted an unsupported graph depth")

    for invalid_dim in (0, -1, 128.5):
        cfg = _cfg("nc", 128)
        cfg.model.hidden_dim = invalid_dim
        try:
            build_model(cfg, info)
        except ValueError as exc:
            assert "hidden_dim must be a positive integer" in str(exc)
        else:
            raise AssertionError(f"MRAGE accepted invalid hidden_dim={invalid_dim}")
