from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from src.data.semantic_graph_manager import ensure_semantic_graph_cache
from src.models.factory import build_model
from src.models.mrage_lp_ablation import Model


VARIANTS = (
    "single_filter",
    "physical_only",
    "semantic_only",
    "no_cross_modal_collaboration",
    "feature_only",
)


@pytest.fixture(scope="module")
def synthetic_case(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("mrage-sports-lp-ablation")
    generator = torch.Generator().manual_seed(6813)
    n = 40
    text = torch.randn(n, 7, generator=generator)
    visual = torch.randn(n, 5, generator=generator)
    x = torch.cat((text, visual), dim=-1)
    nodes = torch.arange(n)
    ring = torch.stack((nodes, (nodes + 1) % n))
    skip = torch.stack((nodes, (nodes + 4) % n))
    edge_index = torch.cat((ring, skip, ring.flip(0), skip.flip(0)), dim=1).contiguous()
    cache = ensure_semantic_graph_cache(
        dataset="SyntheticSportsLP",
        task="lp",
        text_features=text,
        visual_features=visual,
        physical_edge_index=edge_index,
        config={
            "mode": "auto", "knn_k": 8, "cache_dir": str(root / "semantic"),
            "validate": True, "faiss_seed": 2026, "faiss_threads": 1,
        },
        project_root=root,
    )
    return {"root": root, "x": x, "edge_index": edge_index, "cache": cache}


def _config(variant: str, task_name: str = "lp"):
    return OmegaConf.create({
        "seed": 42,
        "dataset": {"name": "SyntheticSportsLP"},
        "task": {"name": task_name},
        "model": {
            "name": "mrage_lp_ablation", "ablation_variant": variant,
            "hidden_dim": 32, "physical_hops": 3, "semantic_hops": 1,
            "num_experts": 4, "top_k": 2, "expert_bottleneck": 16,
            "expert_eta": 0.1, "dropout": 0.0, "late_cross_dropout": 0.0,
            "fusion_dropout": 0.0, "balance_weight": 0.01, "rho_init": 0.25,
            "gamma_init": 0.08, "beta_init": 0.25, "same_node_enabled": True,
            "semantic_knn_k": 8, "nc_unimodal_weight": 0.3, "eps": 1.0e-8,
            "physical_chunk_size": 128, "semantic_chunk_size": 128,
            "cross_edge_chunk_size": 64, "attention_dim": 8,
            "router_hidden_dim": 32, "router_embed_dim": 4,
            "channel_gate_hidden": 16,
            "semantic_context_physical_fanout": [1, 1, 1],
        },
    })


def _model(variant, synthetic_case):
    model = build_model(
        _config(variant),
        {"input_dim": 12, "num_nodes": 40, "num_classes": None,
         "text_dim": 7, "visual_dim": 5},
    )
    model.load_semantic_cache(str(synthetic_case["cache"].path))
    return model


def _sampled_batch(case):
    loader = LinkNeighborLoader(
        Data(x=case["x"], edge_index=case["edge_index"]),
        edge_label_index=torch.tensor([[0, 1, 0, 9], [1, 2, 12, 19]]),
        edge_label=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        num_neighbors=[2, 2, 2], batch_size=4, shuffle=False,
        subgraph_type="bidirectional", num_workers=0,
    )
    return next(iter(loader))


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_run_sampled_lp_forward_backward_and_optimizer(variant, synthetic_case):
    model = _model(variant, synthetic_case)
    assert model.supports_link_prediction is True
    assert model.requires_global_semantic_candidates is False
    assert model.semantic_candidate_fingerprint == synthetic_case["cache"].metadata[
        "candidate_fingerprint"
    ]
    model.bind_lp_graph(synthetic_case["x"], synthetic_case["edge_index"])
    batch = _sampled_batch(synthetic_case)
    original_n = batch.n_id.numel()
    original_edges = batch.edge_index.clone()
    model.prepare_sampled_lp_batch(batch, sample_seed=42)
    if variant in {"physical_only", "feature_only"}:
        assert batch.n_id.numel() == original_n
        assert torch.equal(batch.edge_index, original_edges)
    if variant == "physical_only":
        assert model._sampled_semantic_edges is not None
        assert all(edges.numel() == 0 for edges, _ in model._sampled_semantic_edges.values())
    if variant == "feature_only":
        assert model._sampled_semantic_edges is None

    model._batch_n_id = batch.n_id
    model.train()
    z, _, _, aux_loss, _ = model(batch.x, batch.edge_index)
    assert z.shape == (batch.num_nodes, model.hidden_dim)
    src, dst = batch.edge_label_index
    logits = (z[src] * z[dst]).sum(dim=-1)
    loss = F.binary_cross_entropy_with_logits(logits, batch.edge_label.float()) + aux_loss
    assert torch.isfinite(loss)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1.0e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    optimizer.step()
    model.clear_sampled_lp_context()


def test_single_filter_reuses_component_ablation_filter(synthetic_case):
    model = _model("single_filter", synthetic_case)
    assert model.filter_bank is None
    assert model.router is None
    assert hasattr(model, "single_filter")
    assert model.single_filter.theta.numel() == 4


def test_lp_ablation_rejects_nc_and_full_is_not_a_formal_variant(synthetic_case):
    info = {"input_dim": 12, "num_nodes": 40, "num_classes": 3,
            "text_dim": 7, "visual_dim": 5}
    with pytest.raises(ValueError, match="supports sampled LP only"):
        build_model(_config("single_filter", task_name="nc"), info)
    with pytest.raises(ValueError, match="ablation_variant"):
        build_model(_config("unknown"), info)
