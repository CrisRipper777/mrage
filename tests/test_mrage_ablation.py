from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.data.semantic_graph_manager import ensure_semantic_graph_cache
from src.models.factory import build_model
from src.models.mrage import Model as MRAGEModel
from src.models.mrage_ablation import Model as AblationModel


VARIANTS = (
    "single_filter",
    "unshared_experts",
    "no_semantic_propagation",
    "no_topological_collaboration",
    "no_semantic_refinement",
)


@pytest.fixture(scope="module")
def synthetic_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("mrage-ablation")
    generator = torch.Generator().manual_seed(9361)
    text = torch.randn(32, 7, generator=generator)
    visual = torch.randn(32, 5, generator=generator)
    x = torch.cat((text, visual), dim=1)
    nodes = torch.arange(32)
    ring = torch.stack((nodes, (nodes + 1) % 32))
    skip = torch.stack((nodes, (nodes + 5) % 32))
    edge_index = torch.cat((ring, skip, ring.flip(0), skip.flip(0)), dim=1).contiguous()
    cache = ensure_semantic_graph_cache(
        dataset="SyntheticAblation",
        task="nc",
        text_features=text,
        visual_features=visual,
        physical_edge_index=edge_index,
        config={
            "mode": "auto", "knn_k": 8, "cache_dir": str(root / "semantic"),
            "validate": True, "faiss_seed": 2026, "faiss_threads": 1,
        },
        project_root=root,
    )
    return {
        "root": root,
        "text": text,
        "visual": visual,
        "x": x,
        "edge_index": edge_index,
        "labels": nodes % 3,
        "cache": cache,
    }


def _config(variant: str = "single_filter", model_name: str = "mrage_ablation"):
    return OmegaConf.create(
        {
            "seed": 42,
            "dataset": {"name": "SyntheticAblation"},
            "task": {"name": "nc"},
            "model": {
                "name": model_name,
                "ablation_variant": variant,
                "hidden_dim": 32,
                "physical_hops": 3,
                "semantic_hops": 1,
                "num_experts": 4,
                "top_k": 2,
                "expert_bottleneck": 16,
                "expert_eta": 0.1,
                "dropout": 0.0,
                "late_cross_dropout": 0.0,
                "fusion_dropout": 0.0,
                "balance_weight": 0.01,
                "rho_init": 0.25,
                "gamma_init": 0.08,
                "beta_init": 0.25,
                "same_node_enabled": True,
                "semantic_knn_k": 8,
                "nc_unimodal_weight": 0.3,
                "eps": 1.0e-8,
                "physical_chunk_size": 128,
                "semantic_chunk_size": 128,
                "cross_edge_chunk_size": 64,
                "attention_dim": 8,
                "router_hidden_dim": 32,
                "router_embed_dim": 4,
                "channel_gate_hidden": 16,
            },
        }
    )


def _new_model(variant: str, synthetic_case, model_name: str = "mrage_ablation"):
    cfg = _config(variant, model_name)
    info = {
        "input_dim": 12,
        "num_nodes": 32,
        "num_classes": 3,
        "text_dim": 7,
        "visual_dim": 5,
    }
    model = build_model(cfg, info)
    model.load_semantic_cache(str(synthetic_case["cache"].path))
    return model


def test_all_variants_instantiate_and_receive_legal_cache(synthetic_case):
    for variant in VARIANTS:
        model = _new_model(variant, synthetic_case)
        assert isinstance(model, AblationModel)
        assert model.semantic_candidate_fingerprint == synthetic_case["cache"].metadata[
            "candidate_fingerprint"
        ]
        assert model.semantic_text_edge_index.numel() > 0
        assert model.semantic_visual_edge_index.numel() > 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_variant_forward_finite_loss_and_backward(variant, synthetic_case):
    model = _new_model(variant, synthetic_case)
    classifier = nn.Linear(model.out_dim, 3)
    model.train()
    z, _, _, aux_loss, info = model(
        synthetic_case["x"], synthetic_case["edge_index"]
    )
    labels = synthetic_case["labels"]
    loss = F.cross_entropy(classifier(z), labels) + aux_loss
    for logits in info["unimodal_logits"].values():
        loss = loss + 0.3 * F.cross_entropy(logits, labels)
    assert torch.isfinite(loss)
    assert set(info["unimodal_logits"]) == {"text", "visual"}
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_single_filter_has_no_router_and_zero_balance(synthetic_case):
    model = _new_model("single_filter", synthetic_case)
    assert model.filter_bank is None
    assert model.router is None
    assert hasattr(model, "single_filter")
    assert model.single_filter.theta.numel() == 4
    assert not any(name.startswith("router.") for name, _ in model.named_parameters())
    model.eval()
    _, _, _, aux_loss, info = model(
        synthetic_case["x"], synthetic_case["edge_index"]
    )
    assert info["balance_loss"].item() == 0.0
    assert aux_loss.item() == 0.0


def test_unshared_pools_are_distinct_and_collaborative_reuses_physical_pool(synthetic_case):
    model = _new_model("unshared_experts", synthetic_case)
    keys = ("text_physical", "text_semantic", "visual_physical", "visual_semantic")
    pools = [model.filter_banks[key] for key in keys]
    parameters = [pool.theta for pool in pools]
    assert len({id(parameter) for parameter in parameters}) == 4
    assert model._bank_for(0, 0) is model.filter_banks["text_physical"]
    assert model._bank_for(0, 1) is model.filter_banks["text_semantic"]
    assert model._bank_for(1, 0) is model.filter_banks["visual_physical"]
    assert model._bank_for(1, 1) is model.filter_banks["visual_semantic"]

    calls = {key: 0 for key in keys}
    hooks = [
        model.filter_banks[key].experts[0].register_forward_hook(
            lambda _module, _inputs, _output, pool=key: calls.__setitem__(pool, calls[pool] + 1)
        )
        for key in keys
    ]
    model.train()
    z, _, _, aux_loss, info = model(
        synthetic_case["x"], synthetic_case["edge_index"]
    )
    assert calls == {
        "text_physical": 2, "text_semantic": 1,
        "visual_physical": 2, "visual_semantic": 1,
    }
    loss = z.square().mean() + aux_loss
    for logits in info["unimodal_logits"].values():
        loss = loss + logits.square().mean()
    loss.backward()
    for hook in hooks:
        hook.remove()
    for key in keys:
        grad = model.filter_banks[key].theta.grad
        assert grad is not None and torch.isfinite(grad).all(), key


def test_no_semantic_propagation_keeps_late_semantic_knn(monkeypatch, synthetic_case):
    model = _new_model("no_semantic_propagation", synthetic_case)
    model.eval()
    basis = [torch.randn(32, model.hidden_dim), torch.randn(32, model.hidden_dim)]
    support = torch.ones(32, dtype=torch.bool)
    current = torch.randn(32, model.hidden_dim)
    response, balance = model._filter(
        basis, support, current, 1, 0, torch.zeros_like(current)
    )
    assert torch.count_nonzero(response) == 0
    assert balance.item() == 0.0
    assert model.semantic_visual_edge_index.numel() > 0

    calls = []
    original_filter = model._filter

    def record_filter(*args):
        result = original_filter(*args)
        calls.append((int(args[3]), result[0], result[1]))
        return result

    model._filter = record_filter
    with torch.no_grad():
        _, _, _, _, info = model(synthetic_case["x"], synthetic_case["edge_index"])
    semantic_calls = [entry for entry in calls if entry[0] == 1]
    assert len(calls) == 6
    assert len(semantic_calls) == 2
    assert all(torch.count_nonzero(entry[1]) == 0 and entry[2].item() == 0.0 for entry in semantic_calls)
    assert torch.allclose(
        info["balance_loss"], torch.stack([entry[2] for entry in calls]).mean()
    )

    import src.models.mrage as mrage_module

    seen_edges = []
    original = mrage_module.receiver_sparse_attention

    def capture_edges(query, key, value, edge_index, chunk_size):
        seen_edges.append(edge_index.detach().cpu().clone())
        return original(query, key, value, edge_index, chunk_size)

    monkeypatch.setattr(mrage_module, "receiver_sparse_attention", capture_edges)
    with torch.no_grad():
        model._late(
            "text", 0, "visual", torch.randn(32, model.hidden_dim),
            torch.randn(32, model.hidden_dim), torch.randn(32, model.hidden_dim),
        )
    assert len(seen_edges) == 1
    assert torch.equal(seen_edges[0], model.semantic_visual_edge_index)


def test_no_topological_collaboration_returns_zero_context_and_fallback(synthetic_case):
    model = _new_model("no_topological_collaboration", synthetic_case)
    reference = torch.randn(32, model.hidden_dim)
    zeros = model._early_cross(
        torch.randn(32, model.hidden_dim), nn.Identity(),
        synthetic_case["edge_index"], torch.ones(synthetic_case["edge_index"].size(1)),
        torch.ones(32, dtype=torch.bool), reference,
    )
    assert torch.count_nonzero(zeros) == 0
    private = torch.randn_like(reference)
    candidate = torch.randn_like(reference)
    assert torch.equal(model._zero_context_fallback(private, candidate, zeros), private)


def test_no_semantic_refinement_returns_collaborative_representation(synthetic_case):
    model = _new_model("no_semantic_refinement", synthetic_case)
    collaborative = torch.randn(32, model.hidden_dim)
    result = model._late(
        "text", 0, "visual", torch.randn_like(collaborative),
        torch.randn_like(collaborative), collaborative,
    )
    assert result is collaborative
    assert torch.equal(result, collaborative)


def test_ablation_checkpoint_save_and_reload(synthetic_case):
    model = _new_model("single_filter", synthetic_case)
    model.eval()
    with torch.no_grad():
        expected = model(synthetic_case["x"], synthetic_case["edge_index"])[0]
    checkpoint = synthetic_case["root"] / "single_filter.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = _new_model("single_filter", synthetic_case)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    restored.eval()
    with torch.no_grad():
        actual = restored(synthetic_case["x"], synthetic_case["edge_index"])[0]
    assert torch.allclose(expected, actual, atol=1.0e-6, rtol=1.0e-6)


def test_internal_full_mode_matches_formal_mrage(synthetic_case):
    base = _new_model("full", synthetic_case, model_name="mrage")
    ablation = _new_model("full", synthetic_case)
    ablation.load_state_dict(copy.deepcopy(base.state_dict()))
    base.eval()
    ablation.eval()
    with torch.no_grad():
        expected = base(synthetic_case["x"], synthetic_case["edge_index"])
        actual = ablation(synthetic_case["x"], synthetic_case["edge_index"])
    assert torch.allclose(expected[0], actual[0], atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(expected[3], actual[3], atol=1.0e-7, rtol=1.0e-7)
    assert torch.allclose(
        expected[4]["balance_loss"], actual[4]["balance_loss"], atol=1.0e-7, rtol=1.0e-7
    )
    for modality in ("text", "visual"):
        assert torch.allclose(
            expected[4]["unimodal_logits"][modality],
            actual[4]["unimodal_logits"][modality],
            atol=1.0e-6,
            rtol=1.0e-6,
        )
