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
from src.models.mrage_capability_ablation import Model as CapabilityModel


VARIANTS = (
    "physical_only",
    "semantic_only",
    "no_cross_modal_collaboration",
    "feature_only",
)


@pytest.fixture(scope="module")
def synthetic_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("mrage-capability")
    generator = torch.Generator().manual_seed(3107)
    text = torch.randn(32, 7, generator=generator)
    visual = torch.randn(32, 5, generator=generator)
    x = torch.cat((text, visual), dim=1)
    nodes = torch.arange(32)
    ring = torch.stack((nodes, (nodes + 1) % 32))
    skip = torch.stack((nodes, (nodes + 5) % 32))
    edge_index = torch.cat((ring, skip, ring.flip(0), skip.flip(0)), dim=1).contiguous()
    cache = ensure_semantic_graph_cache(
        dataset="CapabilitySynthetic",
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


def _config(variant: str = "physical_only", model_name: str = "mrage_capability_ablation",
            *, same_node_enabled: bool = True, task_name: str = "nc"):
    return OmegaConf.create(
        {
            "seed": 42,
            "dataset": {"name": "CapabilitySynthetic"},
            "task": {"name": task_name},
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
                "same_node_enabled": same_node_enabled,
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


def _model(variant: str, synthetic_case, *, same_node_enabled: bool = True,
           model_name: str = "mrage_capability_ablation"):
    model = build_model(
        _config(variant, model_name, same_node_enabled=same_node_enabled),
        {
            "input_dim": 12, "num_nodes": 32, "num_classes": 3,
            "text_dim": 7, "visual_dim": 5,
        },
    )
    model.load_semantic_cache(str(synthetic_case["cache"].path))
    return model


def _capture_filter_calls(model):
    original = model._filter
    calls = []

    def wrapped(*args):
        response, balance = original(*args)
        calls.append(
            (int(args[3]), int(args[4]), response.detach().clone(), balance.detach().clone())
        )
        return response, balance

    model._filter = wrapped
    return calls


def _capture_late_states(model):
    original = model._late
    private_by_source = {}
    collaborative_by_target = {}

    def wrapped(target, target_id, source, h_source, private_source, collaborative_target):
        private_by_source[source] = private_source.detach().clone()
        collaborative_by_target[target] = collaborative_target.detach().clone()
        return original(
            target, target_id, source, h_source, private_source, collaborative_target
        )

    model._late = wrapped
    return private_by_source, collaborative_by_target


def _assert_late_states_equal(private_by_source, collaborative_by_target):
    assert set(private_by_source) == {"text", "visual"}
    assert set(collaborative_by_target) == {"text", "visual"}
    for modality in ("text", "visual"):
        assert torch.allclose(
            private_by_source[modality], collaborative_by_target[modality],
            atol=1.0e-6, rtol=1.0e-6,
        ), modality


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_forward_backward_optimizer_step_and_checkpoint(variant, synthetic_case):
    model = _model(variant, synthetic_case)
    model.train()
    classifier = nn.Linear(model.out_dim, 3)
    parameters = [p for p in model.parameters() if p.requires_grad] + list(classifier.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1.0e-3)
    z, _, _, aux_loss, info = model(
        synthetic_case["x"], synthetic_case["edge_index"]
    )
    assert z.shape == (32, model.hidden_dim)
    assert set(info["unimodal_logits"]) == {"text", "visual"}
    assert all(value.shape == (32, 3) for value in info["unimodal_logits"].values())
    labels = synthetic_case["labels"]
    loss = F.cross_entropy(classifier(z), labels) + aux_loss
    for logits in info["unimodal_logits"].values():
        loss = loss + 0.15 * F.cross_entropy(logits, labels)
    assert torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    optimizer.step()

    model.eval()
    with torch.no_grad():
        expected = model(synthetic_case["x"], synthetic_case["edge_index"])[0]
    checkpoint = synthetic_case["root"] / f"{variant}.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = _model(variant, synthetic_case)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    restored.eval()
    with torch.no_grad():
        actual = restored(synthetic_case["x"], synthetic_case["edge_index"])[0]
    assert torch.allclose(expected, actual, atol=1.0e-6, rtol=1.0e-6)


def test_invalid_variant_and_lp_are_rejected(synthetic_case):
    info = {
        "input_dim": 12, "num_nodes": 32, "num_classes": 3,
        "text_dim": 7, "visual_dim": 5,
    }
    invalid = _config("not_a_variant")
    with pytest.raises(ValueError, match="ablation_variant"):
        build_model(invalid, info)
    lp = _config("physical_only", task_name="lp")
    with pytest.raises(ValueError, match="supports NC only.*LP is not implemented"):
        build_model(lp, info)


def test_physical_only_blocks_semantics_and_uses_same_node_or_none(monkeypatch, synthetic_case):
    model = _model("physical_only", synthetic_case)
    model.eval()
    calls = _capture_filter_calls(model)

    import src.models.mrage as mrage_module

    def unexpected_attention(*_args, **_kwargs):
        raise AssertionError("physical_only must not run Semantic sparse attention")

    monkeypatch.setattr(mrage_module, "receiver_sparse_attention", unexpected_attention)
    with torch.no_grad():
        z_before = model(synthetic_case["x"], synthetic_case["edge_index"])[0]
    semantic_calls = [entry for entry in calls if entry[0] == 1]
    physical_calls = [entry for entry in calls if entry[0] == 0]
    assert len(calls) == 6 and len(semantic_calls) == 2 and len(physical_calls) == 4
    assert all(torch.count_nonzero(entry[2]) == 0 and entry[3].item() == 0 for entry in semantic_calls)
    assert all(torch.isfinite(entry[3]) for entry in calls)
    assert any(torch.count_nonzero(entry[2]) > 0 for entry in physical_calls)

    # The encoder's prediction cannot depend on Semantic KNN operators here.
    model.semantic_text_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    model.semantic_text_weight = torch.ones(3)
    model.semantic_visual_edge_index = torch.tensor([[3, 4], [4, 3]])
    model.semantic_visual_weight = torch.ones(2)
    with torch.no_grad():
        z_after = model(synthetic_case["x"], synthetic_case["edge_index"])[0]
        z_no_physical = model(synthetic_case["x"], torch.empty((2, 0), dtype=torch.long))[0]
    assert torch.allclose(z_before, z_after, atol=1.0e-6, rtol=1.0e-6)
    assert not torch.allclose(z_before, z_no_physical, atol=1.0e-7, rtol=1.0e-7)

    no_same_node = _model("physical_only", synthetic_case, same_node_enabled=False)
    no_same_node.eval()
    with torch.no_grad():
        z_none_only = no_same_node(synthetic_case["x"], synthetic_case["edge_index"])[0]
    assert torch.isfinite(z_none_only).all()


def test_semantic_only_is_independent_of_physical_edges_and_keeps_semantics(synthetic_case):
    model = _model("semantic_only", synthetic_case)
    model.eval()
    calls = _capture_filter_calls(model)
    private_states, collaborative_states = _capture_late_states(model)
    with torch.no_grad():
        z_graph = model(synthetic_case["x"], synthetic_case["edge_index"])[0]
    physical_calls = [entry for entry in calls if entry[0] == 0]
    semantic_calls = [entry for entry in calls if entry[0] == 1]
    assert len(physical_calls) == 4 and len(semantic_calls) == 2
    assert all(torch.count_nonzero(entry[2]) == 0 and entry[3].item() == 0 for entry in physical_calls)
    assert all(torch.count_nonzero(entry[2]) > 0 for entry in semantic_calls)
    _assert_late_states_equal(private_states, collaborative_states)
    assert model.semantic_text_edge_index.numel() > 0
    assert model.semantic_visual_edge_index.numel() > 0

    # Same features and attached V7B.1 cache, but substitute the physical graph
    # only at the encoder call. No physical message may reach the output.
    with torch.no_grad():
        z_empty = model(synthetic_case["x"], torch.empty((2, 0), dtype=torch.long))[0]
        z_changed = model(
            synthetic_case["x"], synthetic_case["edge_index"][:, ::3].contiguous()
        )[0]
    assert torch.allclose(z_graph, z_empty, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(z_graph, z_changed, atol=1.0e-6, rtol=1.0e-6)


def test_no_cross_modal_collaboration_keeps_private_graph_and_final_fusion(monkeypatch, synthetic_case):
    model = _model("no_cross_modal_collaboration", synthetic_case)
    model.eval()
    calls = _capture_filter_calls(model)
    private_states, collaborative_states = _capture_late_states(model)

    import src.models.mrage as mrage_module

    def unexpected_attention(*_args, **_kwargs):
        raise AssertionError("cross-off must bypass Late Semantic attention")

    monkeypatch.setattr(mrage_module, "receiver_sparse_attention", unexpected_attention)
    cross = model._early_cross(
        torch.randn(32, model.hidden_dim), nn.Identity(),
        synthetic_case["edge_index"], torch.ones(synthetic_case["edge_index"].size(1)),
        torch.ones(32, dtype=torch.bool), torch.randn(32, model.hidden_dim),
    )
    assert torch.count_nonzero(cross) == 0
    with torch.no_grad():
        z, _, _, _, info = model(synthetic_case["x"], synthetic_case["edge_index"])
    assert len(calls) == 6
    assert all(torch.count_nonzero(entry[2]) > 0 for entry in calls)
    _assert_late_states_equal(private_states, collaborative_states)
    assert z.shape == (32, model.hidden_dim)
    assert info["unimodal_logits"]["text"].shape == (32, 3)
    changed_x = synthetic_case["x"].clone()
    changed_x[:, :7] += 0.25
    with torch.no_grad():
        assert not torch.allclose(z, model(changed_x, synthetic_case["edge_index"])[0])


def test_feature_only_skips_all_graph_paths_and_retains_feature_modules(monkeypatch, synthetic_case):
    model = _model("feature_only", synthetic_case)
    model.eval()
    import src.models.mrage as mrage_module

    def unexpected_graph_call(*_args, **_kwargs):
        raise AssertionError("feature_only called a graph propagation/cross operator")

    monkeypatch.setattr(mrage_module, "chunked_weighted_spmm", unexpected_graph_call)
    monkeypatch.setattr(mrage_module, "receiver_sparse_attention", unexpected_graph_call)
    model._filter = unexpected_graph_call
    model._early_cross = unexpected_graph_call
    model._late = unexpected_graph_call
    model._physical_operator = unexpected_graph_call
    model._basis = unexpected_graph_call

    with torch.no_grad():
        z_a, _, _, aux_a, info_a = model(synthetic_case["x"], synthetic_case["edge_index"])
        z_b, _, _, aux_b, info_b = model(
            synthetic_case["x"], torch.empty((2, 0), dtype=torch.long)
        )
        z_none, _, _, _, _ = model(synthetic_case["x"], None)
    assert torch.allclose(z_a, z_b, atol=1.0e-7, rtol=1.0e-7)
    assert torch.allclose(z_a, z_none, atol=1.0e-7, rtol=1.0e-7)
    assert aux_a.item() == 0.0 and aux_b.item() == 0.0
    assert info_a["balance_loss"].item() == 0.0
    assert info_b["balance_loss"].item() == 0.0
    assert set(info_a["unimodal_logits"]) == {"text", "visual"}
    assert all(p.requires_grad for p in model.projectors.parameters())
    assert all(p.requires_grad for p in model.unimodal_heads.parameters())
    assert all(p.requires_grad for p in model.fusion_skip.parameters())
    assert all(p.requires_grad for p in model.fusion_linear1.parameters())
    assert all(p.requires_grad for p in model.fusion_linear2.parameters())
    assert all(p.requires_grad for p in model.fusion_norm.parameters())
    assert not any(p.requires_grad for p in model.filter_bank.parameters())
    assert not any(p.requires_grad for p in model.router.parameters())
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert all(
        name.startswith(("projectors.", "unimodal_heads.", "fusion_skip.", "fusion_linear1.", "fusion_linear2.", "fusion_norm."))
        for name in trainable
    )

    # Semantic operators are not read in this path, even when replaced.
    model.semantic_text_edge_index = torch.tensor([[0, 1], [1, 0]])
    model.semantic_text_weight = torch.ones(2)
    model.semantic_visual_edge_index = torch.tensor([[2, 3], [3, 2]])
    model.semantic_visual_weight = torch.ones(2)
    with torch.no_grad():
        z_sem_changed = model(synthetic_case["x"], synthetic_case["edge_index"])[0]
    assert torch.allclose(z_a, z_sem_changed, atol=1.0e-7, rtol=1.0e-7)
    changed_text = synthetic_case["x"].clone()
    changed_text[:, :7] += 0.2
    with torch.no_grad():
        z_feature_changed = model(changed_text, synthetic_case["edge_index"])[0]
    assert not torch.allclose(z_a, z_feature_changed)


def test_internal_full_mode_matches_formal_mrage(synthetic_case):
    full = _model("full", synthetic_case)
    formal = _model("full", synthetic_case, model_name="mrage")
    full.load_state_dict(copy.deepcopy(formal.state_dict()))
    full.eval()
    formal.eval()
    with torch.no_grad():
        expected = formal(synthetic_case["x"], synthetic_case["edge_index"])
        actual = full(synthetic_case["x"], synthetic_case["edge_index"])
    assert torch.allclose(expected[0], actual[0], atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(expected[3], actual[3], atol=1.0e-7, rtol=1.0e-7)
    assert torch.allclose(
        expected[4]["balance_loss"], actual[4]["balance_loss"], atol=1.0e-7, rtol=1.0e-7
    )
