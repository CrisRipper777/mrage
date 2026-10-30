from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.data.semantic_candidates_v7b import build_semantic_candidate_cache_v7b
from src.data.semantic_graph import load_semantic_graph
from src.data.semantic_graph_manager import ensure_semantic_graph_cache
from src.models.factory import build_model
from src.models.ops.semantic_operators import build_modality_knn_operators


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(701)
    text = torch.randn(32, 7, generator=generator)
    visual = torch.randn(32, 5, generator=generator)
    nodes = torch.arange(32)
    ring = torch.stack((nodes, (nodes + 1) % 32))
    skip = torch.stack((nodes, (nodes + 5) % 32))
    physical = torch.cat((ring, skip, ring.flip(0), skip.flip(0)), dim=1).contiguous()
    return text, visual, physical


def _config(root: Path, mode: str = "auto") -> dict:
    return {
        "mode": mode,
        "knn_k": 8,
        "cache_dir": str(root),
        "validate": True,
        "faiss_seed": 2026,
        "faiss_threads": 1,
    }


def test_cache_miss_hit_and_frozen_v7b_tensor_equivalence(tmp_path: Path) -> None:
    text, visual, physical = _inputs()
    config = _config(tmp_path / "managed")
    miss = ensure_semantic_graph_cache(
        dataset="Synthetic", task="nc", text_features=text,
        visual_features=visual, physical_edge_index=physical,
        config=config, project_root=tmp_path,
    )
    hit = ensure_semantic_graph_cache(
        dataset="Synthetic", task="nc", text_features=text,
        visual_features=visual, physical_edge_index=physical,
        config=config, project_root=tmp_path,
    )
    assert miss.cache_hit is False
    assert hit.cache_hit is True
    assert miss.path == hit.path

    raw_path, raw_metadata = build_semantic_candidate_cache_v7b(
        dataset="Synthetic", text_features=text, visual_features=visual,
        physical_edge_index=physical, cache_dir=tmp_path / "raw_v7b",
        top_k=8, seed=2026, thread_count=1,
    )
    managed_graph = load_semantic_graph(
        miss.path,
        expected_dataset="Synthetic",
        expected_identity=miss.identity,
        expected_identity_fingerprint=miss.identity_fingerprint,
        validate=True,
    )
    raw_graph = torch.load(raw_path, map_location="cpu", weights_only=False)
    for key in ("edge_index", "unique_edge_index", "degree", "source_flags"):
        assert torch.equal(managed_graph[key], raw_graph[key]), key
    for key in ("candidate_fingerprint", "v7b_fingerprint"):
        assert managed_graph["metadata"][key] == raw_metadata[key]

    expected_operators = build_modality_knn_operators(
        raw_graph["unique_edge_index"], raw_graph["source_flags"], 32
    )
    for modality in ("text", "visual"):
        for key in ("edge_index", "weight"):
            assert torch.equal(
                managed_graph["operators"][modality][key], expected_operators[modality][key]
            ), f"{modality}.{key}"


def test_cache_identity_isolated_by_task_features_and_physical_graph(tmp_path: Path) -> None:
    text, visual, physical = _inputs()
    config = _config(tmp_path / "managed")

    nc = ensure_semantic_graph_cache(
        dataset="Synthetic", task="nc", text_features=text,
        visual_features=visual, physical_edge_index=physical,
        config=config, project_root=tmp_path,
    )
    lp = ensure_semantic_graph_cache(
        dataset="Synthetic", task="lp", text_features=text,
        visual_features=visual, physical_edge_index=physical,
        config=config, project_root=tmp_path,
    )
    changed_text = text.clone()
    changed_text[0, 0] += 0.125
    changed_features = ensure_semantic_graph_cache(
        dataset="Synthetic", task="nc", text_features=changed_text,
        visual_features=visual, physical_edge_index=physical,
        config=config, project_root=tmp_path,
    )
    changed_physical = torch.cat((physical, torch.tensor([[0], [11]])), dim=1)
    changed_topology = ensure_semantic_graph_cache(
        dataset="Synthetic", task="nc", text_features=text,
        visual_features=visual, physical_edge_index=changed_physical,
        config=config, project_root=tmp_path,
    )

    identities = {
        nc.identity_fingerprint,
        lp.identity_fingerprint,
        changed_features.identity_fingerprint,
        changed_topology.identity_fingerprint,
    }
    assert len(identities) == 4
    assert len({nc.path, lp.path, changed_features.path, changed_topology.path}) == 4


def test_corrupt_auto_cache_is_quarantined_and_existing_mode_fails(tmp_path: Path) -> None:
    text, visual, physical = _inputs()
    root = tmp_path / "managed"
    config = _config(root)
    initial = ensure_semantic_graph_cache(
        dataset="Synthetic", task="nc", text_features=text,
        visual_features=visual, physical_edge_index=physical,
        config=config, project_root=tmp_path,
    )
    initial.path.write_bytes(b"not a torch cache")

    repaired = ensure_semantic_graph_cache(
        dataset="Synthetic", task="nc", text_features=text,
        visual_features=visual, physical_edge_index=physical,
        config=config, project_root=tmp_path,
    )
    assert repaired.cache_hit is False
    assert repaired.path.is_file()
    assert list(repaired.path.parent.glob("semantic_graph.corrupt.*.pt"))

    missing = _config(root, mode="existing")
    with pytest.raises(FileNotFoundError):
        ensure_semantic_graph_cache(
            dataset="Synthetic", task="nc", text_features=text + 1,
            visual_features=visual, physical_edge_index=physical,
            config=missing, project_root=tmp_path,
        )


def test_checkpoint_load_checks_attached_semantic_identity(tmp_path: Path) -> None:
    text, visual, physical = _inputs()
    changed_physical = torch.cat((physical, torch.tensor([[1], [13]])), dim=1)
    config = _config(tmp_path / "managed")
    original_cache = ensure_semantic_graph_cache(
        dataset="Synthetic", task="nc", text_features=text,
        visual_features=visual, physical_edge_index=physical,
        config=config, project_root=tmp_path,
    )
    other_cache = ensure_semantic_graph_cache(
        dataset="Synthetic", task="nc", text_features=text,
        visual_features=visual, physical_edge_index=changed_physical,
        config=config, project_root=tmp_path,
    )

    cfg = OmegaConf.create(
        {
            "model": {
                "name": "mrage", "hidden_dim": 256, "physical_hops": 3,
                "semantic_hops": 1, "num_experts": 4, "top_k": 2,
                    "expert_bottleneck": 64, "dropout": 0.0,
                "late_cross_dropout": 0.0, "fusion_dropout": 0.0,
                "balance_weight": 0.01, "rho_init": 0.25,
                "gamma_init": 0.08, "beta_init": 0.25,
                "same_node_enabled": True, "eps": 1.0e-8,
                "physical_chunk_size": 128, "semantic_chunk_size": 128,
                    "cross_edge_chunk_size": 64, "attention_dim": 32,
                    "router_hidden_dim": 256, "router_embed_dim": 8,
                    "channel_gate_hidden": 128, "expert_eta": 0.1,
            },
            "task": {"name": "nc"},
        }
    )
    model_info = {
        "input_dim": 12, "num_nodes": 32, "num_classes": 3,
        "text_dim": 7, "visual_dim": 5,
    }
    model_a = build_model(cfg, model_info)
    model_a.load_semantic_cache(str(original_cache.path))
    state = copy.deepcopy(model_a.state_dict())
    model_b = build_model(cfg, model_info)
    model_b.load_semantic_cache(str(other_cache.path))

    with pytest.raises(RuntimeError, match="task/topology identity"):
        model_b.load_state_dict(state)
    model_b.load_state_dict(model_b.state_dict())
