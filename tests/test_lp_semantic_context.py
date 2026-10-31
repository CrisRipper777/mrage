from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader

from src.data.semantic_graph_manager import ensure_semantic_graph_cache
from src.models.mrage_lp import Model
from src.tasks.lp import _build_message_edge_lookup, _exclude_positive_label_edges_by_global_eid


@pytest.fixture
def context_model(tmp_path):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(1729)
    n = 40
    x = torch.randn(n, 12, generator=generator)
    nodes = torch.arange(n - 1)  # Last node is physically isolated.
    near = torch.stack((nodes, (nodes + 1) % (n - 1)))
    far = torch.stack((nodes, (nodes + 4) % (n - 1)))
    edges = torch.cat((near, far, near.flip(0), far.flip(0), near[:, :3]), dim=1)
    cache = ensure_semantic_graph_cache(
        dataset="Context", task="lp", text_features=x[:, :7], visual_features=x[:, 7:],
        physical_edge_index=edges,
        config={"mode": "auto", "knn_k": 8, "cache_dir": str(tmp_path),
                "validate": True, "faiss_seed": 2026, "faiss_threads": 1},
        project_root=tmp_path,
    )
    cfg = OmegaConf.create({"model": {"hidden_dim": 16, "expert_bottleneck": 4,
                                     "router_hidden_dim": 16, "attention_dim": 4,
                                     "channel_gate_hidden": 8}, "task": {"name": "lp"}})
    model = Model(cfg, {"input_dim": 12, "num_nodes": n, "num_classes": None,
                        "text_dim": 7, "visual_dim": 5})
    model.load_semantic_cache(str(cache.path))
    model.bind_lp_graph(x, edges)
    return model


def _fresh_context(model, nodes, seed):
    """Original Loader path, including feature gathering and iterator RNG draw."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed % (2**31 - 1))
        loader = NeighborLoader(
            Data(x=model._lp_global_x, edge_index=model._lp_global_edge_index),
            input_nodes=nodes, num_neighbors=list(model.semantic_context_physical_fanout),
            batch_size=max(nodes.numel(), 1), shuffle=False,
            subgraph_type="bidirectional", num_workers=0,
        )
        return next(iter(loader)) if nodes.numel() else None


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_cached_sampler_matches_fresh_loader_and_preserves_rng(context_model, seed):
    model = context_model
    sampler = model._lp_context_sampler
    for nodes in (torch.tensor([7, 0, 39, 3]), torch.arange(40), torch.empty(0, dtype=torch.long)):
        rng = torch.random.get_rng_state().clone()
        expected = _fresh_context(model, nodes, seed)
        actual = model._sample_semantic_source_context(nodes, sample_seed=seed)
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert model._lp_context_sampler is sampler
        if expected is None:
            assert actual is None
        else:
            for key in ("n_id", "e_id", "edge_index"):
                assert torch.equal(actual[key], expected[key]), key
            assert actual.x is None  # Context sampling must not gather unused features.


def _reference_expansion(model, batch, seed):
    """Small independent set/list oracle for closure, masking, and edge order."""
    original = batch.n_id.tolist()
    target = set(batch.n_id[batch.edge_label_index].flatten().tolist())
    operators = {}
    for modality in ("text", "visual"):
        edge = getattr(model, f"semantic_{modality}_edge_index")
        order = torch.argsort(edge[1], stable=True)
        operators[modality] = (
            edge[:, order], getattr(model, f"semantic_{modality}_weight")[order], order,
        )
    late = {src for edge, _, _ in operators.values()
            for src, dst in edge.t().tolist() if dst in target}
    semantic_targets = target | late
    semantic_sources = {src for edge, _, _ in operators.values()
                        for src, dst in edge.t().tolist() if dst in semantic_targets}
    extra = _fresh_context(model, torch.tensor(sorted(late), dtype=torch.long), seed)
    new_nodes = set() if extra is None else set(extra.n_id.tolist()) | late | semantic_sources
    combined = original + sorted(new_nodes - set(original))
    mapping = {node: index for index, node in enumerate(combined)}
    positive = batch.n_id[batch.edge_label_index[:, batch.edge_label > .5]].t().tolist()
    forbidden = {(src, dst) for src, dst in positive} | {(dst, src) for src, dst in positive}
    seen = set(batch.e_id.tolist())
    added_ids, added_edges = [], []
    if extra is not None:
        for eid, (src, dst) in zip(extra.e_id.tolist(), extra.n_id[extra.edge_index].t().tolist()):
            global_pair = tuple(model._lp_global_edge_index[:, eid].tolist())
            if eid not in seen and global_pair not in forbidden:
                seen.add(eid)
                added_ids.append(eid)
                added_edges.append([mapping[src], mapping[dst]])
    edges = torch.tensor(added_edges, dtype=torch.long).reshape(-1, 2).t()
    expected = {
        "n_id": torch.tensor(combined), "x": model._lp_global_x[combined],
        "e_id": torch.cat((batch.e_id, torch.tensor(added_ids, dtype=torch.long))),
        "edge_index": torch.cat((batch.edge_index, edges), dim=1),
        "edge_label": batch.edge_label, "edge_label_index": batch.edge_label_index,
    }
    for modality, (edge, weight, _) in operators.items():
        chosen = torch.tensor([i for i, dst in enumerate(edge[1].tolist())
                               if dst in semantic_targets], dtype=torch.long)
        global_edges = edge[:, chosen]
        expected[f"{modality}_edges"] = torch.tensor(
            [[mapping[src], mapping[dst]] for src, dst in global_edges.t().tolist()],
            dtype=torch.long,
        ).reshape(-1, 2).t()
        expected[f"{modality}_weights"] = weight[chosen]
        # Read flags through the canonical undirected candidate order.
        codes = []
        for other_edge, _, _ in operators.values():
            codes.extend(min(s, d) * model.num_nodes + max(s, d) for s, d in other_edge.t().tolist())
        flag_indices = {code: i for i, code in enumerate(sorted(set(codes)))}
        indices = torch.tensor([flag_indices[min(s, d) * model.num_nodes + max(s, d)]
                                for s, d in global_edges.t().tolist()], dtype=torch.long)
        expected[f"{modality}_flags"] = model.semantic_source_flags_v7b[indices]
    return expected


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("empty_semantics", [False, True])
def test_preparation_keeps_full_closure_masking_and_order(context_model, seed, empty_semantics):
    model = context_model
    if empty_semantics:
        for modality in ("text", "visual"):
            setattr(model, f"semantic_{modality}_edge_index", torch.empty((2, 0), dtype=torch.long))
            setattr(model, f"semantic_{modality}_weight", torch.empty(0))
        model.semantic_source_flags_v7b = torch.empty((0, 2), dtype=torch.uint8)
        model.build_semantic_incoming_index()
    torch.manual_seed(seed)
    loader = LinkNeighborLoader(
        Data(x=model._lp_global_x, edge_index=model._lp_global_edge_index),
        edge_label_index=torch.tensor([[0, 1, 0, 39], [1, 2, 12, 19]]),
        edge_label=torch.tensor([1., 1., 0., 0.]), num_neighbors=[2, 2, 2],
        batch_size=4, shuffle=False, subgraph_type="bidirectional", num_workers=0,
    )
    batch = next(iter(loader))
    lookup = _build_message_edge_lookup(model._lp_global_edge_index, model.num_nodes)
    _exclude_positive_label_edges_by_global_eid(batch, lookup)
    expected = _reference_expansion(model, batch, seed)
    rng = torch.random.get_rng_state().clone()
    state_keys = set(model.state_dict())
    model.prepare_sampled_lp_batch(batch, sample_seed=seed)
    assert torch.equal(torch.random.get_rng_state(), rng)
    for key in ("x", "n_id", "e_id", "edge_index", "edge_label", "edge_label_index"):
        assert torch.equal(batch[key], expected[key]), key
    for modality in ("text", "visual"):
        edge, weight = model._sampled_semantic_edges[modality]
        assert torch.equal(edge, expected[f"{modality}_edges"])
        assert torch.equal(weight, expected[f"{modality}_weights"])
        assert torch.equal(model._sampled_semantic_edge_flags[modality], expected[f"{modality}_flags"])
    assert set(model.state_dict()) == state_keys
    model.clear_sampled_lp_context()
    assert model._batch_n_id is None and model._sampled_semantic_edges is None


def test_rebinding_rebuilds_sampler_and_edge_ids(context_model):
    model = context_model
    sampler = model._lp_context_sampler
    graph = model._lp_global_edge_index.flip(1).clone()
    rng = torch.random.get_rng_state().clone()
    model.bind_lp_graph(model._lp_global_x, graph)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert model._lp_context_sampler is not sampler
    nodes = torch.tensor([0, 7, 39])
    expected = _fresh_context(model, nodes, 43)
    actual = model._sample_semantic_source_context(nodes, sample_seed=43)
    for key in ("n_id", "e_id", "edge_index"):
        assert torch.equal(actual[key], expected[key])


def test_duplicate_eids_keep_first_occurrence_in_sampler_order():
    eids = torch.tensor([7, 2, 7, 5, 2, 9, 5])
    assert Model._first_unique_positions(eids).tolist() == [0, 1, 3, 5]
    assert Model._first_unique_positions(eids[:0]).numel() == 0
