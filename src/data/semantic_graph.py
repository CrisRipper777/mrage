"""Build and validate the frozen V7B.1 provenance-aware semantic graph."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.semantic_candidates import (
    RULE_VERSION,
    _array_fingerprint,
    canonical_undirected_edges,
)
from src.data.semantic_candidates_v7b import (
    V7B_RULE_VERSION,
    build_semantic_candidate_cache_v7b,
    load_semantic_candidate_cache_v7b,
)
from src.models.ops.semantic_operators import build_modality_knn_operators


def _digest_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _feature_array(value: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    array = np.ascontiguousarray(
        value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value,
        dtype=np.float32,
    )
    if array.ndim != 2:
        raise ValueError(f"{name} features must have shape [num_nodes, dim]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} features contain NaN or infinite values")
    return array


def semantic_graph_identity(
    *,
    dataset: str,
    task: str,
    text_features: np.ndarray | torch.Tensor,
    visual_features: np.ndarray | torch.Tensor,
    physical_edge_index: torch.Tensor | np.ndarray,
    knn_k: int = 8,
    faiss_seed: int = 2026,
    faiss_threads: int = 8,
) -> tuple[dict[str, Any], str]:
    """Return a deterministic identity for this task's complete graph input."""
    if int(knn_k) != 8:
        raise ValueError("the frozen V7B.1 semantic graph is validated only for K=8")
    if task not in {"nc", "lp"}:
        raise ValueError(f"semantic graph task must be 'nc' or 'lp', got {task!r}")
    if int(faiss_threads) < 1:
        raise ValueError("faiss_threads must be positive")

    text = _feature_array(text_features, "Text")
    visual = _feature_array(visual_features, "Visual")
    if text.shape[0] != visual.shape[0] or text.shape[0] < 2:
        raise ValueError("Text and Visual matrices must have the same node count >= 2")
    num_nodes = int(text.shape[0])

    def feature_source(values: np.ndarray) -> dict[str, Any]:
        digest = _array_fingerprint(values)
        return {
            "identifier": "in_memory_array",
            "file_sha256": digest,
            "array_sha256": digest,
            "file_bytes": int(values.nbytes),
        }

    feature_identity = {
        "dataset": str(dataset),
        "num_nodes": num_nodes,
        "text_dim": int(text.shape[1]),
        "visual_dim": int(visual.shape[1]),
        "text": feature_source(text),
        "visual": feature_source(visual),
    }
    feature_fingerprint = _digest_json(feature_identity)

    physical = canonical_undirected_edges(physical_edge_index, num_nodes)
    physical_fingerprint = hashlib.sha256(
        np.ascontiguousarray(physical, dtype=np.int64).view(np.uint8).tobytes()
    ).hexdigest()
    import faiss

    identity = {
        "schema": "mrage-semantic-graph-identity-v1",
        "dataset": str(dataset),
        "task": task,
        "num_nodes": num_nodes,
        # The feature hashes are row-order-sensitive and therefore bind node order.
        "node_order_fingerprint": feature_fingerprint,
        "feature_fingerprint": feature_fingerprint,
        "physical_edge_fingerprint": physical_fingerprint,
        "rule_version": RULE_VERSION,
        "v7b_rule_version": V7B_RULE_VERSION,
        "knn_k": int(knn_k),
        "faiss": {
            "version": str(getattr(faiss, "__version__", "unknown")),
            "seed": int(faiss_seed),
            "threads": int(faiss_threads),
        },
        "source_markers": "two uint8 source counts per canonical candidate edge",
    }
    return identity, _digest_json(identity)


def build_semantic_graph(
    *,
    dataset: str,
    text_features: np.ndarray | torch.Tensor,
    visual_features: np.ndarray | torch.Tensor,
    physical_edge_index: torch.Tensor | np.ndarray,
    cache_dir: str | Path,
    knn_k: int = 8,
    faiss_seed: int = 2026,
    faiss_threads: int = 8,
) -> tuple[Path, dict[str, Any]]:
    """Run the original mean-center, L2, FAISS K=8 V7B.1 construction."""
    if int(knn_k) != 8:
        raise ValueError("the frozen V7B.1 semantic graph is validated only for K=8")
    return build_semantic_candidate_cache_v7b(
        dataset=dataset,
        text_features=text_features,
        visual_features=visual_features,
        physical_edge_index=physical_edge_index,
        cache_dir=cache_dir,
        top_k=8,
        seed=int(faiss_seed),
        thread_count=int(faiss_threads),
    )


def load_semantic_graph(
    path: str | Path,
    *,
    expected_dataset: str,
    expected_identity: dict[str, Any],
    expected_identity_fingerprint: str,
    validate: bool = True,
) -> dict[str, Any]:
    """Load a manager cache, verify its identity and restore modality operators."""
    cache_path = Path(path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    manager = payload.get("manager") if isinstance(payload, dict) else None
    if not isinstance(manager, dict):
        raise ValueError("semantic graph cache has no MRAGE manager identity")
    if manager.get("identity") != expected_identity:
        raise ValueError("semantic graph cache identity does not match current task inputs")
    if manager.get("identity_fingerprint") != expected_identity_fingerprint:
        raise ValueError("semantic graph cache identity fingerprint is inconsistent")
    if _digest_json(manager["identity"]) != manager["identity_fingerprint"]:
        raise ValueError("semantic graph cache identity metadata is corrupt")

    graph = load_semantic_candidate_cache_v7b(
        cache_path,
        expected_dataset=expected_dataset,
        expected_feature_fingerprint=expected_identity["feature_fingerprint"],
    )
    metadata = graph["metadata"]
    if metadata.get("physical_edge_fingerprint") != expected_identity["physical_edge_fingerprint"]:
        raise ValueError("semantic graph was built against a different physical topology")
    if int(metadata.get("top_k", -1)) != 8:
        raise ValueError("semantic graph cache does not use the frozen K=8 rule")
    provenance = metadata.get("index_provenance", {})
    if str(provenance.get("faiss_version")) != expected_identity["faiss"]["version"]:
        raise ValueError("semantic graph FAISS version does not match the requested configuration")
    if int(provenance.get("random_seed", -1)) != expected_identity["faiss"]["seed"]:
        raise ValueError("semantic graph FAISS seed does not match the requested configuration")
    if int(provenance.get("thread_count", -1)) != expected_identity["faiss"]["threads"]:
        raise ValueError("semantic graph FAISS thread configuration does not match")
    if metadata.get("cache_contains_labels") is not False or metadata.get("candidate_phase_label_access") is not False:
        raise ValueError("semantic graph metadata does not certify label-free construction")

    unique = graph["unique_edge_index"].long().contiguous()
    directed = graph["edge_index"].long().contiguous()
    flags = graph["source_flags"].to(torch.uint8).contiguous()
    degree = graph["degree"].long().contiguous()
    num_nodes = int(metadata["num_nodes"])
    if unique.ndim != 2 or unique.shape[0] != 2:
        raise ValueError("semantic candidate edges must have shape [2,E]")
    if directed.shape != (2, 2 * unique.size(1)):
        raise ValueError("directed semantic graph does not contain both orientations")
    if flags.shape != (unique.size(1), 2):
        raise ValueError("semantic source markers do not align with candidate edges")
    if degree.shape != (num_nodes,):
        raise ValueError("semantic candidate degree vector has an invalid shape")
    if flags.numel() and (int(flags.max()) > 2 or bool((flags.sum(dim=1) == 0).any())):
        raise ValueError("semantic source markers are incomplete or out of range")
    if unique.numel() and (
        int(unique.min()) < 0 or int(unique.max()) >= num_nodes or bool((unique[0] >= unique[1]).any())
    ):
        raise ValueError("semantic candidate edges are not canonical valid node pairs")
    if validate:
        expected_directed = torch.cat((unique, unique.flip(0)), dim=1)
        if not torch.equal(directed, expected_directed):
            raise ValueError("semantic directed edges are not the symmetrized canonical candidate set")
        expected_degree = torch.bincount(unique.reshape(-1), minlength=num_nodes)
        if not torch.equal(degree, expected_degree):
            raise ValueError("semantic candidate degree vector does not match the candidate edges")

    operators = build_modality_knn_operators(unique, flags, num_nodes)
    for modality, operator in operators.items():
        edges, weights = operator["edge_index"], operator["weight"]
        if edges.size(1) != weights.numel() or not bool(torch.isfinite(weights).all()):
            raise ValueError(f"restored {modality} semantic operator is invalid")
        if validate and edges.numel():
            column = 0 if modality == "text" else 1
            selected = unique[:, flags[:, column] > 0]
            expected_edges = torch.cat((selected, selected.flip(0)), dim=1).contiguous()
            if not torch.equal(edges, expected_edges):
                raise ValueError(f"restored {modality} operator does not follow its source markers")
            expected_degree = torch.bincount(edges[1], minlength=num_nodes).float()
            expected_weights = expected_degree[edges[1]].clamp_min(1).reciprocal()
            if not torch.equal(weights, expected_weights):
                raise ValueError(f"restored {modality} operator weights do not match V7B.1 normalization")

    graph["operators"] = operators
    graph["manager_identity"] = manager
    return graph
