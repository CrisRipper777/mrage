"""Task-isolated, locked and atomic semantic graph cache management."""

from __future__ import annotations

import fcntl
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.data.semantic_graph import (
    build_semantic_graph,
    load_semantic_graph,
    semantic_graph_identity,
)


@dataclass(frozen=True)
class SemanticGraphCache:
    path: Path
    identity: dict[str, Any]
    identity_fingerprint: str
    metadata: dict[str, Any]
    cache_hit: bool


def _resolve_cache_root(cache_dir: str | Path, project_root: str | Path) -> Path:
    path = Path(str(cache_dir)).expanduser()
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()


def _save_atomic(path: Path, payload: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=".semantic_graph.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def ensure_semantic_graph_cache(
    *,
    dataset: str,
    task: str,
    text_features: torch.Tensor,
    visual_features: torch.Tensor,
    physical_edge_index: torch.Tensor,
    config,
    project_root: str | Path,
) -> SemanticGraphCache:
    """Load or build one validated cache for a task/topology/input identity.

    The caller supplies only features and the task's allowed physical graph.
    Labels and held-out LP edges are never accepted by this interface.
    """
    mode = str(config.get("mode", "auto")).strip().lower()
    if mode not in {"auto", "existing", "rebuild"}:
        raise ValueError("semantic_graph.mode must be auto, existing, or rebuild")
    knn_k = int(config.get("knn_k", 8))
    faiss_seed = int(config.get("faiss_seed", 2026))
    faiss_threads = int(config.get("faiss_threads", 8))
    identity, identity_fingerprint = semantic_graph_identity(
        dataset=dataset,
        task=task,
        text_features=text_features,
        visual_features=visual_features,
        physical_edge_index=physical_edge_index,
        knn_k=knn_k,
        faiss_seed=faiss_seed,
        faiss_threads=faiss_threads,
    )
    validate = bool(config.get("validate", True))
    root = _resolve_cache_root(config.get("cache_dir", "artifacts/semantic_graphs"), project_root)
    identity_dir = root / dataset / task / identity_fingerprint
    identity_dir.mkdir(parents=True, exist_ok=True)
    cache_path = identity_dir / "semantic_graph.pt"
    lock_path = identity_dir / ".lock"

    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if mode != "rebuild" and cache_path.is_file():
            try:
                graph = load_semantic_graph(
                    cache_path,
                    expected_dataset=dataset,
                    expected_identity=identity,
                    expected_identity_fingerprint=identity_fingerprint,
                    validate=validate,
                )
                return SemanticGraphCache(
                    path=cache_path,
                    identity=identity,
                    identity_fingerprint=identity_fingerprint,
                    metadata=graph["metadata"],
                    cache_hit=True,
                )
            except Exception as error:
                if mode == "existing":
                    raise RuntimeError(f"existing semantic graph cache is invalid: {error}") from error
                quarantine = identity_dir / f"semantic_graph.corrupt.{time.time_ns()}.pt"
                os.replace(cache_path, quarantine)

        if mode == "existing":
            raise FileNotFoundError(
                f"no valid semantic graph cache for {dataset}/{task} identity {identity_fingerprint}"
            )

        with tempfile.TemporaryDirectory(prefix=".build-", dir=identity_dir) as build_dir:
            built_path, metadata = build_semantic_graph(
                dataset=dataset,
                text_features=text_features,
                visual_features=visual_features,
                physical_edge_index=physical_edge_index,
                cache_dir=build_dir,
                knn_k=knn_k,
                faiss_seed=faiss_seed,
                faiss_threads=faiss_threads,
            )
            payload = torch.load(built_path, map_location="cpu", weights_only=False)
            if metadata.get("feature_fingerprint") != identity["feature_fingerprint"]:
                raise RuntimeError("semantic builder feature fingerprint differs from manager identity")
            if metadata.get("physical_edge_fingerprint") != identity["physical_edge_fingerprint"]:
                raise RuntimeError("semantic builder physical graph fingerprint differs from manager identity")
            payload["manager"] = {
                "identity": identity,
                "identity_fingerprint": identity_fingerprint,
                "task": task,
                "candidate_fingerprint": metadata["candidate_fingerprint"],
                "v7b_fingerprint": metadata["v7b_fingerprint"],
            }
            _save_atomic(cache_path, payload)

        graph = load_semantic_graph(
            cache_path,
            expected_dataset=dataset,
            expected_identity=identity,
            expected_identity_fingerprint=identity_fingerprint,
            validate=validate,
        )
        return SemanticGraphCache(
            path=cache_path,
            identity=identity,
            identity_fingerprint=identity_fingerprint,
            metadata=graph["metadata"],
            cache_hit=False,
        )
