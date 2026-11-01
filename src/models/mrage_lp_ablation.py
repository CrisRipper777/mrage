"""Sampled-LP MRAGE ablations for the Sports copurchase benchmark.

This class combines the existing component/capability ablation operators with
the formal MRAGE LinkNeighborLoader adapter. It keeps LP sampling and final
inference in the existing task runner.
"""

from __future__ import annotations

from typing import Any

import torch

from src.models.mrage_ablation import (
    Model as ComponentAblationModel,
    SingleGraphFilter,
)
from src.models.mrage_capability_ablation import Model as CapabilityAblationModel
from src.models.mrage_lp import Model as SampledLPModel


class Model(CapabilityAblationModel, ComponentAblationModel, SampledLPModel):
    """MRAGE sampled-LP adapter with four capability variants and Single Filter."""

    ABLATION_VARIANTS = (
        "single_filter",
        "physical_only",
        "semantic_only",
        "no_cross_modal_collaboration",
        "feature_only",
        "full",
    )

    def __init__(self, cfg, data_info: dict[str, Any]):
        task_name = str(cfg.task.get("name", "lp")).strip().lower()
        if task_name != "lp":
            raise ValueError(
                "mrage_lp_ablation supports sampled LP only; "
                f"task={task_name!r} is unsupported."
            )
        variant = str(cfg.model.get("ablation_variant", "single_filter")).strip().lower()
        if variant not in self.ABLATION_VARIANTS:
            raise ValueError(
                f"model.ablation_variant must be one of {self.ABLATION_VARIANTS}, "
                f"got {variant!r}"
            )

        # Invoke the official sampled-LP adapter initializer directly. Its
        # super() continues to MRAGEModel in this class's shared MRO.
        SampledLPModel.__init__(self, cfg, data_info)
        self.ablation_variant = variant

        if variant == "single_filter":
            self.filter_bank = None
            self.router = None
            with torch.random.fork_rng(devices=[]):
                self.single_filter = SingleGraphFilter(
                    self.hidden_dim,
                    int(cfg.model.get("expert_bottleneck", 64)),
                    self.dropout_p,
                    self.eta,
                )
        elif variant == "feature_only":
            self._freeze_graph_modules_for_feature_only()

    def bind_lp_graph(self, x: torch.Tensor, train_edge_index: torch.Tensor) -> None:
        # These variants have no semantic-source path that needs the adapter's
        # expanded physical context. Their sampled supervision graph remains
        # the standard LinkNeighborLoader [5, 5, 5] graph.
        if self.ablation_variant in {"physical_only", "feature_only"}:
            return
        super().bind_lp_graph(x, train_edge_index)

    def prepare_sampled_lp_batch(self, batch, *, sample_seed: int) -> None:
        if self.ablation_variant == "physical_only":
            # Do not let semantic KNN candidates add nodes or physical edges to
            # a topology-only batch. Base MRAGE still expects a local semantic
            # operator, so provide empty local operators for its zeroed filters.
            empty_edges = torch.empty((2, 0), dtype=torch.long)
            empty_weights = torch.empty(0, dtype=torch.float32)
            self._sampled_semantic_edges = {
                "text": (empty_edges.clone(), empty_weights.clone()),
                "visual": (empty_edges.clone(), empty_weights.clone()),
            }
            self._sampled_semantic_edge_flags = {
                "text": torch.empty((0, 2), dtype=torch.uint8),
                "visual": torch.empty((0, 2), dtype=torch.uint8),
            }
            self._batch_n_id = batch.n_id
            self.last_sampled_context = {
                "approximation": "physical-only; standard LinkNeighborLoader subgraph only",
                "semantic_edges_text": 0,
                "semantic_edges_visual": 0,
                "extra_train_physical_edges": 0,
            }
            return
        if self.ablation_variant == "feature_only":
            # Feature-only predictions use only endpoint features. Skip the
            # expensive semantic closure and extra physical-context sampling.
            self._sampled_semantic_edges = None
            self._sampled_semantic_edge_flags = None
            self._batch_n_id = batch.n_id
            self.last_sampled_context = {
                "approximation": "feature-only; graph context unused",
                "semantic_edges_text": 0,
                "semantic_edges_visual": 0,
                "extra_train_physical_edges": 0,
            }
            return
        super().prepare_sampled_lp_batch(batch, sample_seed=sample_seed)
