# Extracted computation from src/models/rmcge_v2.py at 8f1f9c4.
from __future__ import annotations

import torch

_MODALITIES = ("text", "visual")



def build_modality_knn_operators(
    unique_edge_index: torch.Tensor,
    source_flags: torch.Tensor,
    num_nodes: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Build separately row-normalized symmetric KNN supports from V7B flags."""
    if unique_edge_index.ndim != 2 or unique_edge_index.shape[0] != 2:
        raise ValueError("unique_edge_index must have shape [2,E]")
    if source_flags.shape != (unique_edge_index.size(1), 2):
        raise ValueError("source_flags must have shape [E,2]")
    result: dict[str, dict[str, torch.Tensor]] = {}
    for column, modality in enumerate(_MODALITIES):
        pairs = unique_edge_index[:, source_flags[:, column] > 0]
        if pairs.numel():
            directed = torch.cat((pairs, pairs.flip(0)), dim=1).contiguous()
            degree = torch.zeros(num_nodes, dtype=torch.float32, device=directed.device)
            degree.index_add_(0, directed[1], torch.ones(directed.size(1), device=directed.device))
            weight = degree[directed[1]].clamp_min(1).reciprocal()
        else:
            directed = unique_edge_index.new_empty((2, 0))
            weight = torch.empty(0, dtype=torch.float32, device=unique_edge_index.device)
        result[modality] = {"edge_index": directed, "weight": weight}
    return result
