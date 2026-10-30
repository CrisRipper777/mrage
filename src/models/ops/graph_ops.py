# Extracted computation from src/models/psce_mag_v7a.py at 8f1f9c4.
from __future__ import annotations

import torch
from torch.autograd import Function



class ChunkedWeightedSpMM(Function):
    """Sparse weighted source-to-target sum with edge-chunk recomputation.

    The custom backward gathers source states again, so autograd does not keep
    an E-by-hidden message tensor for every edge.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, edge_index: torch.Tensor, weight: torch.Tensor, chunk_size: int):
        src, dst = edge_index
        output = torch.zeros_like(x)
        chunk_size = int(chunk_size)
        for start in range(0, src.numel(), chunk_size):
            end = min(start + chunk_size, src.numel())
            source = x.index_select(0, src[start:end])
            message = source * weight[start:end].unsqueeze(-1)
            output.index_add_(0, dst[start:end], message)
        ctx.save_for_backward(x, edge_index, weight)
        ctx.chunk_size = chunk_size
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, edge_index, weight = ctx.saved_tensors
        src, dst = edge_index
        grad_x = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(weight) if ctx.needs_input_grad[2] else None
        for start in range(0, src.numel(), ctx.chunk_size):
            end = min(start + ctx.chunk_size, src.numel())
            source_ids = src[start:end]
            target_ids = dst[start:end]
            grad_target = grad_output.index_select(0, target_ids)
            source = x.index_select(0, source_ids)
            if grad_x is not None:
                grad_x.index_add_(
                    0,
                    source_ids,
                    grad_target * weight[start:end].unsqueeze(-1),
                )
            if grad_weight is not None:
                grad_weight[start:end] = (grad_target * source).sum(dim=-1)
        return grad_x, None, grad_weight, None


def chunked_weighted_spmm(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int = 32768,
) -> torch.Tensor:
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("semantic edge_index must have shape [2,E]")
    if edge_index.size(1) != weight.numel():
        raise ValueError("semantic edge and weight lengths differ")
    return ChunkedWeightedSpMM.apply(x, edge_index, weight, int(chunk_size))
