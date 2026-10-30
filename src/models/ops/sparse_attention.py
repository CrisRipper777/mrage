# Extracted computation from src/models/cg_cge_v1.py at 8f1f9c4.
from __future__ import annotations

import torch



class _ReceiverSparseAttention(torch.autograd.Function):
    """Segmented softmax aggregation with bounded E x D temporary memory."""

    @staticmethod
    def forward(ctx, query, key, value, src, dst, chunk_size):
        if query.ndim != 2 or key.shape != query.shape:
            raise ValueError("query/key must have matching [N, attention_dim] shapes")
        if value.ndim != 2 or value.size(0) != query.size(0):
            raise ValueError("value must have shape [N, value_dim]")
        if src.ndim != 1 or dst.shape != src.shape:
            raise ValueError("src and dst must be matching edge vectors")
        if src.device.type == "cpu" and src.numel() and (
            int(src.min()) < 0 or int(src.max()) >= query.size(0)
            or int(dst.min()) < 0 or int(dst.max()) >= query.size(0)
        ):
            raise ValueError("attention edge contains an out-of-range global node ID")
        if int(chunk_size) < 1:
            raise ValueError("attention chunk size must be positive")

        n, edge_count = query.size(0), src.numel()
        scale = float(query.size(1)) ** -0.5
        q32, k32, v32 = query.float(), key.float(), value.float()
        if edge_count:
            scores = q32.new_empty((edge_count,))
            for start in range(0, edge_count, int(chunk_size)):
                end = min(start + int(chunk_size), edge_count)
                scores[start:end] = (
                    q32.index_select(0, dst[start:end])
                    * k32.index_select(0, src[start:end])
                ).sum(-1) * scale
            row_max = scores.new_full((n,), -torch.inf)
            row_max.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
            exp_scores = torch.exp(scores - row_max.index_select(0, dst))
            row_sum = scores.new_zeros((n,))
            row_sum.index_add_(0, dst, exp_scores)
            attention = exp_scores / row_sum.index_select(0, dst).clamp_min(1.0e-30)
        else:
            attention = q32.new_empty((0,))

        output = v32.new_zeros((n, value.size(1)))
        for start in range(0, edge_count, int(chunk_size)):
            end = min(start + int(chunk_size), edge_count)
            src_chunk, dst_chunk = src[start:end], dst[start:end]
            weighted = v32.index_select(0, src_chunk) * attention[start:end, None]
            output.index_add_(0, dst_chunk, weighted)

        ctx.save_for_backward(query, key, value, src, dst, attention)
        ctx.chunk_size = int(chunk_size)
        ctx.scale = scale
        return output.to(value.dtype), attention

    @staticmethod
    def backward(ctx, grad_output, grad_attention_output):
        query, key, value, src, dst, attention = ctx.saved_tensors
        n, edge_count = query.size(0), src.numel()
        q32, k32, v32 = query.float(), key.float(), value.float()
        grad_out = (
            torch.zeros_like(value, dtype=torch.float32)
            if grad_output is None
            else grad_output.float()
        )
        grad_att_direct = (
            torch.zeros_like(attention)
            if grad_attention_output is None
            else grad_attention_output.float()
        )
        grad_value = torch.zeros_like(v32)
        grad_att = torch.zeros_like(attention, dtype=torch.float32)

        # First pass computes dL/da and dL/dV without retaining edge vectors
        # wider than a single chunk.
        for start in range(0, edge_count, ctx.chunk_size):
            end = min(start + ctx.chunk_size, edge_count)
            src_chunk, dst_chunk = src[start:end], dst[start:end]
            go = grad_out.index_select(0, dst_chunk)
            v = v32.index_select(0, src_chunk)
            alpha = attention[start:end].float()
            grad_att[start:end] = (go * v).sum(-1) + grad_att_direct[start:end]
            grad_value.index_add_(0, src_chunk, go * alpha[:, None])

        row_inner = grad_att.new_zeros((n,))
        if edge_count:
            row_inner.index_add_(0, dst, attention.float() * grad_att)
        grad_score = attention.float() * (grad_att - row_inner.index_select(0, dst))
        grad_query = torch.zeros_like(q32)
        grad_key = torch.zeros_like(k32)
        for start in range(0, edge_count, ctx.chunk_size):
            end = min(start + ctx.chunk_size, edge_count)
            src_chunk, dst_chunk = src[start:end], dst[start:end]
            scaled = grad_score[start:end, None] * ctx.scale
            grad_query.index_add_(
                0, dst_chunk, scaled * k32.index_select(0, src_chunk)
            )
            grad_key.index_add_(
                0, src_chunk, scaled * q32.index_select(0, dst_chunk)
            )

        return (
            grad_query.to(query.dtype),
            grad_key.to(key.dtype),
            grad_value.to(value.dtype),
            None,
            None,
            None,
        )


def receiver_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    edge_index: torch.Tensor,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate only listed ``src -> dst`` edges with per-receiver softmax."""
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2,E]")
    return _ReceiverSparseAttention.apply(
        query, key, value, edge_index[0], edge_index[1], int(chunk_size)
    )
