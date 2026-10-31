"""Independent NC ablations for the frozen MRAGE encoder.

The implementation subclasses the formal model and changes only the filter
bank or the specific cross-modal path named by ``model.ablation_variant``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.models.mrage import Bottleneck, Model as MRAGEModel, SharedExpertBank
from src.models.ops.semantic_operators import _MODALITIES


class SingleGraphFilter(nn.Module):
    """One learnable polynomial graph filter shared by every view."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int, dropout: float, eta: float):
        super().__init__()
        self.eta = float(eta)
        # Start from the mean normalized polynomial represented by the original
        # four fixed expert initializers, then learn one shared coefficient set.
        initial = torch.tensor(
            [
                [0.62, -0.27, 0.18, -0.09],
                [-0.31, 0.57, -0.11, 0.19],
                [0.22, 0.16, -0.53, 0.29],
                [-0.18, -0.23, 0.37, 0.46],
            ], dtype=torch.float32,
        )
        initial = initial / initial.abs().sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        self.theta = nn.Parameter(initial.mean(dim=0))
        self.bottleneck = Bottleneck(hidden_dim, bottleneck_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, basis: list[torch.Tensor], support: torch.Tensor
    ) -> torch.Tensor:
        max_order = len(basis) - 1
        raw = self.theta[: max_order + 1]
        coefficients = raw / (raw.abs().sum() + 1.0e-8)
        filtered = torch.zeros_like(basis[0])
        for order, coefficient in enumerate(coefficients):
            filtered = filtered + coefficient.to(filtered.dtype) * basis[order]
        residual = self.eta * self.bottleneck(filtered)
        return self.norm(filtered + residual) * support.to(filtered.dtype).unsqueeze(-1)


class Model(MRAGEModel):
    """MRAGE encoder with one of five controlled NC component ablations."""

    ABLATION_VARIANTS = (
        "single_filter",
        "unshared_experts",
        "no_semantic_propagation",
        "no_topological_collaboration",
        "no_semantic_refinement",
        "full",
    )

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__(cfg, data_info)
        variant = str(cfg.model.get("ablation_variant", "single_filter")).strip().lower()
        if variant not in self.ABLATION_VARIANTS:
            raise ValueError(
                f"model.ablation_variant must be one of {self.ABLATION_VARIANTS}, got {variant!r}"
            )
        self.ablation_variant = variant
        if variant == "single_filter":
            # Do not leave unused Top-2 routing or four-expert parameters in
            # the single-filter model/checkpoint/parameter count.
            self.filter_bank = None
            self.router = None
            with torch.random.fork_rng(devices=[]):
                self.single_filter = SingleGraphFilter(
                    self.hidden_dim,
                    int(cfg.model.get("expert_bottleneck", 64)),
                    self.dropout_p,
                    self.eta,
                )
        elif variant == "unshared_experts":
            # Private and collaborative calls use the same modality/relation
            # pool. Selection is explicit; no forward-time module swapping.
            self.filter_bank = None
            pool_names = (
                "text_physical", "text_semantic", "visual_physical", "visual_semantic"
            )
            with torch.random.fork_rng(devices=[]):
                self.filter_banks = nn.ModuleDict(
                    {
                        name: SharedExpertBank(
                            self.hidden_dim,
                            int(cfg.model.get("expert_bottleneck", 64)),
                            self.dropout_p,
                            self.eta,
                        )
                        for name in pool_names
                    }
                )

    def _bank_for(self, modality_id: int, relation_id: int) -> SharedExpertBank:
        modality = _MODALITIES[int(modality_id)]
        relation = ("physical", "semantic")[int(relation_id)]
        return self.filter_banks[f"{modality}_{relation}"]

    @staticmethod
    def _route(
        outputs: torch.Tensor,
        logits: torch.Tensor,
        support: torch.Tensor,
        top_k: int,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dense = torch.softmax(logits.float(), dim=-1)
        top_logits, top_ids = torch.topk(logits, top_k, dim=-1)
        top_weights = torch.softmax(top_logits.float(), dim=-1).to(outputs.dtype)
        route_weights = torch.zeros_like(dense).scatter(1, top_ids, top_weights.float()).to(outputs.dtype)
        selected = torch.zeros_like(dense, dtype=torch.bool).scatter(1, top_ids, True)
        response = (outputs * route_weights.transpose(0, 1).unsqueeze(-1)).sum(dim=0)
        if bool(support.any()):
            importance = dense[support].mean(dim=0)
            selected_share = selected[support].float().mean(dim=0) / float(top_k)
            balance = int(num_experts) * (importance * selected_share).sum()
        else:
            balance = dense.sum() * 0.0
        return response, balance

    def _filter(self, basis, support, current, relation_id, modality_id, cross_context):
        if self.ablation_variant == "no_semantic_propagation" and int(relation_id) == 1:
            zero = current * 0.0
            return zero, zero.sum()

        if self.ablation_variant == "single_filter":
            if self.training and torch.is_grad_enabled():
                response = checkpoint(
                    lambda *inputs: self.single_filter(list(inputs[:-1]), inputs[-1]),
                    *basis, support, use_reentrant=False, preserve_rng_state=True,
                )
            else:
                response = self.single_filter(basis, support)
            return response, response.sum() * 0.0

        if self.ablation_variant != "unshared_experts":
            return super()._filter(
                basis, support, current, relation_id, modality_id, cross_context
            )

        max_order = len(basis) - 1
        if self.training and torch.is_grad_enabled():
            def compute(*inputs):
                return self._unshared_filter_impl(
                    list(inputs[: max_order + 1]), inputs[max_order + 1],
                    inputs[max_order + 2], inputs[max_order + 3],
                    modality_id, relation_id,
                )

            return checkpoint(
                compute, *basis, support, current, cross_context,
                use_reentrant=False, preserve_rng_state=True,
            )

        bank = self._bank_for(modality_id, relation_id)
        outputs, _, _, _ = bank(basis, support, max_order)
        first = basis[1] if len(basis) > 1 else basis[0]
        logits = self.router(current, first, cross_context, modality_id, relation_id)
        return self._route(outputs, logits, support, self.top_k, self.num_experts)

    def _unshared_filter_impl(
        self,
        basis: list[torch.Tensor],
        support: torch.Tensor,
        current: torch.Tensor,
        cross_context: torch.Tensor,
        modality_id: int,
        relation_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Checkpoint-safe Top-2 computation using the explicitly selected pool."""
        bank = self._bank_for(modality_id, relation_id)
        max_order = len(basis) - 1
        coefficients = bank.coefficients(max_order, basis[0].dtype)
        first = basis[1] if len(basis) > 1 else basis[0]
        n = current.size(0)
        mod = self.router.modality_embedding.weight[modality_id].expand(n, -1)
        rel = self.router.relation_embedding.weight[relation_id].expand(n, -1)
        router_parts = (current, first, first - current, mod, rel, cross_context)
        first_layer = self.router.net[0]
        offset = 0
        pre_activation = None
        for part in router_parts:
            width = part.size(-1)
            contribution = F.linear(part, first_layer.weight[:, offset:offset + width], bias=None)
            pre_activation = contribution if pre_activation is None else pre_activation + contribution
            offset += width
        if offset != first_layer.in_features:
            raise RuntimeError("router input components do not match its configured linear layer")
        if first_layer.bias is not None:
            pre_activation = pre_activation + first_layer.bias
        router_hidden = F.gelu(pre_activation)
        mask = support.to(basis[0].dtype).unsqueeze(-1)
        outputs = []
        for expert_id, expert in enumerate(bank.experts):
            filtered = None
            for order in range(max_order + 1):
                term = coefficients[expert_id, order] * basis[order]
                filtered = term if filtered is None else filtered + term
            residual = self.eta * expert(filtered)
            outputs.append(bank.norms[expert_id](filtered + residual) * mask)
        logits = self.router.net[3](self.router.net[2](router_hidden))
        dense = torch.softmax(logits.float(), dim=-1)
        top_logits, top_ids = torch.topk(logits, self.top_k, dim=-1)
        top_weights = torch.softmax(top_logits.float(), dim=-1)
        route_weights = torch.zeros_like(dense).scatter(1, top_ids, top_weights)
        selected = torch.zeros_like(dense, dtype=torch.bool).scatter(1, top_ids, True)
        response = None
        for expert_id, value in enumerate(outputs):
            weighted = value * route_weights[:, expert_id:expert_id + 1].to(value.dtype)
            response = weighted if response is None else response + weighted
        if bool(support.any()):
            importance = dense[support].mean(dim=0)
            selected_share = selected[support].float().mean(dim=0) / float(self.top_k)
            balance = self.num_experts * (importance * selected_share).sum()
        else:
            balance = dense.sum() * 0.0
        return response, balance

    def _early_cross(self, source_state, projection, edge_index, edge_weight, support, reference):
        if self.ablation_variant == "no_topological_collaboration":
            return torch.zeros_like(reference)
        return super()._early_cross(
            source_state, projection, edge_index, edge_weight, support, reference
        )

    def _late(self, target, target_id, source, h_source, u_private_source, u_collaborative_target):
        if self.ablation_variant == "no_semantic_refinement":
            return u_collaborative_target
        return super()._late(
            target, target_id, source, h_source, u_private_source, u_collaborative_target
        )
