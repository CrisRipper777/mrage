"""Capability-level NC ablations for the frozen MRAGE encoder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mrage import Model as MRAGEModel


class Model(MRAGEModel):
    """MRAGE with one high-level information capability removed."""

    ABLATION_VARIANTS = (
        "physical_only",
        "semantic_only",
        "no_cross_modal_collaboration",
        "feature_only",
        "full",
    )

    def __init__(self, cfg, data_info: dict[str, Any]):
        task_name = str(cfg.task.get("name", "nc")).strip().lower()
        if task_name != "nc":
            raise ValueError(
                "mrage_capability_ablation supports NC only; "
                f"task={task_name!r} is unsupported (LP is not implemented)."
            )
        variant = str(cfg.model.get("ablation_variant", "physical_only")).strip().lower()
        if variant not in self.ABLATION_VARIANTS:
            raise ValueError(
                f"model.ablation_variant must be one of {self.ABLATION_VARIANTS}, "
                f"got {variant!r}"
            )
        super().__init__(cfg, data_info)
        self.ablation_variant = variant
        if variant == "feature_only":
            self._freeze_graph_modules_for_feature_only()

    def _freeze_graph_modules_for_feature_only(self) -> None:
        """Keep only modality projectors, NC heads, and original fusion trainable."""
        unused_modules = (
            "filter_bank", "router", "cross_projections", "query_projections",
            "key_projections", "source_selector", "channel_gate", "joint_view",
        )
        for name in unused_modules:
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        for name in (
            "cross_lambda_raw", "tau_raw", "rho_raw", "gamma_raw", "beta_raw",
        ):
            parameter = getattr(self, name, None)
            if isinstance(parameter, nn.Parameter):
                parameter.requires_grad_(False)

    def _filter(
        self, basis, support, current, relation_id, modality_id, cross_context
    ):
        disable = (
            self.ablation_variant == "physical_only" and int(relation_id) == 1
        ) or (
            self.ablation_variant == "semantic_only" and int(relation_id) == 0
        )
        if disable:
            # Connected zero tensors keep non-reentrant checkpoint/backward
            # behavior well-defined while removing this relation's response.
            zero = current * 0.0
            return zero, zero.sum()
        return super()._filter(
            basis, support, current, relation_id, modality_id, cross_context
        )

    def _early_cross(
        self, source_state, projection, edge_index, edge_weight, support, reference
    ):
        if self.ablation_variant in {
            "semantic_only", "no_cross_modal_collaboration"
        }:
            return torch.zeros_like(reference)
        return super()._early_cross(
            source_state, projection, edge_index, edge_weight, support, reference
        )

    def _late(
        self,
        target,
        target_id,
        source,
        h_source,
        u_private_source,
        u_collaborative_target,
    ):
        if self.ablation_variant == "physical_only":
            return self._late_same_node_only(
                target, target_id, source, h_source, u_private_source,
                u_collaborative_target,
            )
        if self.ablation_variant == "no_cross_modal_collaboration":
            return u_collaborative_target
        return super()._late(
            target, target_id, source, h_source, u_private_source,
            u_collaborative_target,
        )

    def _late_same_node_only(
        self,
        target: str,
        target_id: int,
        source: str,
        h_source: torch.Tensor,
        u_private_source: torch.Tensor,
        u_collaborative_target: torch.Tensor,
    ) -> torch.Tensor:
        """Reuse MRAGE's Late selector/gate while making Semantic KNN unavailable."""
        del h_source  # Same-node projection uses the source's private state only.
        projection_name = "visual_to_text" if source == "visual" else "text_to_visual"
        phi = self.cross_projections[projection_name]
        n = u_collaborative_target.size(0)
        c_sem = torch.zeros_like(u_collaborative_target)
        support_sem = torch.zeros(n, dtype=torch.bool, device=u_collaborative_target.device)
        if self.same_node_enabled:
            c_self = phi(u_private_source)
            support_self = torch.ones_like(support_sem)
        else:
            c_self = torch.zeros_like(c_sem)
            support_self = torch.zeros_like(support_sem)

        cosine = F.cosine_similarity(
            c_sem.detach(), c_self.detach(), dim=-1, eps=1.0e-8
        )
        selector_parts = (
            u_collaborative_target,
            c_sem,
            c_self,
            cosine[:, None],
            support_sem.to(c_sem.dtype)[:, None],
            support_self.to(c_sem.dtype)[:, None],
        )
        logits = self._sequential_from_parts(self.source_selector, selector_parts).clone()
        # The Semantic-neighbor option is always masked; Same-node remains
        # available only when enabled, and None is always available.
        logits[:, 0] = -torch.inf
        logits[~support_self, 1] = -torch.inf
        source_weights = torch.softmax(logits, dim=-1)
        any_support = support_self
        none_only = torch.zeros_like(source_weights)
        none_only[:, 2] = 1.0
        source_weights = torch.where(any_support[:, None], source_weights, none_only)
        p_accept = source_weights[:, 1]
        candidate_cond = (
            source_weights[:, 1:2] * c_self
        ) / p_accept.clamp_min(1.0e-8)[:, None]
        candidate_cond = candidate_cond * any_support.to(candidate_cond.dtype).unsqueeze(-1)

        interaction_parts = (
            u_collaborative_target,
            candidate_cond,
            u_collaborative_target * candidate_cond,
            (u_collaborative_target - candidate_cond).abs(),
        )
        gate = torch.sigmoid(self._sequential_from_parts(self.channel_gate, interaction_parts))
        unbounded = (
            self.cross_lambda[target_id]
            * p_accept[:, None]
            * gate
            * candidate_cond
        )
        unbounded = unbounded * any_support.to(unbounded.dtype).unsqueeze(-1)
        ref_norm = u_collaborative_target.float().norm(dim=-1, keepdim=True)
        delta_norm = unbounded.float().norm(dim=-1, keepdim=True)
        scale = torch.minimum(
            torch.ones_like(delta_norm),
            self.tau[target_id] * ref_norm / (delta_norm + self.eps),
        )
        controlled = unbounded * scale.to(unbounded.dtype)
        if self.training and self.late_cross_dropout > 0:
            keep = torch.rand(
                (controlled.size(0), 1), device=controlled.device
            ) >= self.late_cross_dropout
            controlled = controlled * keep.to(controlled.dtype)
        controlled = controlled * any_support.to(controlled.dtype).unsqueeze(-1)
        return u_collaborative_target + controlled

    def _feature_only_forward(self, x: torch.Tensor) -> tuple:
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected x shape [nodes,{self.input_dim}]")
        if self._batch_n_id is None:
            if self.num_nodes and x.size(0) != self.num_nodes:
                raise RuntimeError("MRAGE full-graph forward requires the cached global node order")
        elif x.size(0) != self._batch_n_id.numel():
            raise RuntimeError("MRAGE sampled features do not align with global batch node IDs")

        h_text = self.projectors["text"](x[:, :self.text_dim])
        h_visual = self.projectors["visual"](x[:, self.text_dim:self.input_dim])
        fused = torch.cat((h_text, h_visual), dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused)
            + self.fusion_linear2(
                self.fusion_dropout(F.gelu(self.fusion_linear1(fused)))
            )
        )
        balance = z.sum() * 0.0
        info = {
            "balance_loss": balance.detach(),
            "aux_loss": balance.detach(),
            "unimodal_logits": {
                "text": self.unimodal_heads["text"](h_text),
                "visual": self.unimodal_heads["visual"](h_visual),
            },
        }
        return z, None, None, balance, info

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None):
        if self.ablation_variant == "feature_only":
            # The original feature projector/fusion/head path is independent of
            # both graph inputs and semantic cache operators.
            return self._feature_only_forward(x)
        return super().forward(x, edge_index)
