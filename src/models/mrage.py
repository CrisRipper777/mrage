"""Formal MRAGE encoder migrated from the frozen MR-CGE V1.3-A model."""

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import remove_self_loops

from src.data.semantic_graph import load_semantic_graph
from src.models.ops.graph_ops import chunked_weighted_spmm
from src.models.ops.projection import _IntrinsicProjector
from src.models.ops.semantic_operators import _MODALITIES
from src.models.ops.sparse_attention import receiver_sparse_attention


# PyTorch's non-reentrant checkpoint recomputation still enters the deprecated
# torch.cpu.amp.autocast context internally. Filter only that upstream warning;
# checkpointing, gradients, and unrelated warnings remain unchanged.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`torch\.cpu\.amp\.autocast\(args\.\.\.\)` is deprecated\.",
    module=r"torch\.utils\.checkpoint",
)


def capture_rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Capture CPU and selected CUDA RNG state for paired filter dropout."""
    cpu_state = torch.random.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    return cpu_state, cuda_state


@contextmanager
def replay_rng_state(state: tuple[torch.Tensor, torch.Tensor | None], device: torch.device):
    """Replay a prior call's RNG stream locally, then restore the caller's stream."""
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.random.set_rng_state(state[0])
        if device.type == "cuda" and state[1] is not None:
            torch.cuda.set_rng_state(state[1], device)
        yield


def _run_bottleneck(module: nn.Module, value: torch.Tensor) -> torch.Tensor:
    if module.training and torch.is_grad_enabled():
        return checkpoint(module, value, use_reentrant=False, preserve_rng_state=True)
    return module(value)


class Bottleneck(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedExpertBank(nn.Module):
    """One expert object bank used for every private/collaborative graph view."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int, dropout: float, eta: float):
        super().__init__()
        self.eta = float(eta)
        self.theta = nn.Parameter(
            torch.tensor(
                [
                    [0.62, -0.27, 0.18, -0.09],
                    [-0.31, 0.57, -0.11, 0.19],
                    [0.22, 0.16, -0.53, 0.29],
                    [-0.18, -0.23, 0.37, 0.46],
                ],
                dtype=torch.float32,
            )
        )
        self.experts = nn.ModuleList(
            [Bottleneck(hidden_dim, bottleneck_dim, dropout) for _ in range(4)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(4)])

    def coefficients(self, max_order: int, dtype: torch.dtype) -> torch.Tensor:
        order_mask = torch.arange(4, device=self.theta.device) <= int(max_order)
        raw = self.theta * order_mask.to(self.theta.dtype)[None, :]
        return (raw / (raw.abs().sum(dim=-1, keepdim=True) + 1.0e-8)).to(dtype)

    def forward(
        self,
        basis: list[torch.Tensor],
        support: torch.Tensor,
        max_order: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        coefficients = self.coefficients(max_order, basis[0].dtype)
        stacked = torch.stack(basis[: max_order + 1], dim=0)
        masked_coefficients = coefficients[:, : max_order + 1]
        filtered = torch.einsum("kl,lnd->knd", masked_coefficients, stacked)
        outputs = []
        residuals = []
        mask = support.to(filtered.dtype).unsqueeze(-1)
        for index, expert in enumerate(self.experts):
            residual = self.eta * _run_bottleneck(expert, filtered[index])
            value = self.norms[index](filtered[index] + residual)
            outputs.append(value * mask)
            residuals.append(residual * mask)
        return torch.stack(outputs), filtered, coefficients, torch.stack(residuals)


class GraphAwareRouter(nn.Module):
    def __init__(self, hidden_dim: int, embed_dim: int, hidden_router_dim: int, dropout: float):
        super().__init__()
        self.modality_embedding = nn.Embedding(2, embed_dim)
        self.relation_embedding = nn.Embedding(2, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(3 * hidden_dim + 2 * embed_dim + hidden_dim, hidden_router_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_router_dim, 4),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.012)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([-0.009, -0.003, 0.004, 0.008]))

    def forward(
        self,
        current: torch.Tensor,
        first_neighbor: torch.Tensor,
        cross_context: torch.Tensor,
        modality_id: int,
        relation_id: int,
    ) -> torch.Tensor:
        n = current.size(0)
        mod = self.modality_embedding.weight[modality_id].expand(n, -1)
        rel = self.relation_embedding.weight[relation_id].expand(n, -1)
        value = torch.cat(
            (current, first_neighbor, first_neighbor - current, mod, rel, cross_context), dim=-1
        )
        return self.net(value)


class JointViewContribution(nn.Module):
    """Shared Physical/Semantic contribution scales with a chunked wide input."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 32):
        super().__init__()
        self.input = nn.Linear(5 * hidden_dim, bottleneck_dim)
        self.output = nn.Linear(bottleneck_dim, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        h: torch.Tensor,
        d_physical: torch.Tensor,
        d_semantic: torch.Tensor,
    ) -> torch.Tensor:
        """Return [s_P, s_S] without allocating a [N, 5H] concatenation."""
        parts = (h, d_physical, d_semantic, (d_physical - d_semantic).abs(), d_physical * d_semantic)
        pre_activation = None
        offset = 0
        for part in parts:
            width = part.size(-1)
            contribution = F.linear(part, self.input.weight[:, offset:offset + width], bias=None)
            pre_activation = contribution if pre_activation is None else pre_activation + contribution
            offset += width
        if offset != self.input.in_features:
            raise ValueError(f"joint-view input has width {offset}; expected {self.input.in_features}")
        if self.input.bias is not None:
            pre_activation = pre_activation + self.input.bias
        strength_logits = self.output(F.gelu(pre_activation))
        return 2.0 * torch.sigmoid(strength_logits)


class Model(nn.Module):
    """Full MRAGE encoder with shared graph experts and adaptive relations."""

    requires_full_lp_sampler_depth = True
    requires_global_semantic_candidates = True
    supports_link_prediction = False
    supports_unimodal_aux_loss = True

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__()
        mc = cfg.model
        self.input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])
        if min(self.text_dim, self.visual_dim) <= 0 or self.text_dim + self.visual_dim != self.input_dim:
            raise ValueError("MRAGE requires one concatenated Text/Visual feature matrix")
        self.num_nodes = int(data_info.get("num_nodes", 0))
        # LP has no class head target; keep zero-width auxiliary heads there.
        # NC supplies its configured class count and follows the frozen V1.3-A path.
        self.num_classes = int(data_info.get("num_classes") or 0)
        raw_hidden_dim = mc.get("hidden_dim", 256)
        try:
            self.hidden_dim = int(raw_hidden_dim)
            is_integral_hidden_dim = float(raw_hidden_dim) == float(self.hidden_dim)
        except (TypeError, ValueError, OverflowError):
            self.hidden_dim = 0
            is_integral_hidden_dim = False
        self.num_experts = int(mc.get("num_experts", 4))
        self.top_k = int(mc.get("top_k", 2))
        self.dropout_p = float(mc.get("dropout", 0.2))
        self.eps = float(mc.get("eps", 1.0e-8))
        self.physical_chunk_size = int(mc.get("physical_chunk_size", 32768))
        self.semantic_chunk_size = int(mc.get("semantic_chunk_size", 32768))
        self.cross_edge_chunk_size = int(mc.get("cross_edge_chunk_size", 8192))
        self.attention_dim = int(mc.get("attention_dim", 32))
        self.late_cross_dropout = float(mc.get("late_cross_dropout", 0.15))
        self.same_node_enabled = bool(mc.get("same_node_enabled", True))
        self.eta = float(mc.get("expert_eta", 0.1))
        self.balance_weight = float(mc.get("balance_weight", 0.01))
        self.physical_hops = int(mc.get("physical_hops", 3))
        self.semantic_hops = int(mc.get("semantic_hops", 1))
        if (self.physical_hops, self.semantic_hops) != (3, 1):
            raise ValueError("MRAGE requires physical_hops=3 and semantic_hops=1")
        if isinstance(raw_hidden_dim, bool) or not is_integral_hidden_dim or self.hidden_dim < 1:
            raise ValueError("model.hidden_dim must be a positive integer")
        if (self.num_experts, self.top_k) != (4, 2):
            raise ValueError("MRAGE requires exactly four experts and Top-2 routing")
        if int(mc.get("semantic_knn_k", 8)) != 8:
            raise ValueError("MRAGE's validated V7B.1 semantic graph requires semantic_knn_k=8")

        # Preserve module creation order so default-model initialization remains
        # bitwise compatible with the frozen reference implementation.
        self.projectors = nn.ModuleDict({
            "text": _IntrinsicProjector(self.text_dim, self.hidden_dim, self.dropout_p),
            "visual": _IntrinsicProjector(self.visual_dim, self.hidden_dim, self.dropout_p),
        })
        self.cross_projections = nn.ModuleDict({
            "visual_to_text": nn.Sequential(
                nn.Linear(self.hidden_dim, 64, bias=False), nn.GELU(), nn.Linear(64, self.hidden_dim, bias=False)
            ),
            "text_to_visual": nn.Sequential(
                nn.Linear(self.hidden_dim, 64, bias=False), nn.GELU(), nn.Linear(64, self.hidden_dim, bias=False)
            ),
        })
        self.query_projections = nn.ModuleDict({m: nn.Linear(self.hidden_dim, self.attention_dim, bias=False) for m in _MODALITIES})
        self.key_projections = nn.ModuleDict({m: nn.Linear(self.hidden_dim, self.attention_dim, bias=False) for m in _MODALITIES})
        self.source_selector = nn.Sequential(
            nn.Linear(3 * self.hidden_dim + 3, 128), nn.GELU(), nn.Linear(128, 3)
        )
        nn.init.zeros_(self.source_selector[-1].weight)
        with torch.no_grad():
            self.source_selector[-1].bias.copy_(torch.log(torch.tensor([0.45, 0.35, 0.20])))
        self.channel_gate = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, int(mc.get("channel_gate_hidden", 128))),
            nn.GELU(), nn.Linear(int(mc.get("channel_gate_hidden", 128)), self.hidden_dim)
        )
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.constant_(self.channel_gate[-1].bias, math.log(0.73 / 0.27))
        self.cross_lambda_raw = nn.Parameter(torch.zeros(2))
        self.tau_raw = nn.Parameter(torch.full((2,), self._logit((0.25 - 0.1) / 0.7)))
        self.rho_raw = nn.Parameter(torch.full((2,), self._logit(float(mc.get("rho_init", 0.25)))))
        self.gamma_raw = nn.Parameter(torch.full((2,), self._logit(float(mc.get("gamma_init", 0.08)))))
        self.beta_raw = nn.Parameter(torch.full((2,), self._logit(float(mc.get("beta_init", 0.25)))))
        self.unimodal_heads = nn.ModuleDict({m: nn.Linear(self.hidden_dim, self.num_classes) for m in _MODALITIES})
        self.fusion_skip = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_linear1 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fusion_dropout = nn.Dropout(float(mc.get("fusion_dropout", 0.2)))
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)

        # Keep the formal MRAGE parameter creation order stable.
        with torch.random.fork_rng(devices=[]):
            self.filter_bank = SharedExpertBank(self.hidden_dim, int(mc.get("expert_bottleneck", 64)), self.dropout_p, self.eta)
            self.router = GraphAwareRouter(
                self.hidden_dim, int(mc.get("router_embed_dim", 8)),
                int(mc.get("router_hidden_dim", 256)), self.dropout_p
            )

        # Isolate the Joint View module's random first layer from the caller's
        # RNG stream; its zero output layer starts at unit scales.
        with torch.random.fork_rng(devices=[]):
            self.joint_view = JointViewContribution(self.hidden_dim, 32)

        self.register_buffer("semantic_source_flags_v7b", torch.empty((0, 2), dtype=torch.uint8), persistent=False)
        self.register_buffer("semantic_text_edge_index", torch.empty((2, 0), dtype=torch.long), persistent=False)
        self.register_buffer("semantic_text_weight", torch.empty((0,), dtype=torch.float32), persistent=False)
        self.register_buffer("semantic_visual_edge_index", torch.empty((2, 0), dtype=torch.long), persistent=False)
        self.register_buffer("semantic_visual_weight", torch.empty((0,), dtype=torch.float32), persistent=False)
        self.register_buffer("semantic_candidate_fingerprint_state", torch.zeros((32,), dtype=torch.uint8))
        self.register_buffer("semantic_feature_fingerprint_state", torch.zeros((32,), dtype=torch.uint8))
        self.register_buffer("semantic_graph_identity_state", torch.zeros((32,), dtype=torch.uint8))
        self.semantic_candidate_fingerprint: str | None = None
        self.semantic_feature_fingerprint: str | None = None
        self.semantic_graph_identity_fingerprint: str | None = None
        self.semantic_v7b_fingerprint: str | None = None
        self.semantic_cache_metadata: dict[str, Any] | None = None
        self.out_dim = self.hidden_dim
        # The sampled LP adapter sets these per mini-batch. They are runtime
        # context only and are intentionally excluded from checkpoints.
        self._batch_n_id: torch.Tensor | None = None
        self._sampled_semantic_edges: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None

    @staticmethod
    def _logit(value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("initial bounded scale must be between zero and one")
        return math.log(value / (1.0 - value))

    @property
    def rho(self) -> torch.Tensor:
        return torch.sigmoid(self.rho_raw)

    @property
    def gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.gamma_raw)

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_raw)

    @property
    def cross_lambda(self) -> torch.Tensor:
        return 0.5 * torch.sigmoid(self.cross_lambda_raw)

    @property
    def tau(self) -> torch.Tensor:
        return 0.1 + 0.7 * torch.sigmoid(self.tau_raw)

    def load_semantic_cache(
        self,
        path: str,
        *,
        expected_dataset: str | None = None,
        expected_feature_fingerprint: str | None = None,
        expected_candidate_fingerprint: str | None = None,
        expected_v7b_fingerprint: str | None = None,
        expected_identity_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        raw_cache = torch.load(path, map_location="cpu", weights_only=False)
        manager_identity = raw_cache.get("manager", {})
        identity = manager_identity.get("identity")
        identity_fingerprint = manager_identity.get("identity_fingerprint")
        if not isinstance(identity, dict) or not identity_fingerprint:
            raise ValueError("MRAGE semantic cache is missing its task/topology identity")
        if expected_identity_fingerprint is not None and identity_fingerprint != expected_identity_fingerprint:
            raise ValueError("semantic cache identity differs from the current task graph")
        cache = load_semantic_graph(
            path,
            expected_dataset=expected_dataset or str(identity["dataset"]),
            expected_identity=identity,
            expected_identity_fingerprint=str(identity_fingerprint),
            validate=True,
        )
        metadata = cache["metadata"]
        if expected_feature_fingerprint is not None and metadata.get("feature_fingerprint") != expected_feature_fingerprint:
            raise ValueError("semantic feature fingerprint mismatch")
        if expected_candidate_fingerprint is not None and metadata.get("candidate_fingerprint") != expected_candidate_fingerprint:
            raise ValueError("semantic candidate fingerprint mismatch")
        if expected_v7b_fingerprint is not None and metadata.get("v7b_fingerprint") != expected_v7b_fingerprint:
            raise ValueError("semantic V7B fingerprint mismatch")
        if self.num_nodes and self.num_nodes != int(metadata["num_nodes"]):
            raise ValueError("semantic cache node count differs from the model graph")
        if int(metadata["text_dim"]) != self.text_dim or int(metadata["visual_dim"]) != self.visual_dim:
            raise ValueError("semantic cache feature dimensions differ from model input")
        flags = cache["source_flags"].to(dtype=torch.uint8, device="cpu")
        unique = cache["unique_edge_index"].to(dtype=torch.long, device="cpu")
        if flags.shape != (unique.size(1), 2):
            raise ValueError("V7B source_flags do not align with unique edge IDs")
        operators = cache["operators"]
        self.semantic_source_flags_v7b = flags
        for modality in _MODALITIES:
            setattr(self, f"semantic_{modality}_edge_index", operators[modality]["edge_index"])
            setattr(self, f"semantic_{modality}_weight", operators[modality]["weight"])
        self.semantic_candidate_fingerprint = str(metadata["candidate_fingerprint"])
        self.semantic_feature_fingerprint = str(metadata["feature_fingerprint"])
        self.semantic_graph_identity_fingerprint = str(identity_fingerprint)
        self.semantic_v7b_fingerprint = str(metadata["v7b_fingerprint"])
        self.semantic_cache_metadata = metadata
        self.semantic_candidate_fingerprint_state.copy_(torch.tensor(list(bytes.fromhex(self.semantic_candidate_fingerprint)), dtype=torch.uint8))
        self.semantic_feature_fingerprint_state.copy_(torch.tensor(list(bytes.fromhex(self.semantic_feature_fingerprint)), dtype=torch.uint8))
        self.semantic_graph_identity_state.copy_(torch.tensor(list(bytes.fromhex(self.semantic_graph_identity_fingerprint)), dtype=torch.uint8))
        if self.num_nodes == 0:
            self.num_nodes = int(metadata["num_nodes"])
        return metadata

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        for name, expected_buffer, description in (
            ("semantic_candidate_fingerprint_state", self.semantic_candidate_fingerprint_state, "candidate graph"),
            ("semantic_feature_fingerprint_state", self.semantic_feature_fingerprint_state, "input features"),
            ("semantic_graph_identity_state", self.semantic_graph_identity_state, "task/topology identity"),
        ):
            saved = state_dict.get(name)
            if saved is not None:
                if not torch.equal(saved.detach().cpu(), expected_buffer.detach().cpu()):
                    raise RuntimeError(f"checkpoint semantic {description} fingerprint differs from the attached V7B.1 cache")
            elif self.semantic_candidate_fingerprint is not None and strict:
                raise RuntimeError(f"checkpoint is missing its semantic {description} fingerprint")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @staticmethod
    def _physical_operator(edge_index: torch.Tensor, n: int, dtype: torch.dtype):
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("physical edge_index must have shape [2,E]")
        loops = int((edge_index[0] == edge_index[1]).sum().item())
        edge_index, _ = remove_self_loops(edge_index.long())
        src, dst = edge_index
        degree = torch.zeros(n, dtype=dtype, device=edge_index.device)
        if dst.numel():
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=dtype))
        inv_sqrt = degree.clamp_min(1.0).rsqrt()
        weight = inv_sqrt[src] * inv_sqrt[dst]
        return edge_index, weight, degree > 0, loops

    @staticmethod
    def _support(edge_index: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        if edge_index.numel():
            mask[edge_index[1]] = True
        return mask

    def _basis(
        self,
        initial: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        support: torch.Tensor,
        max_order: int,
        chunk_size: int,
    ) -> list[torch.Tensor]:
        basis = [F.layer_norm(initial, (self.hidden_dim,))]
        for _ in range(max_order):
            propagated = chunked_weighted_spmm(basis[-1], edge_index, edge_weight, chunk_size)
            basis.append(propagated * support.to(propagated.dtype).unsqueeze(-1))
        return basis

    def _filter(
        self,
        basis: list[torch.Tensor],
        support: torch.Tensor,
        current: torch.Tensor,
        relation_id: int,
        modality_id: int,
        cross_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_order = len(basis) - 1
        if not (self.training and torch.is_grad_enabled()):
            outputs, filtered, coefficients, residual = self.filter_bank(basis, support, max_order)
            first = basis[1] if len(basis) > 1 else basis[0]
            logits = self.router(current, first, cross_context, modality_id, relation_id)
            dense = torch.softmax(logits.float(), dim=-1)
            top_logits, top_ids = torch.topk(logits, self.top_k, dim=-1)
            top_weights = torch.softmax(top_logits.float(), dim=-1).to(outputs.dtype)
            route_weights = torch.zeros_like(dense).scatter(1, top_ids, top_weights.float()).to(outputs.dtype)
            selected = torch.zeros_like(dense, dtype=torch.bool).scatter(1, top_ids, True)
            active = support
            if bool(active.any()):
                importance = dense[active].mean(dim=0)
                selected_share = selected[active].float().mean(dim=0) / float(self.top_k)
                balance = self.num_experts * (importance * selected_share).sum()
            else:
                importance = dense.sum(dim=0) * 0.0
                balance = dense.sum() * 0.0
            response = (outputs * route_weights.transpose(0, 1).unsqueeze(-1)).sum(dim=0)
            return response, balance

        def compute(*inputs):
            return self._m_filter_impl(
                list(inputs[:max_order + 1]), inputs[max_order + 1], inputs[max_order + 2],
                inputs[max_order + 3], modality_id, relation_id,
            )

        args = (*basis, support, current, cross_context)
        return checkpoint(compute, *args, use_reentrant=False, preserve_rng_state=True)

    def _m_filter_impl(
        self, basis: list[torch.Tensor], support: torch.Tensor, current: torch.Tensor,
        cross_context: torch.Tensor, modality_id: int, relation_id: int,
    ):
        """Compute one expert/router call without retaining a four-expert stack."""
        max_order = len(basis) - 1
        coefficients = self.filter_bank.coefficients(max_order, basis[0].dtype)
        first = basis[1] if len(basis) > 1 else basis[0]
        # Prepare the deterministic router input projection before expert
        # Dropout. The router Dropout is applied after all four experts to
        # preserve the established train-time RNG consumption order.
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
            raise RuntimeError("router input component widths do not match its configured linear layer")
        if first_layer.bias is not None:
            pre_activation = pre_activation + first_layer.bias
        router_hidden = F.gelu(pre_activation)
        mask = support.to(basis[0].dtype).unsqueeze(-1)
        outputs = []
        filtered_values = []
        for expert_id, expert in enumerate(self.filter_bank.experts):
            filtered = None
            for order in range(max_order + 1):
                term = coefficients[expert_id, order] * basis[order]
                filtered = term if filtered is None else filtered + term
            # The enclosing non-reentrant checkpoint recomputes this complete
            # expert/router boundary, so avoid nesting one checkpoint per MLP.
            residual = self.eta * expert(filtered)
            value = self.filter_bank.norms[expert_id](filtered + residual) * mask
            outputs.append(value)

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

        active = support
        if bool(active.any()):
            importance = dense[active].mean(dim=0)
            selected_share = selected[active].float().mean(dim=0) / float(self.top_k)
            balance = self.num_experts * (importance * selected_share).sum()
        else:
            balance = dense.sum() * 0.0
        return response, balance

    @staticmethod
    def _zero_context_fallback(private_response: torch.Tensor, collaborative_response: torch.Tensor,
                               context: torch.Tensor) -> torch.Tensor:
        nonzero = context.abs().sum(dim=-1) > 0
        return torch.where(nonzero[:, None], collaborative_response, private_response)

    @staticmethod
    def _feature_rms(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            value = value[mask]
        if value.numel() == 0:
            return value.new_zeros((), dtype=torch.float32)
        return value.detach().float().square().mean().sqrt()

    def _early_cross(
        self,
        source_state: torch.Tensor,
        projection: nn.Module,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        support: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        projected = projection(source_state)
        raw = chunked_weighted_spmm(projected, edge_index, edge_weight, self.physical_chunk_size)
        raw = raw * support.to(raw.dtype).unsqueeze(-1)
        raw_rms = self._feature_rms(raw, support)
        target_rms = self._feature_rms(reference, support)
        if float(raw_rms.cpu()) <= self.eps:
            return torch.zeros_like(raw)
        scale = (target_rms / (raw_rms + self.eps)).detach()
        return raw * scale.to(raw.dtype)

    @staticmethod
    def _sequential_from_parts(module: nn.Sequential, parts: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Evaluate a Sequential MLP without materializing its wide concatenated input."""
        first = module[0]
        if not isinstance(first, nn.Linear):
            return module(torch.cat(parts, dim=-1))
        offset = 0
        pre_activation = None
        for part in parts:
            width = part.size(-1)
            contribution = F.linear(part, first.weight[:, offset:offset + width], bias=None)
            pre_activation = contribution if pre_activation is None else pre_activation + contribution
            offset += width
        if offset != first.in_features:
            raise ValueError(f"MLP input parts have width {offset}; expected {first.in_features}")
        if first.bias is not None:
            pre_activation = pre_activation + first.bias
        value = pre_activation
        for layer in list(module.children())[1:]:
            value = layer(value)
        return value

    def _late(
        self,
        target: str,
        target_id: int,
        source: str,
        h_source: torch.Tensor,
        u_private_source: torch.Tensor,
        u_collaborative_target: torch.Tensor,
    ) -> torch.Tensor:
        projection_name = "visual_to_text" if source == "visual" else "text_to_visual"
        phi = self.cross_projections[projection_name]
        if self._sampled_semantic_edges is not None:
            edge_index = self._sampled_semantic_edges[source][0]
        else:
            edge_index = getattr(self, f"semantic_{source}_edge_index").to(u_collaborative_target.device)
        support_sem = self._support(edge_index, u_collaborative_target.size(0), u_collaborative_target.device)
        query = self.query_projections[target](F.layer_norm(u_collaborative_target, (self.hidden_dim,)))
        key = self.key_projections[source](F.layer_norm(u_private_source, (self.hidden_dim,)))
        value = phi(0.7 * h_source + 0.3 * u_private_source)
        c_sem, _ = receiver_sparse_attention(query, key, value, edge_index, self.cross_edge_chunk_size)
        c_sem = c_sem * support_sem.to(c_sem.dtype).unsqueeze(-1)
        if self.same_node_enabled:
            c_self = phi(u_private_source)
            support_self = torch.ones_like(support_sem)
        else:
            c_self = torch.zeros_like(c_sem)
            support_self = torch.zeros_like(support_sem)
        cosine = F.cosine_similarity(c_sem.detach(), c_self.detach(), dim=-1, eps=1.0e-8)
        selector_parts = (
            u_collaborative_target, c_sem, c_self, cosine[:, None],
            support_sem.to(c_sem.dtype)[:, None], support_self.to(c_sem.dtype)[:, None],
        )
        logits = self._sequential_from_parts(self.source_selector, selector_parts).clone()
        logits[~support_sem, 0] = -torch.inf
        logits[~support_self, 1] = -torch.inf
        source_weights = torch.softmax(logits, dim=-1)
        any_support = support_sem | support_self
        none_only = torch.zeros_like(source_weights)
        none_only[:, 2] = 1.0
        source_weights = torch.where(any_support[:, None], source_weights, none_only)
        p_accept = source_weights[:, 0] + source_weights[:, 1]
        candidate_cond = (
            source_weights[:, 0:1] * c_sem + source_weights[:, 1:2] * c_self
        ) / p_accept.clamp_min(1.0e-8)[:, None]
        candidate_cond = candidate_cond * any_support.to(candidate_cond.dtype).unsqueeze(-1)
        interaction_parts = (
            u_collaborative_target, candidate_cond,
            u_collaborative_target * candidate_cond,
            (u_collaborative_target - candidate_cond).abs(),
        )
        gate = torch.sigmoid(self._sequential_from_parts(self.channel_gate, interaction_parts))
        unbounded = self.cross_lambda[target_id] * p_accept[:, None] * gate * candidate_cond
        unbounded = unbounded * any_support.to(unbounded.dtype).unsqueeze(-1)
        ref_norm = u_collaborative_target.float().norm(dim=-1, keepdim=True)
        delta_norm = unbounded.float().norm(dim=-1, keepdim=True)
        scale = torch.minimum(torch.ones_like(delta_norm), self.tau[target_id] * ref_norm / (delta_norm + self.eps))
        controlled = unbounded * scale.to(unbounded.dtype)
        if self.training and self.late_cross_dropout > 0:
            keep = torch.rand((controlled.size(0), 1), device=controlled.device) >= self.late_cross_dropout
            controlled = controlled * keep.to(controlled.dtype)
        controlled = controlled * any_support.to(controlled.dtype).unsqueeze(-1)
        return u_collaborative_target + controlled

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None):
        if edge_index is None:
            raise ValueError("MRAGE requires a physical edge_index")
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected x shape [nodes,{self.input_dim}]")
        if self._batch_n_id is None:
            if self.num_nodes and x.size(0) != self.num_nodes:
                raise RuntimeError("MRAGE full-graph forward requires the cached global node order")
        elif x.size(0) != self._batch_n_id.numel():
            raise RuntimeError("MRAGE sampled features do not align with global batch node IDs")
        if self.semantic_candidate_fingerprint is None:
            raise RuntimeError("MRAGE requires a validated V7B.1 semantic cache")

        raw = {"text": x[:, :self.text_dim], "visual": x[:, self.text_dim:self.input_dim]}
        h = {modality: self.projectors[modality](raw[modality]) for modality in _MODALITIES}
        physical_edges, physical_weight, physical_support, _ = self._physical_operator(
            edge_index, x.size(0), h["text"].dtype
        )
        bases: dict[str, dict[str, list[torch.Tensor]]] = {modality: {} for modality in _MODALITIES}
        supports: dict[str, dict[str, torch.Tensor]] = {modality: {} for modality in _MODALITIES}
        for modality in _MODALITIES:
            bases[modality]["physical"] = self._basis(
                h[modality], physical_edges, physical_weight, physical_support,
                self.physical_hops, self.physical_chunk_size,
            )
            if self._sampled_semantic_edges is not None:
                semantic_edges, semantic_weight = self._sampled_semantic_edges[modality]
            else:
                semantic_edges = getattr(self, f"semantic_{modality}_edge_index").to(x.device)
                semantic_weight = getattr(self, f"semantic_{modality}_weight").to(
                    device=x.device, dtype=h[modality].dtype
                )
            semantic_support = self._support(semantic_edges, x.size(0), x.device)
            bases[modality]["semantic"] = self._basis(
                h[modality], semantic_edges, semantic_weight, semantic_support,
                self.semantic_hops, self.semantic_chunk_size,
            )
            supports[modality]["physical"] = physical_support
            supports[modality]["semantic"] = semantic_support

        private: dict[str, torch.Tensor] = {}
        private_strengths: dict[str, torch.Tensor] = {}
        responses: dict[str, dict[str, torch.Tensor]] = {modality: {} for modality in _MODALITIES}
        balances: list[torch.Tensor] = []
        paired_rng_states: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        zero_contexts = {modality: torch.zeros_like(h[modality]) for modality in _MODALITIES}
        for modality_id, modality in enumerate(_MODALITIES):
            for relation_id, relation in enumerate(("physical", "semantic")):
                if relation == "physical":
                    paired_rng_states[modality] = capture_rng_state(x.device)
                response, balance = self._filter(
                    bases[modality][relation], supports[modality][relation], h[modality],
                    relation_id, modality_id, zero_contexts[modality],
                )
                responses[modality][relation] = response
                balances.append(balance)
            physical_delta = self.rho[modality_id] * responses[modality]["physical"]
            semantic_delta = self.gamma[modality_id] * responses[modality]["semantic"]
            strengths = self.joint_view(h[modality], physical_delta, semantic_delta)
            private_strengths[modality] = strengths
            private[modality] = (
                h[modality]
                + strengths[:, 0:1] * physical_delta
                + strengths[:, 1:2] * semantic_delta
            )

        collaborative: dict[str, torch.Tensor] = {}
        for modality_id, modality in enumerate(_MODALITIES):
            other = "visual" if modality == "text" else "text"
            projection_name = "visual_to_text" if modality == "text" else "text_to_visual"
            cross_context = self._early_cross(
                h[other], self.cross_projections[projection_name], physical_edges,
                physical_weight, physical_support, bases[modality]["physical"][1],
            )
            collaborative_basis = list(bases[modality]["physical"])
            collaborative_basis[1] = (
                collaborative_basis[1] + self.beta[modality_id] * cross_context
            ) * physical_support.to(h[modality].dtype).unsqueeze(-1)
            with replay_rng_state(paired_rng_states[modality], x.device):
                collaborative_response, balance = self._filter(
                    collaborative_basis, physical_support, h[modality], 0,
                    modality_id, cross_context,
                )
            collaborative_response = self._zero_context_fallback(
                responses[modality]["physical"], collaborative_response, cross_context
            )
            collab_physical_delta = self.rho[modality_id] * collaborative_response
            collab_semantic_delta = self.gamma[modality_id] * responses[modality]["semantic"]
            collab_strengths = self.joint_view(h[modality], collab_physical_delta, collab_semantic_delta)
            collaborative[modality] = private[modality] + self.rho[modality_id] * (
                collab_strengths[:, 0:1] * collaborative_response
                - private_strengths[modality][:, 0:1] * responses[modality]["physical"]
            )
            balances.append(balance)

        final: dict[str, torch.Tensor] = {}
        for target_id, target in enumerate(_MODALITIES):
            source = "visual" if target == "text" else "text"
            final[target] = self._late(
                target, target_id, source, h[source], private[source], collaborative[target]
            )

        balance = torch.stack(balances).mean()
        aux_loss = self.balance_weight * balance
        fused = torch.cat((final["text"], final["visual"]), dim=-1)
        z = self.fusion_norm(
            self.fusion_skip(fused)
            + self.fusion_linear2(self.fusion_dropout(F.gelu(self.fusion_linear1(fused))))
        )
        info: dict[str, Any] = {
            "balance_loss": balance.detach(),
            "aux_loss": aux_loss.detach(),
            "unimodal_logits": {
                modality: self.unimodal_heads[modality](private[modality])
                for modality in _MODALITIES
            },
        }
        return z, None, None, aux_loss, info
