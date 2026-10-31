from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops, degree


def _softplus_inverse(value: float) -> float:
    value = max(float(value), 1.0e-4)
    return math.log(math.expm1(value))


def _tan_k(value: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    scale = curvature.abs().clamp_min(1.0e-12).sqrt()
    angle = scale * value
    positive = torch.tan(angle.clamp(max=math.pi / 2 - 1.0e-4)) / scale
    negative = torch.tanh(angle) / scale
    curved = torch.where(curvature > 0, positive, negative)
    return torch.where(curvature.abs() < 1.0e-10, value, curved)


def _atan_k(value: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    scale = curvature.abs().clamp_min(1.0e-12).sqrt()
    argument = (scale * value).clamp(min=-1.0 + 1.0e-6, max=1.0 - 1.0e-6)
    positive = torch.atan(scale * value) / scale
    negative = torch.atanh(argument) / scale
    curved = torch.where(curvature > 0, positive, negative)
    return torch.where(curvature.abs() < 1.0e-10, value, curved)


class _Curvature(nn.Module):
    """A signed, trainable stereographic curvature parameter."""

    def __init__(self, initial: float, learnable: bool = True):
        super().__init__()
        self.initial = float(initial)
        self.sign = 1.0 if initial > 0 else -1.0
        if initial == 0:
            self.register_parameter("raw_magnitude", None)
        elif learnable:
            self.raw_magnitude = nn.Parameter(
                torch.tensor(_softplus_inverse(abs(initial)), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "raw_magnitude",
                torch.tensor(_softplus_inverse(abs(initial)), dtype=torch.float32),
            )

    def value(self, reference: torch.Tensor) -> torch.Tensor:
        if self.raw_magnitude is None:
            return reference.new_zeros(())
        return self.sign * F.softplus(self.raw_magnitude.to(reference))


def _project(x: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    max_norm = (1.0 - 1.0e-5) / curvature.abs().clamp_min(1.0e-12).sqrt()
    scale = torch.clamp(max_norm / norm, max=1.0)
    scale = torch.where(curvature < 0, scale, torch.ones_like(scale))
    return x * scale


def _expmap0(v: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    k = curvature.to(v)
    out = (_tan_k(norm, k) / norm) * v
    return _project(out, k)


def _logmap0(y: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    norm = y.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    k = curvature.to(y)
    return (_atan_k(norm, k) / norm) * y


def _mobius_add(x: torch.Tensor, y: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    k = curvature.to(x)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    numerator = (1.0 - 2.0 * k * xy - k * y2) * x + (1.0 + k * x2) * y
    denominator = 1.0 - 2.0 * k * xy + k.square() * x2 * y2
    result = numerator / denominator.clamp_min(1.0e-8)
    return _project(result, k)


class _KappaLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, curvature: _Curvature):
        super().__init__()
        self.curvature = curvature
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.curvature.value(x)
        tangent = _logmap0(x, k)
        transformed = F.linear(tangent, F.dropout(self.weight, p=0.0, training=self.training))
        mapped = _expmap0(transformed, k)
        bias = _expmap0(self.bias.expand_as(mapped), k)
        return _mobius_add(mapped, bias, k)


class _KappaGCNConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, curvature: _Curvature):
        super().__init__()
        self.curvature = curvature
        self.linear = _KappaLinear(in_dim, out_dim, self.curvature)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        k = self.curvature.value(x)
        x = self.linear(x)
        tangent = _logmap0(x, k)
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=int(x.size(0)))
        src, dst = edge_index
        deg = degree(dst, int(x.size(0)), dtype=x.dtype).clamp_min(1.0)
        inv_sqrt = deg.rsqrt()
        weight = inv_sqrt[src] * inv_sqrt[dst]
        aggregated = torch.zeros_like(tangent)
        aggregated.index_add_(0, dst, tangent[src] * weight.unsqueeze(-1))
        return _expmap0(aggregated, k)


class _RiemannianExpert(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, curvature: float):
        super().__init__()
        # The reference implementation shares one stereographic manifold and
        # its learnable curvature across the input map and both graph layers.
        self.curvature = _Curvature(curvature, learnable=curvature != 0)
        self.conv1 = _KappaGCNConv(in_dim, hidden_dim, self.curvature)
        self.conv2 = _KappaGCNConv(hidden_dim, out_dim, self.curvature)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        k = self.curvature.value(x)
        h = _expmap0(x, k)
        h = self.conv1(h, edge_index)
        h = self.conv2(h, edge_index)
        return _logmap0(h, k)


def _random_walk_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    src, dst = edge_index
    deg = degree(dst, num_nodes, dtype=torch.float32).clamp_min(1.0)
    values = deg[dst].reciprocal()
    indices = torch.stack((dst, src), dim=0)
    return torch.sparse_coo_tensor(
        indices, values, (num_nodes, num_nodes), device=edge_index.device
    ).coalesce()


class Model(nn.Module):
    """GraphMoRE encoder adapted to the project's shared NC/LP task heads.

    The original all-node NetworkX ego-graph expansion is replaced by sparse
    multi-scale neighborhood pooling. Curvature experts, topology-aware
    routing, pair-aligned expert weights, input skip features, expert
    LayerNorm, and the native distortion objective are retained. LP uses the
    shared three-hop LinkNeighborLoader subgraph.
    """

    supports_link_prediction = True
    lp_sampled_adapter_version = "sampled_subgraph_multiscale_v1"
    lp_sampled_context = (
        "Riemannian experts, multi-scale gate contexts, and distortion are "
        "computed on the current three-hop LinkNeighborLoader subgraph."
    )

    def __init__(self, cfg, data_info: dict):
        super().__init__()
        input_dim = int(data_info["input_dim"])
        hidden_dim = int(cfg.model.get("hidden_dim", 32))
        embed_dim = int(cfg.model.get("embed_dim", 32))
        curvatures = [float(value) for value in cfg.model.get("curvatures", [-3, -1, 0, 1, 3])]
        hops = [int(value) for value in cfg.model.get("local_hops", [2, 3])]
        if not curvatures or 0 not in curvatures:
            raise ValueError("graphmore.curvatures must include at least one expert and curvature 0")
        if not hops or any(value < 1 for value in hops):
            raise ValueError("graphmore.local_hops must contain positive hop counts")
        self.embed_dim = embed_dim
        self.curvatures = curvatures
        self.local_hops = hops
        self.adapter_revision = str(cfg.model.get("adapter_revision", "graphmore_shared_curvature_skip_v2"))
        task_name = str(cfg.task.get("name", "nc")).strip().lower()
        weight_key = "distortion_weight_lp" if task_name == "lp" else "distortion_weight_nc"
        self.distortion_weight = float(cfg.model.get(weight_key, 1.0e-4))
        self.max_distortion_edges = int(cfg.model.get("max_distortion_edges", 8192))
        expert_dim = len(curvatures) * embed_dim
        self.out_dim = input_dim + expert_dim
        self.experts = nn.ModuleList(
            [_RiemannianExpert(input_dim, hidden_dim, embed_dim, value) for value in curvatures]
        )
        self.expert_norm = nn.LayerNorm(expert_dim)
        self.gate_encoders = nn.ModuleList(
            nn.ModuleList(
                [GCNConv(input_dim, hidden_dim), GCNConv(hidden_dim, embed_dim)]
            )
            for _ in hops
        )
        self.gate_classifier = nn.Linear(embed_dim * len(hops), len(curvatures))

    def _gate(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = int(x.size(0))
        walk = _random_walk_adjacency(edge_index, num_nodes)
        scale_outputs = []
        for hop, layers in zip(self.local_hops, self.gate_encoders):
            h = layers[0](x, edge_index)
            h = layers[1](h, edge_index)
            pooled = h
            current = h
            for _ in range(hop):
                current = torch.sparse.mm(walk, current)
                pooled = pooled + current
            scale_outputs.append(pooled / float(hop + 1))
        return F.softmax(self.gate_classifier(torch.cat(scale_outputs, dim=-1)), dim=-1)

    def _distortion(
        self,
        expert_z: torch.Tensor,
        gate: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return expert_z.new_tensor(0.0)
        if edge_index.size(1) > self.max_distortion_edges:
            keep = torch.randperm(edge_index.size(1), device=edge_index.device)[: self.max_distortion_edges]
            edge_index = edge_index[:, keep]
        src, dst = edge_index
        pair_weights = F.softmax(gate[src] * gate[dst], dim=-1)
        differences = (expert_z[src] - expert_z[dst]).square().sum(dim=-1)
        mixed_distance = (differences * pair_weights).sum(dim=-1)
        return (mixed_distance - 1.0).abs().mean()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None):
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        expert_z = torch.stack([expert(x, edge_index) for expert in self.experts], dim=1)
        gate = self._gate(x, edge_index)
        distortion = self._distortion(expert_z, gate, edge_index)
        expert_z = self.expert_norm(expert_z.reshape(x.size(0), -1))
        expert_z = expert_z.reshape(x.size(0), len(self.curvatures), self.embed_dim)
        weighted_experts = (expert_z * gate.unsqueeze(-1)).reshape(x.size(0), -1)
        # GraphMoRE concatenates the raw node features with its gated expert
        # representation before the task-specific predictor. The project
        # replaces the reference two-layer GNN classifier with its shared
        # linear benchmark head, while retaining this input skip feature.
        z = torch.cat((x, weighted_experts), dim=-1)
        aux_loss = self.distortion_weight * distortion
        return z, None, None, aux_loss, {
            "graphmore_losses": {"distortion": distortion.detach()},
            "graphmore_adapter_revision": self.adapter_revision,
        }

    def forward_lp_sampled(self, x: torch.Tensor, edge_index: torch.Tensor):
        if edge_index is None:
            raise ValueError("GraphMoRE sampled LP requires a message subgraph")
        return self.forward(x, edge_index)

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        del batch_size
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        edge_index = (
            torch.empty((2, 0), dtype=torch.long, device=device)
            if edge_index is None else edge_index.to(device)
        )
        z, _, _, _, _ = self.forward(x.to(device), edge_index)
        return z.detach().cpu()
