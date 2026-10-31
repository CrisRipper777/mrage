from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class _MoEInputProjection(nn.Module):
    """Top-1 MoE feature projection with the paper's load-balancing term."""

    def __init__(self, input_dim: int, hidden_dim: int, num_tokens: int, dropout: float):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.num_experts = self.num_tokens
        self.hidden_dim = int(hidden_dim)
        self.router = nn.Linear(input_dim + self.num_tokens, self.num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.PReLU(hidden_dim),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(self.num_experts)
            ]
        )
        self.shared_expert = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.aux_loss_alpha = 0.1

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor):
        batch, length, feature_dim = x.shape
        flat = x.reshape(-1, feature_dim)
        route_ids = token_ids.view(1, length).expand(batch, -1).reshape(-1)
        route_one_hot = F.one_hot(route_ids, num_classes=self.num_tokens).to(flat.dtype)
        scores = F.softmax(self.router(torch.cat((flat, route_one_hot), dim=-1)), dim=-1)
        choices = scores.argmax(dim=-1)
        routed = flat.new_zeros((flat.size(0), self.hidden_dim))
        for expert_id, expert in enumerate(self.experts):
            mask = choices == expert_id
            routed[mask] = expert(flat[mask])
        output = routed + self.shared_expert(flat)
        if self.training and flat.size(0) > 1:
            load = F.one_hot(choices, num_classes=self.num_experts).to(flat.dtype).mean(dim=0)
            importance = scores.mean(dim=0)
            balance = (load * importance).sum() * self.num_experts * self.aux_loss_alpha
        else:
            balance = flat.new_tensor(0.0)
        return output.view(batch, length, -1), balance


class Model(nn.Module):
    """NTSFormer adapted to ordinary transductive NC and shared task heads.

    SIGN feature propagation is precomputed once on the complete training
    graph. The transformer retains the self-only student token and two
    neighbor-aware teacher tokens. A shared external linear classifier is
    used for all three views; teacher predictions train the task and the
    student is trained with the native teacher-to-student KL objective.
    """

    supports_link_prediction = False
    supports_nc_node_batches = True
    requires_batched_inference = True
    nc_training_adapter_version = "ntsformer_full_graph_sign_node_batch_v2"

    def __init__(self, cfg, data_info: dict):
        super().__init__()
        self.input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info.get("text_dim", 0) or 0)
        self.visual_dim = int(data_info.get("visual_dim", 0) or 0)
        if self.text_dim <= 0 or self.visual_dim <= 0 or self.text_dim + self.visual_dim != self.input_dim:
            raise ValueError(
                "ntsformer requires the project's concatenated [text, visual] features; "
                f"got input={self.input_dim}, text={self.text_dim}, visual={self.visual_dim}"
            )
        self.pre_k = int(cfg.model.get("pre_k", 2))
        self.hidden_dim = int(cfg.model.get("hidden_dim", 128))
        self.num_heads = int(cfg.model.get("num_heads", 2))
        self.num_layers = int(cfg.model.get("num_layers", 1))
        self.dropout_rate = float(cfg.model.get("dropout", 0.1))
        self.node_batch_size = int(cfg.model.get("node_batch_size", 8192))
        self.self_teaching_weight = float(cfg.model.get("self_teaching_weight", 1.0))
        self.max_modality_dim = max(self.text_dim, self.visual_dim)
        self.num_feature_tokens = 2 * (self.pre_k + 1)
        self.num_tokens = self.num_feature_tokens + 3
        self.out_dim = self.hidden_dim
        self.nc_auxiliary_scope = "all_nodes"
        self._sign_hops: list[torch.Tensor] | None = None
        self._cached_num_nodes: int | None = None

        self.input_projection = _MoEInputProjection(
            self.max_modality_dim,
            self.hidden_dim,
            self.num_feature_tokens,
            self.dropout_rate,
        )
        self.student_cls = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * self.hidden_dim**-0.5)
        self.teacher_cls = nn.Parameter(torch.randn(1, 2, self.hidden_dim) * self.hidden_dim**-0.5)
        self.position = nn.Parameter(torch.randn(1, self.num_tokens, self.hidden_dim) * self.hidden_dim**-0.5)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=4 * self.hidden_dim,
            dropout=self.dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(self.num_tokens, self.num_tokens, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def _prepare_sign_hops(self, x: torch.Tensor, edge_index: torch.Tensor) -> None:
        num_nodes = int(x.size(0))
        if self._sign_hops is not None and self._cached_num_nodes == num_nodes:
            return
        x_cpu = x.detach().to(device="cpu", dtype=torch.float32).contiguous()
        edge_cpu = edge_index.detach().to(device="cpu", dtype=torch.long).contiguous()
        norm_edge, norm_weight = gcn_norm(
            edge_cpu,
            edge_weight=None,
            num_nodes=num_nodes,
            improved=False,
            add_self_loops=True,
            flow="source_to_target",
            dtype=torch.float32,
        )
        src, dst = norm_edge
        adjacency = torch.sparse_coo_tensor(
            torch.stack((dst, src), dim=0),
            norm_weight,
            (num_nodes, num_nodes),
            dtype=torch.float32,
        ).coalesce()
        hops = [x_cpu.to(torch.float16)]
        current = x_cpu
        for _ in range(self.pre_k):
            current = torch.sparse.mm(adjacency, current)
            hops.append(current.to(torch.float16))
        self._sign_hops = hops
        self._cached_num_nodes = num_nodes

    def _batch_tokens(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._prepare_sign_hops(x, edge_index)
        assert self._sign_hops is not None
        ids_cpu = node_ids.detach().to(device="cpu", dtype=torch.long)
        tokens = []
        for hop_features in self._sign_hops:
            selected = hop_features[ids_cpu].to(device=x.device, dtype=x.dtype)
            text = F.pad(selected[:, : self.text_dim], (0, self.max_modality_dim - self.text_dim))
            visual = F.pad(
                selected[:, self.text_dim : self.text_dim + self.visual_dim],
                (0, self.max_modality_dim - self.visual_dim),
            )
            tokens.extend((text, visual))

        # MoE-project only real modality/hop tokens. Learned student/teacher
        # CLS tokens are inserted after projection, as in the reference model.
        raw = torch.stack(tokens, dim=1)
        projected_features, moe_loss = self.input_projection(
            raw,
            torch.arange(self.num_feature_tokens, device=x.device, dtype=torch.long),
        )
        projected = torch.cat(
            (
                projected_features[:, :2],
                self.student_cls.expand(node_ids.numel(), -1, -1),
                projected_features[:, 2:],
                self.teacher_cls.expand(node_ids.numel(), -1, -1),
            ),
            dim=1,
        )
        return projected + self.position, moe_loss

    def forward_node_batch(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_ids: torch.Tensor,
    ):
        tokens, moe_loss = self._batch_tokens(x, edge_index, node_ids)
        hidden = self.transformer(tokens, mask=self.causal_mask)
        student = hidden[:, 2]
        teacher_1, teacher_2 = hidden[:, -2], hidden[:, -1]
        z = 0.5 * (teacher_1 + teacher_2)
        aux_loss = moe_loss
        info = {
            "ntsformer_losses": {"moe_balance": moe_loss.detach()},
            "ntsformer_views": {
                "student": student,
                "teacher_1": teacher_1,
                "teacher_2": teacher_2,
            },
        }
        return z, None, None, aux_loss, info

    def additional_nc_loss(self, classifier: nn.Module, info: dict) -> torch.Tensor:
        views = info.get("ntsformer_views", {}) if isinstance(info, dict) else {}
        if not {"student", "teacher_1", "teacher_2"}.issubset(views):
            raise RuntimeError("NTSFormer training forward omitted its self-teaching views")
        student_logits = classifier(views["student"])
        teacher_prob = 0.5 * (
            F.softmax(classifier(views["teacher_1"]), dim=-1)
            + F.softmax(classifier(views["teacher_2"]), dim=-1)
        )
        return self.self_teaching_weight * F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            teacher_prob.detach(),
            reduction="batchmean",
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None):
        if edge_index is None:
            raise ValueError("NTSFormer requires the full graph for SIGN precomputation")
        node_ids = torch.arange(x.size(0), device=x.device)
        return self.forward_node_batch(x, edge_index, node_ids)

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 2048,
    ) -> torch.Tensor:
        if edge_index is None:
            raise ValueError("NTSFormer inference requires the full graph")
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        self._prepare_sign_hops(x, edge_index)
        out = torch.empty((x.size(0), self.out_dim), dtype=torch.float32, device="cpu")
        effective_batch_size = max(int(batch_size), self.node_batch_size)
        for start in range(0, int(x.size(0)), max(effective_batch_size, 1)):
            ids = torch.arange(start, min(start + effective_batch_size, int(x.size(0))), device=device)
            z, _, _, _, _ = self.forward_node_batch(x, edge_index, ids)
            out[ids.cpu()] = z.detach().float().cpu()
        return out
