from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
from torch_geometric.utils import add_self_loops, coalesce, degree, to_undirected


def _mms_loss(similarity: torch.Tensor, margin: float = 0.001) -> torch.Tensor:
    diagonal = margin * torch.eye(similarity.size(0), device=similarity.device)
    similarity = similarity - diagonal
    target = torch.arange(similarity.size(0), device=similarity.device)
    return F.cross_entropy(similarity, target) + F.cross_entropy(similarity.t(), target)


def _cluster_loss(
    embedding: torch.Tensor,
    centroids: torch.Tensor,
    labels: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    """OpenMAG DGF hard-positive community contrastive objective."""
    if embedding.numel() == 0 or centroids.numel() == 0:
        return embedding.new_tensor(0.0)
    total = embedding.new_tensor(0.0)
    for cluster_id in range(centroids.size(0)):
        mask = labels == cluster_id
        count = int(mask.sum().item())
        if count == 0:
            continue
        cluster_nodes = embedding[mask]
        positive_sim = cluster_nodes @ centroids[cluster_id]
        top_count = min(count, max(1, int(math.ceil(count * theta))))
        numerator = positive_sim.topk(top_count).values.exp().sum()
        denominator = (cluster_nodes @ centroids.t()).exp().sum()
        total = total - torch.log(numerator.clamp_min(1e-12) / denominator.clamp_min(1e-12))
    return total / max(embedding.size(0), 1)


def _window_walks(walks: torch.Tensor, context_size: int) -> torch.Tensor:
    count = walks.size(1) - context_size + 1
    if count <= 0:
        return walks[:, :0]
    return torch.cat([walks[:, start : start + context_size] for start in range(count)], dim=0)


def _graph_contrastive_loss(
    positive_walks: torch.Tensor,
    negative_walks: torch.Tensor,
    embedding: torch.Tensor,
) -> torch.Tensor:
    num_nodes, hidden_dim = embedding.shape
    positive_walks = positive_walks[(positive_walks < num_nodes).all(dim=1)]
    negative_walks = negative_walks[(negative_walks < num_nodes).all(dim=1)]
    if positive_walks.numel() == 0 or negative_walks.numel() == 0:
        return embedding.new_tensor(0.0)
    max_samples = 4096
    if positive_walks.size(0) > max_samples:
        positive_walks = positive_walks[torch.randperm(positive_walks.size(0), device=embedding.device)[:max_samples]]
    if negative_walks.size(0) > max_samples:
        negative_walks = negative_walks[torch.randperm(negative_walks.size(0), device=embedding.device)[:max_samples]]

    start_pos = positive_walks[:, 0]
    rest_pos = positive_walks[:, 1:].contiguous()
    pos_logits = (embedding[start_pos].unsqueeze(1) * embedding[rest_pos.flatten()].view(
        positive_walks.size(0), -1, hidden_dim
    )).sum(dim=-1)
    positive_score = torch.logsumexp(pos_logits, dim=-1)

    start_neg = negative_walks[:, 0]
    rest_neg = negative_walks[:, 1:].contiguous()
    neg_logits = (embedding[start_neg].unsqueeze(1) * embedding[rest_neg.flatten()].view(
        negative_walks.size(0), -1, hidden_dim
    )).sum(dim=-1)
    negative_score = torch.logsumexp(neg_logits, dim=-1)
    combined = torch.logsumexp(torch.stack((negative_score, positive_score), dim=-1), dim=-1)
    return -(positive_score - combined).mean()


class Model(nn.Module):
    """DGF filtering core with the three OpenMAG model objectives.

    Keeps the dual graph filtering core: L2-normalized per-modality
    projections, fixed mean fusion ZM = (ZM1 + ZM2) / 2, feature-domain
    symmetric-softmax shift operators, and the node-domain truncated Neumann
    series sum_{t} (alpha/(alpha+1) * NAM)^t ZM.

    Its cross-modal, random-walk graph, and community contrastive losses are
    added to the shared NC task loss. To keep the full-graph NC objective
    within memory, contrastive anchors use a random 1024-node subset, matching
    the mini-batch scale used by the reference training code.
    """

    supports_link_prediction = True
    lp_sampled_context = (
        "Ten DGF filter iterations run over the shared three-hop "
        "LinkNeighborLoader subgraph; they do not expand sampler depth."
    )
    lp_sampled_adapter_version = "fixed_three_hop_v1"

    def __init__(self, cfg, data_info):
        super().__init__()
        text_dim = int(data_info["text_dim"])
        visual_dim = int(data_info["visual_dim"])
        if text_dim <= 0 or visual_dim <= 0:
            raise ValueError(
                f"dgf requires both modalities, got text_dim={text_dim}, visual_dim={visual_dim}"
            )
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.hidden_dim = int(cfg.model.hidden_dim)
        self.alpha = float(cfg.model.get("alpha", 1.0))
        self.beta = float(cfg.model.get("beta", 1.0))
        # These are recurrent filter iterations over the sampled graph. They
        # do not increase LinkNeighborLoader's fixed three-hop fanout.
        self.num_layers = int(cfg.model.get("num_layers", 10))
        self.max_aux_nodes = int(cfg.model.get("max_aux_nodes", 1024))
        self.walks_per_node = int(cfg.model.get("walks_per_node", 10))
        self.walk_length = int(cfg.model.get("walk_length", 5))
        self.context_size = int(cfg.model.get("context_size", 3))
        self.num_clusters = int(cfg.model.get("num_clusters", 10))
        self.community_theta = float(cfg.model.get("community_theta", 0.3))
        self.aas = bool(cfg.model.get("aas", True))
        self.aas_std = float(cfg.model.get("aas_std", 1.0))
        self.lp_sampled_adapter_version = str(
            cfg.model.get("lp_sampled_adapter_version", "fixed_three_hop_v1")
        )
        # Official core: no activation, no dropout, no norm layers — L2 only.
        self.linear1 = nn.Linear(text_dim, self.hidden_dim)  # text
        self.linear2 = nn.Linear(visual_dim, self.hidden_dim)  # image
        self.out_dim = self.hidden_dim
        self._nam_cache: tuple | None = None
        self._rw_cache: tuple | None = None

    # ------------------------------------------------------------------
    # Feature-domain shift operator (OpenMAG DGFCore.symmetric_softmax)
    # ------------------------------------------------------------------

    def _symmetric_softmax(self, zm: torch.Tensor) -> torch.Tensor:
        n, _d = zm.shape
        scale = 1.0 / math.sqrt(n)
        similarity = zm.t() @ zm * scale
        exp_sim = torch.exp(similarity - similarity.max())  # numerical stability
        row_sum = torch.sqrt(exp_sim.sum(dim=1, keepdim=True) + 1e-10)
        col_sum = torch.sqrt(exp_sim.sum(dim=0, keepdim=True) + 1e-10)
        return exp_sim / (row_sum @ col_sum)

    # ------------------------------------------------------------------
    # Node-domain normalized adjacency (self-loops added: OpenMAG NC
    # loader uses self_loop=True; RPTA protocol allows standard conv
    # self-loops).
    # ------------------------------------------------------------------

    def _build_nam(
        self, edge_index: torch.Tensor, num_nodes: int, device: torch.device
    ) -> torch.Tensor:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)[0]
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=torch.float32)
        inv_sqrt = (deg + 1e-10).pow(-0.5)
        values = inv_sqrt[row] * inv_sqrt[col]
        return torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).to(device)

    def _get_nam(
        self, edge_index: torch.Tensor, num_nodes: int, device: torch.device
    ) -> torch.Tensor:
        # Identity-keyed cache: full-graph training reuses the same tensors
        # every epoch; LinkNeighborLoader subgraphs get fresh tensors per
        # batch and are rebuilt (their edge_index genuinely differs).
        cache = self._nam_cache
        if cache is not None and cache[0] is edge_index and cache[1] == num_nodes:
            return cache[2]
        nam = self._build_nam(edge_index, num_nodes, device)
        self._nam_cache = (edge_index, num_nodes, nam)
        return nam

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        device = x.device
        num_nodes = int(x.size(0))
        x_t = x[:, : self.text_dim]
        x_v = x[:, self.text_dim : self.text_dim + self.visual_dim]

        zm1 = F.normalize(self.linear1(x_t), p=2, dim=1)
        zm2 = F.normalize(self.linear2(x_v), p=2, dim=1)
        zm = 0.5 * (zm1 + zm2)

        with torch.no_grad():
            sm = 0.5 * (self._symmetric_softmax(zm1) + self._symmetric_softmax(zm2))
            beta_term = self.beta / (self.beta + 1.0)
            d = zm.size(1)
            sum_beta = torch.zeros(d, d, device=device)
            current_power = torch.eye(d, device=device)
            for _ in range(self.num_layers):
                sum_beta = sum_beta + current_power
                current_power = current_power @ (beta_term * sm)

        nam = self._get_nam(edge_index, num_nodes, device)
        alpha_term = self.alpha / (self.alpha + 1.0)
        part_alpha = zm.clone()
        current = zm.clone()
        for _ in range(self.num_layers):
            current = alpha_term * torch.sparse.mm(nam, current)
            part_alpha = part_alpha + current

        h = (1.0 / ((self.alpha + 1.0) * (self.beta + 1.0))) * (part_alpha @ sum_beta)
        return F.normalize(h, p=2, dim=1)

    def _get_rw_adj(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        cached = self._rw_cache
        if cached is not None and cached[0] is edge_index and cached[1] == num_nodes:
            return cached[2]
        undirected = coalesce(to_undirected(edge_index, num_nodes=num_nodes), num_nodes=num_nodes)
        values = torch.ones(undirected.size(1), dtype=torch.float32, device=edge_index.device)
        adj = torch.sparse_coo_tensor(undirected, values, (num_nodes, num_nodes)).coalesce()
        # torch_cluster's CSR random-walk kernel expects row pointers and columns.
        csr_cpu = adj.to_sparse_csr().cpu()
        self._rw_cache = (edge_index, num_nodes, adj, csr_cpu)
        return adj

    def _sampled_kmeans(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        points = F.normalize(embedding.detach(), p=2, dim=1)
        k = min(max(self.num_clusters, 1), points.size(0))
        centers = points[torch.randperm(points.size(0), device=points.device)[:k]].clone()
        for _ in range(8):
            labels = (points @ centers.t()).argmax(dim=1)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, labels, points)
            counts = torch.bincount(labels, minlength=k).to(points.dtype).unsqueeze(1)
            updated = sums / counts.clamp_min(1.0)
            centers = F.normalize(torch.where(counts > 0, updated, centers), p=2, dim=1)
        labels = (points @ centers.t()).argmax(dim=1)
        return centers.detach(), labels.detach()

    def _dgf_auxiliary_loss(
        self,
        z_text: torch.Tensor,
        z_visual: torch.Tensor,
        h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        num_nodes = h.size(0)
        sample_count = min(max(2, self.max_aux_nodes), num_nodes)
        sample_idx = torch.randperm(num_nodes, device=h.device)[:sample_count]
        z1 = z_text[sample_idx]
        z2 = z_visual[sample_idx]
        hs = h[sample_idx]

        loss_mod = (
            _mms_loss(z1 @ z2.t())
            + _mms_loss(z1 @ hs.t())
            + _mms_loss(z2 @ hs.t())
        )

        adj = self._get_rw_adj(edge_index, num_nodes)
        if self.aas and adj._nnz() > 0:
            row, col = adj.indices()
            norm1, norm2 = F.normalize(z_text, dim=1), F.normalize(z_visual, dim=1)
            edge_similarity = (norm1[row] * norm2[col]).sum(dim=1)
            pair_count = row.numel()
            random_row = torch.randint(num_nodes, (pair_count,), device=h.device)
            random_col = torch.randint(num_nodes, (pair_count,), device=h.device)
            random_similarity = (norm1[random_row] * norm2[random_col]).sum(dim=1)
            threshold = random_similarity.mean() + self.aas_std * random_similarity.std()
            keep = edge_similarity >= threshold
            if bool(keep.any()):
                filtered = torch.sparse_coo_tensor(
                    torch.stack((row[keep], col[keep])),
                    torch.ones(int(keep.sum().item()), device=h.device),
                    (num_nodes, num_nodes),
                ).coalesce()
            else:
                filtered = adj
        else:
            filtered = adj

        walk_csr = self._rw_cache[3] if filtered is adj else filtered.to_sparse_csr().cpu()
        starts = sample_idx.repeat(self.walks_per_node)
        positive_raw = torch.ops.torch_cluster.random_walk(
            walk_csr.crow_indices().to(h.device),
            walk_csr.col_indices().to(h.device),
            starts,
            self.walk_length,
            1.0,
            1.0,
        )
        if isinstance(positive_raw, (tuple, list)):
            positive_raw = positive_raw[0]
        positive_walks = _window_walks(positive_raw, self.context_size)
        negative_raw = torch.cat(
            (
                starts.unsqueeze(1),
                torch.randint(num_nodes, (starts.numel(), self.walk_length), device=h.device),
            ),
            dim=1,
        )
        negative_walks = _window_walks(negative_raw, self.context_size)
        loss_nbr = _graph_contrastive_loss(positive_walks, negative_walks, h)

        centers, labels = self._sampled_kmeans(hs)
        loss_comm = _cluster_loss(F.normalize(hs, dim=1), centers, labels, self.community_theta)

        # OpenMAG normalizes each term's gradient while retaining the raw sum
        # as the logged value, so each active objective gets equal gradient scale.
        eps = 1e-8
        raw_loss = loss_mod + loss_nbr + loss_comm
        normalized = loss_mod / (loss_mod.detach() + eps) + loss_nbr / (loss_nbr.detach() + eps)
        if bool(loss_comm.detach() > eps):
            normalized = normalized + loss_comm / (loss_comm.detach() + eps)
        total = raw_loss.detach() + (normalized - normalized.detach())
        return total, {"cross_modal": loss_mod.detach(), "graph": loss_nbr.detach(), "community": loss_comm.detach()}

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index=None):
        if edge_index is None:
            raise ValueError("DGF requires edge_index")
        z = self._encode(x, edge_index)
        if self.training:
            x_t = x[:, : self.text_dim]
            x_v = x[:, self.text_dim : self.text_dim + self.visual_dim]
            z_text = F.normalize(self.linear1(x_t), p=2, dim=1)
            z_visual = F.normalize(self.linear2(x_v), p=2, dim=1)
            aux_loss, aux_info = self._dgf_auxiliary_loss(z_text, z_visual, z, edge_index)
        else:
            aux_loss, aux_info = z.new_tensor(0.0), {}
        return z, None, None, aux_loss, {"dgf_losses": aux_info}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """Chunked layerwise-style eval: per-node projections on device, then
        full-graph filtering on CPU (identical math to a full-graph forward)."""
        self.eval()
        if edge_index is None:
            raise ValueError("DGF requires edge_index")
        if device is None:
            device = next(self.parameters()).device
        num_nodes = int(x.size(0))
        zm1 = torch.empty((num_nodes, self.hidden_dim), dtype=x.dtype)
        zm2 = torch.empty_like(zm1)
        for start in range(0, num_nodes, batch_size):
            end = min(start + batch_size, num_nodes)
            chunk = x[start:end].to(device)
            zm1[start:end] = F.normalize(self.linear1(chunk[:, : self.text_dim]), p=2, dim=1).cpu()
            zm2[start:end] = F.normalize(
                self.linear2(chunk[:, self.text_dim : self.text_dim + self.visual_dim]), p=2, dim=1
            ).cpu()
        zm = 0.5 * (zm1 + zm2)
        sm = 0.5 * (self._symmetric_softmax(zm1) + self._symmetric_softmax(zm2))
        beta_term = self.beta / (self.beta + 1.0)
        d = zm.size(1)
        sum_beta = torch.zeros(d, d)
        current_power = torch.eye(d)
        for _ in range(self.num_layers):
            sum_beta = sum_beta + current_power
            current_power = current_power @ (beta_term * sm)
        nam = self._build_nam(edge_index.cpu(), num_nodes, torch.device("cpu"))
        alpha_term = self.alpha / (self.alpha + 1.0)
        part_alpha = zm.clone()
        current = zm.clone()
        for _ in range(self.num_layers):
            current = alpha_term * torch.sparse.mm(nam, current)
            part_alpha = part_alpha + current
        h = (1.0 / ((self.alpha + 1.0) * (self.beta + 1.0))) * (part_alpha @ sum_beta)
        return F.normalize(h, p=2, dim=1)
