from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from src.models.factory import build_model
from src.models.predictor import LinkPredictor
from src.tasks.lp import _resolve_lp_num_neighbors


ROOT = Path(__file__).resolve().parents[1]
NC_MODELS = ("mlp", "gcn", "sage", "mmgcn", "mgat", "dip", "dgf", "dmgc", "rolemag", "lgmrec")
LP_MODELS = ("mlp", "gcn", "sage", "mmgcn", "mgat", "dgf", "dmgc")


def _model_cfg(name: str, task: str):
    model = OmegaConf.load(ROOT / "configs" / "model" / f"{name}.yaml")
    return OmegaConf.create({"model": model, "task": {"name": task, "num_neighbors": [2, 2, 2]}})


def _inputs():
    torch.manual_seed(411)
    n = 24
    text = torch.randn(n, 7)
    visual = torch.randn(n, 5)
    x = torch.cat((text, visual), dim=1)
    nodes = torch.arange(n)
    ring = torch.stack((nodes, (nodes + 1) % n))
    skip = torch.stack((nodes, (nodes + 4) % n))
    edge_index = torch.cat((ring, skip, ring.flip(0), skip.flip(0)), dim=1).contiguous()
    labels = nodes % 3
    info = {
        "input_dim": 12, "num_nodes": n, "num_classes": 3,
        "text_dim": 7, "visual_dim": 5,
    }
    return x, edge_index, labels, info


@pytest.mark.parametrize("name", NC_MODELS)
def test_baseline_nc_forward_loss_and_update(name: str) -> None:
    torch.set_num_threads(1)
    x, edge_index, labels, info = _inputs()
    model = build_model(_model_cfg(name, "nc"), info)
    model.train()
    z, _, _, aux_loss, _ = model(x, None if name == "mlp" else edge_index)
    assert z.shape == (x.size(0), model.out_dim)
    assert torch.isfinite(z).all()
    head = torch.nn.Linear(model.out_dim, 3)
    loss = F.cross_entropy(head(z[:16]), labels[:16]) + aux_loss
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
    torch.optim.Adam(model.parameters(), lr=1.0e-4).step()


@pytest.mark.parametrize("name", LP_MODELS)
def test_baseline_sampled_lp_pair_forward_loss_and_update(name: str) -> None:
    torch.set_num_threads(1)
    x, edge_index, _, info = _inputs()
    cfg = _model_cfg(name, "lp")
    model = build_model(cfg, info)

    edge_label_index = torch.tensor(
        [[0, 2, 0, 2], [1, 3, 7, 9]], dtype=torch.long
    )
    edge_label = torch.tensor([1.0, 1.0, 0.0, 0.0])
    if name == "mlp":
        batch_x = x[edge_label_index.reshape(-1)]
        z, _, _, aux_loss, _ = model(batch_x, None)
        z_pairs = z.view(edge_label_index.size(1), 2, -1)
        z_src, z_dst = z_pairs[:, 0], z_pairs[:, 1]
    else:
        task_cfg = OmegaConf.create(
            {"model": cfg.model, "task": {"num_neighbors": [2, 2, 2]}}
        )
        if name == "dgf":
            assert model.requires_full_lp_sampler_depth is True
            task_cfg.task.num_neighbors = [2, 2, 2]
            fanouts = _resolve_lp_num_neighbors(task_cfg, model)
            assert len(fanouts) == int(cfg.model.num_layers)
        else:
            fanouts = _resolve_lp_num_neighbors(task_cfg, model)
        loader = LinkNeighborLoader(
            Data(x=x, edge_index=edge_index),
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            num_neighbors=fanouts,
            batch_size=4,
            shuffle=False,
            subgraph_type="bidirectional",
            num_workers=0,
        )
        batch = next(iter(loader))
        if hasattr(model, "_batch_n_id"):
            model._batch_n_id = batch.n_id
        z, _, _, aux_loss, _ = model(batch.x, batch.edge_index)
        source, target = batch.edge_label_index
        z_src, z_dst = z[source], z[target]

    predictor = LinkPredictor(model.out_dim, 16, 2, 0.0)
    logits = predictor.score_pairs(z_src, z_dst)
    loss = F.binary_cross_entropy_with_logits(logits, edge_label) + aux_loss
    assert logits.shape == edge_label.shape
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=1.0e-4).step()

