from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader


def _tensor_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _parameter_grads(module):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in module.named_parameters()
    }


def _build_models(side: str, project_root: Path, source_root: Path, payload):
    if side == "reference":
        sys.path.insert(0, str(source_root))
        from src.models.mrcge_final import Model as NCModel
        from src.models.mrcge_final_sampled import Model as LPModel
        from src.models.predictor import LinkPredictor
    else:
        sys.path.insert(0, str(project_root))
        from src.models.mrage import Model as NCModel
        from src.models.mrage_lp import Model as LPModel
        from src.models.predictor import LinkPredictor
    model_config = OmegaConf.to_container(
        OmegaConf.load(project_root / "configs/model/mrage.yaml"), resolve=True
    )
    if side == "reference":
        model_config.update(
            name="mrcge_final",
            condition="M_repaired",
            v13_variant="joint_view",
            sender_private_ratio=0.0,
        )
    task = {"name": "nc", "loss": {"aux_weight": 1.0, "unimodal_weight": 0.3}}
    cfg = OmegaConf.create({"model": model_config, "task": task})
    return NCModel, LPModel, LinkPredictor, cfg


def run(side: str, project_root: Path, source_root: Path, input_path: Path, output_path: Path):
    torch.set_num_threads(1)
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    NCModel, LPModel, LinkPredictor, cfg = _build_models(
        side, project_root, source_root, payload
    )
    x, physical = payload["x"], payload["physical"]
    text, visual = payload["text"], payload["visual"]
    data_info = payload["data_info"]

    torch.manual_seed(221)
    model = NCModel(cfg, data_info)
    model.load_semantic_cache(
        payload["nc_cache_path"],
        expected_dataset="Parity",
        expected_feature_fingerprint=payload["feature_fingerprint"],
        expected_candidate_fingerprint=payload["candidate_fingerprint"],
        expected_v7b_fingerprint=payload["nc_v7b_fingerprint"],
        **(
            {"expected_identity_fingerprint": payload["nc_identity_fingerprint"]}
            if side == "target" else {}
        ),
    )
    classifier = torch.nn.Linear(model.out_dim, int(data_info["num_classes"]))
    initial_model_state = _tensor_state(model)
    initial_classifier_state = _tensor_state(classifier)

    model.eval()
    classifier.eval()
    with torch.no_grad():
        z, _, _, aux_loss, info = model(x, physical)
        target = payload["labels"][payload["train_idx"]]
        main_ce = F.cross_entropy(classifier(z[payload["train_idx"]]), target)
        unimodal_ce = 0.5 * sum(
            F.cross_entropy(info["unimodal_logits"][modality][payload["train_idx"]], target)
            for modality in ("text", "visual")
        )
        nc_loss = main_ce + 0.3 * unimodal_ce + aux_loss
        eval_values = {
            "z": z.detach().cpu(),
            "balance_loss": info["balance_loss"].detach().cpu(),
            "aux_loss": aux_loss.detach().cpu(),
            "unimodal_text_logits": info["unimodal_logits"]["text"].detach().cpu(),
            "unimodal_visual_logits": info["unimodal_logits"]["visual"].detach().cpu(),
            "nc_loss": nc_loss.detach().cpu(),
        }

    model.train()
    classifier.train()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()), lr=1.0e-3, weight_decay=1.0e-4
    )
    torch.manual_seed(993)
    z, _, _, aux_loss, info = model(x, physical)
    target = payload["labels"][payload["train_idx"]]
    main_ce = F.cross_entropy(classifier(z[payload["train_idx"]]), target)
    unimodal_ce = 0.5 * sum(
        F.cross_entropy(info["unimodal_logits"][modality][payload["train_idx"]], target)
        for modality in ("text", "visual")
    )
    train_loss = main_ce + 0.3 * unimodal_ce + aux_loss
    train_loss.backward()
    gradients = _parameter_grads(model)
    optimizer.step()
    updated_model_state = _tensor_state(model)
    updated_classifier_state = _tensor_state(classifier)
    nc_values = {
        "initial_model_state": initial_model_state,
        "initial_classifier_state": initial_classifier_state,
        "eval": eval_values,
        "training_loss": train_loss.detach().cpu(),
        "gradients": gradients,
        "updated_model_state": updated_model_state,
        "updated_classifier_state": updated_classifier_state,
    }

    lp_cfg = OmegaConf.create(
        {"model": cfg.model, "task": {"name": "lp", "loss": {"aux_weight": 1.0}}}
    )
    torch.manual_seed(221)
    lp_model = LPModel(lp_cfg, {**data_info, "num_classes": 0})
    lp_model.load_semantic_cache(
        payload["lp_cache_path"],
        expected_dataset="Parity",
        expected_feature_fingerprint=payload["feature_fingerprint"],
        expected_candidate_fingerprint=payload["candidate_fingerprint"],
        expected_v7b_fingerprint=payload["lp_v7b_fingerprint"],
        **(
            {"expected_identity_fingerprint": payload["lp_identity_fingerprint"]}
            if side == "target" else {}
        ),
    )
    train_graph = payload["train_graph"]
    lp_model.bind_lp_graph(x, train_graph)
    data = Data(x=x, edge_index=train_graph)
    loader = LinkNeighborLoader(
        data,
        edge_label_index=payload["lp_edge_label_index"],
        edge_label=payload["lp_edge_label"],
        num_neighbors=[2, 2, 2],
        batch_size=4,
        shuffle=False,
        subgraph_type="bidirectional",
        num_workers=0,
    )
    batch = next(iter(loader))
    from src.tasks.lp import _build_message_edge_lookup, _exclude_positive_label_edges_by_global_eid
    lookup = _build_message_edge_lookup(train_graph, int(x.size(0)))
    _exclude_positive_label_edges_by_global_eid(batch, lookup)
    lp_model.prepare_sampled_lp_batch(batch, sample_seed=2028)
    context = {
        "x": batch.x.detach().cpu().clone(),
        "edge_index": batch.edge_index.detach().cpu().clone(),
        "n_id": batch.n_id.detach().cpu().clone(),
        "e_id": batch.e_id.detach().cpu().clone(),
        "edge_label_index": batch.edge_label_index.detach().cpu().clone(),
        "edge_label": batch.edge_label.detach().cpu().clone(),
        "semantic_text_edges": lp_model._sampled_semantic_edges["text"][0].detach().cpu().clone(),
        "semantic_text_weights": lp_model._sampled_semantic_edges["text"][1].detach().cpu().clone(),
        "semantic_visual_edges": lp_model._sampled_semantic_edges["visual"][0].detach().cpu().clone(),
        "semantic_visual_weights": lp_model._sampled_semantic_edges["visual"][1].detach().cpu().clone(),
    }
    lp_model._batch_n_id = batch.n_id
    lp_model.train()
    torch.manual_seed(884)
    z_lp, _, _, _, _ = lp_model(batch.x, batch.edge_index)
    predictor = LinkPredictor(
        in_dim=lp_model.out_dim, hidden_dim=16, num_layers=2, dropout=0.02
    )
    src, dst = batch.edge_label_index
    lp_prediction = predictor.score_pairs(z_lp[src], z_lp[dst])
    lp_values = {
        "context": context,
        "node_embeddings": z_lp.detach().cpu(),
        "predictions": lp_prediction.detach().cpu(),
        "sampled_context_metadata": dict(lp_model.last_sampled_context),
    }
    torch.save({"nc": nc_values, "lp": lp_values}, output_path)


if __name__ == "__main__":
    side = sys.argv[1]
    project_root = Path(sys.argv[2]).resolve()
    source_root = Path(sys.argv[3]).resolve()
    run(side, project_root, source_root, Path(sys.argv[4]), Path(sys.argv[5]))
