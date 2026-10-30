from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.data import load_mag_data
from src.data.loaders import resolve_path
from src.data.semantic_graph_manager import ensure_semantic_graph_cache
from src.tasks import evaluate_lp_checkpoint, evaluate_nc_checkpoint, run_lp, run_nc
from src.utils.device import get_device
from src.utils.logging import setup_logger


def _log_data_info(logger, data, model_name: str) -> None:
    logger.info("Dataset: %s | Source: %s | Task: %s", data.name, data.source, data.task)
    logger.info("Model: %s", model_name)
    logger.info("X: %s | dtype=%s", tuple(data.x.shape), data.x.dtype)
    if data.x_i is not None:
        logger.info("X_i: %s | X_t: %s", tuple(data.x_i.shape), tuple(data.x_t.shape))
    logger.info("Graph edge_index: %s | num_nodes=%d | num_edges=%d", tuple(data.edge_index.shape), data.num_nodes, data.num_edges)
    if data.y is not None:
        logger.info("Labels: shape=%s | num_classes=%s", tuple(data.y.shape), data.num_classes)
    if data.train_idx is not None:
        logger.info(
            "NC split: train=%d | val=%d | test=%d",
            int(data.train_idx.numel()),
            int(data.val_idx.numel()),
            int(data.test_idx.numel()),
        )
    if data.edge_split is not None:
        logger.info(
            "LP split: train=%d | valid=%d x %d neg | test=%d x %d neg",
            int(data.edge_split.train["source_node"].numel()),
            int(data.edge_split.valid["source_node"].numel()),
            int(data.edge_split.valid["target_node_neg"].size(1)),
            int(data.edge_split.test["source_node"].numel()),
            int(data.edge_split.test["target_node_neg"].size(1)),
        )
    for key, value in data.info.items():
        logger.info("%s: %s", key, value)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = setup_logger(output_dir, cfg.logging.level)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    torch_threads = cfg.task.get("torch_threads")
    if torch_threads is not None:
        torch.set_num_threads(int(torch_threads))

    device = get_device(str(cfg.device))
    logger.info("Device: %s", device)

    data = load_mag_data(cfg, str(cfg.task.name), int(cfg.seed))
    edge_override = cfg.task.get("edge_index_override_path")
    if edge_override:
        payload = torch.load(resolve_path(str(edge_override)), map_location="cpu", weights_only=False)
        edge_index = payload.get("edge_index") if isinstance(payload, dict) else payload
        edge_index = torch.as_tensor(edge_index, dtype=torch.long).contiguous()
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("task.edge_index_override_path must contain edge_index with shape [2, num_edges]")
        if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= data.num_nodes):
            raise ValueError("graph override contains node IDs outside the loaded dataset")
        data.edge_index = edge_index
        data.info["edge_index_override_path"] = str(resolve_path(str(edge_override)))
    if str(cfg.model.name).lower() == "mrage":
        if data.x_t is None or data.x_i is None:
            raise ValueError("MRAGE requires separate Text and Visual feature matrices")
        model_knn_k = int(cfg.model.get("semantic_knn_k", 8))
        graph_knn_k = int(cfg.semantic_graph.knn_k)
        if model_knn_k != graph_knn_k:
            raise ValueError(
                "model.semantic_knn_k and semantic_graph.knn_k must match; "
                f"got {model_knn_k} and {graph_knn_k}"
            )
        semantic_cache = ensure_semantic_graph_cache(
            dataset=str(data.name),
            task=str(cfg.task.name).lower(),
            text_features=data.x_t,
            visual_features=data.x_i,
            # LP data.edge_index contains only the official Train physical graph.
            physical_edge_index=data.edge_index,
            config=cfg.semantic_graph,
            project_root=Path(__file__).resolve().parents[1],
        )
        cfg.semantic_graph.resolved_cache_path = str(semantic_cache.path)
        cfg.semantic_graph.identity_fingerprint = semantic_cache.identity_fingerprint
        cfg.semantic_graph.feature_fingerprint = str(semantic_cache.metadata["feature_fingerprint"])
        cfg.semantic_graph.candidate_fingerprint = str(semantic_cache.metadata["candidate_fingerprint"])
        cfg.semantic_graph.v7b_fingerprint = str(semantic_cache.metadata["v7b_fingerprint"])
        logger.info(
            "Semantic graph cache %s: %s | identity=%s | candidates=%s | v7b=%s",
            "HIT" if semantic_cache.cache_hit else "MISS",
            semantic_cache.path,
            semantic_cache.identity_fingerprint,
            semantic_cache.metadata["candidate_fingerprint"],
            semantic_cache.metadata["v7b_fingerprint"],
        )
    _log_data_info(logger, data, str(cfg.model.name))

    phase = str(cfg.task.get("phase", "train")).strip().lower()
    if phase == "final_test":
        if str(cfg.task.name) == "nc":
            results = evaluate_nc_checkpoint(cfg, data, device, logger)
        elif str(cfg.task.name) == "lp":
            results = evaluate_lp_checkpoint(cfg, data, device, logger)
        else:
            raise ValueError(f"Unsupported task: {cfg.task.name}")
    elif phase != "train":
        raise ValueError("task.phase must be 'train' or 'final_test'")
    elif int(cfg.task.epochs) <= 0:
        logger.info("task.epochs <= 0, stopping after data loading/split preparation")
        results = {}
    elif str(cfg.task.name) == "nc":
        results = run_nc(cfg, data, device, logger)
    elif str(cfg.task.name) == "lp":
        results = run_lp(cfg, data, device, logger)
    else:
        raise ValueError(f"Unsupported task: {cfg.task.name}")

    serializable = {
        key: ({"mean": value[0], "std": value[1]} if isinstance(value, (tuple, list))
              else {"value": float(value)})
        for key, value in results.items()
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved results: %s", output_dir / "results.json")


if __name__ == "__main__":
    main()
