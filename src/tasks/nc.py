from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader as TorchDataLoader

from src.data import MAGData
from src.models import build_model
from src.tasks.common import (
    build_optimizer, clone_state_dict, load_state_dict_cpu, log_optimizer,
    scheduler_step, validation_metrics_match,
)
from src.tasks.inference import infer_all_embeddings, resolve_inference_mode
from src.utils.metrics import format_pct
from src.utils.run_artifacts import write_json_atomic
from src.utils.seeds import set_seed
from src.utils.summary import count_parameters, mean_std


def _uses_graph_encoder(cfg) -> bool:
    return str(cfg.model.name).lower() != "mlp"


def _resolve_training_mode(cfg) -> str:
    mode = str(cfg.task.get("training_mode", "full_graph")).strip().lower()
    if mode != "full_graph":
        raise ValueError(
            "unified_full_graph_nc_v1 requires task.training_mode='full_graph'; "
            f"got {mode!r}"
        )
    return "full_graph" if _uses_graph_encoder(cfg) else "feature_only"


def _compose_full_graph_loss(
    *,
    model,
    info: dict,
    main_ce: torch.Tensor,
    aux_loss: torch.Tensor,
    aux_weight: float,
    unimodal_weight: float,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
) -> torch.Tensor:
    if unimodal_weight < 0:
        raise ValueError("task.loss.unimodal_weight must be non-negative")
    if unimodal_weight == 0:
        return main_ce + aux_weight * aux_loss
    if not bool(getattr(model, "supports_unimodal_aux_loss", False)):
        raise ValueError("positive unimodal_weight requires a model that declares supports_unimodal_aux_loss")
    if not isinstance(info, dict) or not isinstance(info.get("unimodal_logits"), dict):
        raise RuntimeError("training forward omitted the declared unimodal_logits interface")
    logits = info["unimodal_logits"]
    if set(logits) != {"text", "visual"}:
        raise RuntimeError("unimodal_logits must contain exactly Text and Visual logits")
    target = labels[train_idx]
    if target.numel() == 0 or bool((target < 0).any()):
        raise ValueError("unimodal auxiliary supervision requires valid training labels")
    losses = []
    for modality in ("text", "visual"):
        value = logits[modality]
        if value.ndim != 2 or value.size(0) != labels.size(0):
            raise RuntimeError(f"{modality} auxiliary logits must cover the full global graph")
        losses.append(nn.functional.cross_entropy(value[train_idx], target))
    return main_ce + unimodal_weight * 0.5 * (losses[0] + losses[1]) + aux_weight * aux_loss


def _training_class_weights(
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    num_classes: int,
    *,
    enabled: bool,
) -> torch.Tensor | None:
    if not enabled:
        return None
    target = labels[train_idx]
    if target.numel() == 0 or bool((target < 0).any()) or bool((target >= num_classes).any()):
        raise ValueError("class-balanced main CE requires valid Train labels only")
    counts = torch.bincount(target.long(), minlength=int(num_classes)).to(torch.float32)
    observed = counts > 0
    if not bool(observed.any()):
        raise ValueError("class-balanced main CE found no observed Train classes")
    weights = torch.zeros_like(counts)
    inverse_sqrt = counts[observed].rsqrt()
    inverse_sqrt = inverse_sqrt / inverse_sqrt.mean()
    weights[observed] = inverse_sqrt.clamp(0.5, 2.0)
    return weights


def _resolve_nc_eval_labels(
    data: MAGData,
    *,
    fixed_labels: list[int] | tuple[int, ...] | str | None = None,
) -> list[int]:
    if data.y is None or data.num_classes is None:
        raise ValueError("NC data must define labels and num_classes")
    if fixed_labels is not None:
        if isinstance(fixed_labels, str):
            if fixed_labels.strip().lower() != "all_classes":
                raise ValueError("task.macro_f1_labels string value must be 'all_classes'")
            labels = list(range(int(data.num_classes)))
        else:
            labels = sorted({int(value) for value in fixed_labels})
        if not labels or labels[0] < 0 or labels[-1] >= int(data.num_classes):
            raise ValueError("task.macro_f1_labels must contain valid configured class IDs")
        return labels
    # The paper protocol scores all configured classes. This set is fixed by
    # the dataset schema and never inferred from any split's observed labels.
    return list(range(int(data.num_classes)))


@torch.no_grad()
def _evaluate_split(
    classifier: nn.Module,
    z: torch.Tensor,
    labels: torch.Tensor,
    idx: torch.Tensor,
    device: torch.device,
    batch_size: int,
    eval_labels: list[int] | tuple[int, ...],
) -> dict[str, float]:
    classifier.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch_idx in TorchDataLoader(idx.cpu(), batch_size=batch_size, shuffle=False):
        logits = classifier(z[batch_idx].to(device))
        preds.append(logits.argmax(dim=-1).cpu())
        targets.append(labels[batch_idx].cpu())
    pred, target = torch.cat(preds), torch.cat(targets)
    return {
        "acc": float((pred == target).float().mean().item()),
        "macro_f1": float(
            f1_score(
                target.numpy(), pred.numpy(), labels=list(eval_labels),
                average="macro", zero_division=0,
            )
        ),
    }


def _format_auxiliary_losses(info: dict | None) -> str:
    if not isinstance(info, dict):
        return ""
    values = []
    for group, parts in info.items():
        if not isinstance(parts, dict):
            continue
        for name, value in parts.items():
            if torch.is_tensor(value) and value.numel() == 1:
                value = float(value.detach().item())
            if isinstance(value, (int, float)):
                values.append(f"{group}.{name}={float(value):.4f}")
    return " | Aux " + ", ".join(values) if values else ""


def _checkpoint_path_for_run(path_like: str | Path, cfg, run_id: int) -> Path:
    path = Path(str(path_like)).expanduser()
    if int(cfg.num_runs) > 1:
        path = path.with_name(f"{path.stem}_run{run_id + 1}{path.suffix}")
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run_single_nc(
    cfg,
    data: MAGData,
    device: torch.device,
    logger: logging.Logger,
    run_id: int,
    seed: int,
    eval_labels: list[int],
) -> dict[str, object]:
    inference_mode = resolve_inference_mode(cfg)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = build_model(cfg, data_info)
    if bool(getattr(model, "requires_global_semantic_candidates", False)):
        cache_path = cfg.semantic_graph.get("resolved_cache_path")
        if not cache_path:
            raise ValueError("MRAGE NC requires automatic semantic graph initialization")
        model.load_semantic_cache(
            str(cache_path),
            expected_dataset=str(cfg.dataset.name),
            expected_feature_fingerprint=cfg.semantic_graph.get("feature_fingerprint"),
            expected_candidate_fingerprint=cfg.semantic_graph.get("candidate_fingerprint"),
            expected_v7b_fingerprint=cfg.semantic_graph.get("v7b_fingerprint"),
            expected_identity_fingerprint=cfg.semantic_graph.get("identity_fingerprint"),
        )
    model = model.to(device)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    optimizer = build_optimizer(list(model.parameters()) + list(classifier.parameters()), cfg, model=model)
    effective_optimizer = log_optimizer(logger, optimizer)
    training_mode = _resolve_training_mode(cfg)
    uses_graph = _uses_graph_encoder(cfg)
    full_graph_training = training_mode == "full_graph"
    unimodal_weight = float(cfg.task.loss.get("unimodal_weight", 0.0))
    if unimodal_weight < 0:
        raise ValueError("task.loss.unimodal_weight must be non-negative")
    if unimodal_weight > 0 and (
        not full_graph_training or not bool(getattr(model, "supports_unimodal_aux_loss", False))
    ):
        raise ValueError("unimodal auxiliary CE requires a supporting full-graph NC model")

    x_all, y_all = data.x.to(device), data.y.to(device)
    edge_index_all = data.edge_index.to(device) if uses_graph else None
    train_idx_all = data.train_idx.to(device)
    train_mask_all = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask_all[train_idx_all] = True
    class_balanced = bool(cfg.task.loss.get("class_balanced_main_ce", False))
    class_weights = _training_class_weights(
        y_all, train_idx_all, int(data.num_classes), enabled=class_balanced
    )
    if class_balanced and not full_graph_training:
        raise ValueError("class-balanced main CE requires full-graph NC")
    train_loader = None
    if not uses_graph:
        train_loader = TorchDataLoader(
            data.train_idx.cpu(), batch_size=int(cfg.task.batch_size), shuffle=True
        )

    logger.info(
        "[Run %d/%d] seed=%d | model parameters=%d | classifier parameters=%d",
        run_id + 1, int(cfg.num_runs), seed,
        count_parameters(model), count_parameters(classifier),
    )
    logger.info("Protocol: %s | training mode: %s | inference mode: %s", cfg.task.protocol_version, training_mode, inference_mode)

    best_val = -1.0
    best_metrics: dict[str, float] = {}
    best_epoch: int | None = None
    best_model_state = None
    best_head_state = None
    patience_total = int(cfg.task.patience)
    patience_left = patience_total
    min_epoch = int(cfg.task.get("early_stop_min_epoch", 1))
    min_delta = float(cfg.task.get("early_stop_min_delta", 0.0))
    grad_clip = float(cfg.task.get("grad_clip", 1.0))
    aux_weight = float(cfg.task.loss.aux_weight)
    inference_batch_size = int(cfg.task.inference_batch_size)
    eval_every = int(cfg.task.eval_every)
    if eval_every < 1:
        raise ValueError("task.eval_every must be >= 1")

    for epoch in range(1, int(cfg.task.epochs) + 1):
        model.train()
        classifier.train()
        info = {}
        if full_graph_training and bool(getattr(model, "supports_nc_node_batches", False)):
            node_batch_size = int(getattr(model, "node_batch_size", cfg.task.batch_size))
            batch_scope = str(getattr(model, "nc_auxiliary_scope", "train_nodes"))
            if batch_scope not in {"train_nodes", "all_nodes"}:
                raise ValueError(f"Unsupported node-batch scope: {batch_scope}")
            batch_nodes = (
                torch.arange(data.num_nodes, dtype=torch.long)
                if batch_scope == "all_nodes"
                else train_idx_all.detach().cpu()
            )
            train_loader = TorchDataLoader(
                batch_nodes,
                batch_size=max(node_batch_size, 1),
                shuffle=True,
            )
            optimizer.zero_grad(set_to_none=True)
            total_examples = int(train_idx_all.numel())
            total_batch_nodes = int(batch_nodes.numel())
            train_loss = 0.0
            for batch_idx in train_loader:
                batch_idx = batch_idx.to(device)
                z, _, _, aux_loss, info = model.forward_node_batch(
                    x_all, edge_index_all, batch_idx
                )
                logits = classifier(z)
                batch_train_mask = train_mask_all[batch_idx]
                if bool(batch_train_mask.any()):
                    main_ce = nn.functional.cross_entropy(
                        logits[batch_train_mask],
                        y_all[batch_idx[batch_train_mask]],
                        weight=class_weights,
                    )
                    task_scale = float(batch_train_mask.sum().item()) / max(total_examples, 1)
                else:
                    main_ce = logits.new_tensor(0.0)
                    task_scale = 0.0
                extra_native_loss = model.additional_nc_loss(classifier, info)
                auxiliary_scale = float(batch_idx.numel()) / max(total_batch_nodes, 1)
                loss = (
                    main_ce * task_scale
                    + aux_weight * (aux_loss + extra_native_loss) * auxiliary_scale
                )
                loss.backward()
                train_loss += float(loss.detach().item())
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(classifier.parameters()),
                max_norm=grad_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
        elif full_graph_training:
            if hasattr(model, "_batch_n_id"):
                model._batch_n_id = None
            optimizer.zero_grad(set_to_none=True)
            z, _, _, aux_loss, info = model(x_all, edge_index_all)
            train_ce = nn.functional.cross_entropy(
                classifier(z[train_idx_all]), y_all[train_idx_all], weight=class_weights
            )
            if unimodal_weight > 0:
                loss = _compose_full_graph_loss(
                    model=model, info=info, main_ce=train_ce, aux_loss=aux_loss,
                    aux_weight=aux_weight, unimodal_weight=unimodal_weight,
                    labels=y_all, train_idx=train_idx_all,
                )
            else:
                loss = train_ce + aux_weight * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(classifier.parameters()),
                max_norm=grad_clip, error_if_nonfinite=True,
            )
            optimizer.step()
            train_loss = float(loss.detach().item())
            del z
        else:
            total_loss = 0.0
            total_examples = 0
            assert train_loader is not None
            for batch_idx in train_loader:
                batch_idx = batch_idx.to(device)
                labels = y_all[batch_idx]
                optimizer.zero_grad(set_to_none=True)
                z, _, _, aux_loss, info = model(x_all[batch_idx], None)
                logits = classifier(z)
                loss = nn.functional.cross_entropy(logits, labels, weight=class_weights) + aux_weight * aux_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(classifier.parameters()),
                    max_norm=grad_clip, error_if_nonfinite=True,
                )
                optimizer.step()
                total_loss += float(loss.detach().item()) * int(labels.numel())
                total_examples += int(labels.numel())
            train_loss = total_loss / max(total_examples, 1)

        scheduler_step(cfg, optimizer, epoch, int(cfg.task.epochs))
        aux_text = _format_auxiliary_losses(info)
        if epoch % eval_every != 0:
            logger.info("Epoch %05d | Train Loss %.4f%s", epoch, train_loss, aux_text)
            continue
        z = infer_all_embeddings(
            model, data, device, uses_graph, inference_batch_size, inference_mode
        )
        val_metrics = _evaluate_split(
            classifier, z, data.y, data.val_idx, device, inference_batch_size, eval_labels
        )
        logger.info(
            "Epoch %05d | Train Loss %.4f%s | Val Acc %.2f | Val Macro-F1 %.2f",
            epoch, train_loss, aux_text, format_pct(val_metrics["acc"]), format_pct(val_metrics["macro_f1"]),
        )
        improved = val_metrics["acc"] > best_val + min_delta
        stop_early = False
        if improved:
            best_val = val_metrics["acc"]
            best_epoch = epoch
            best_metrics = {"val_acc": val_metrics["acc"], "val_macro_f1": val_metrics["macro_f1"]}
            best_model_state = clone_state_dict(model)
            best_head_state = clone_state_dict(classifier)
            patience_left = patience_total
        elif epoch >= min_epoch:
            patience_left -= 1
            logger.info("Patience %d/%d | Best Val Acc %.2f", patience_total - patience_left, patience_total, format_pct(best_val))
            stop_early = patience_left <= 0
        del z
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if stop_early:
            logger.info("Early stopping at epoch %03d", epoch)
            break

    if best_model_state is None or best_head_state is None:
        raise RuntimeError("NC training finished without a validation-selected checkpoint")
    load_state_dict_cpu(model, best_model_state)
    load_state_dict_cpu(classifier, best_head_state)
    if bool(cfg.task.get("evaluate_test", True)):
        z = infer_all_embeddings(model, data, device, uses_graph, inference_batch_size, inference_mode)
        test_metrics = _evaluate_split(
            classifier, z, data.y, data.test_idx, device, inference_batch_size, eval_labels
        )
        best_metrics.update({"test_acc": test_metrics["acc"], "test_macro_f1": test_metrics["macro_f1"]})
        del z

    run_metadata = {
        "best_epoch": int(best_epoch),
        "model_parameters": count_parameters(model),
        "classifier_parameters": count_parameters(classifier),
        "optimizer": type(optimizer).__name__,
        "effective_optimizer": effective_optimizer,
        "protocol_version": str(cfg.task.get("protocol_version", "")),
        "split_path": data.info.get("nc_split_path", data.info.get("node_split_path")),
        "split_sha256": data.info.get("nc_split_sha256", data.info.get("node_split_sha256")),
        "split_seed": data.info.get("nc_split_seed"),
    }
    nc_adapter = getattr(model, "nc_training_adapter_version", None)
    if nc_adapter is not None:
        run_metadata["nc_training_adapter_version"] = str(nc_adapter)
    for name in (
        "semantic_candidate_fingerprint", "semantic_feature_fingerprint",
        "semantic_graph_identity_fingerprint", "semantic_v7b_fingerprint",
    ):
        value = getattr(model, name, None)
        if value is not None:
            run_metadata[name] = value

    save_path = cfg.task.get("save_ckpt_path")
    if save_path:
        path = _checkpoint_path_for_run(save_path, cfg, run_id)
        torch.save(
            {
                "task": "nc",
                "dataset": str(cfg.dataset.name),
                "model": str(cfg.model.name),
                "protocol_version": str(cfg.task.get("protocol_version", "")),
                "seed": seed,
                "selection": "best_val_accuracy",
                "epoch": int(best_epoch),
                "metrics": dict(best_metrics),
                "run_metadata": run_metadata,
                "model_state": clone_state_dict(model),
                "head_state": clone_state_dict(classifier),
                "data_info": data_info,
            },
            path,
        )
        logger.info("Saved checkpoint: %s | best_epoch=%s", path, best_epoch)

    logger.info(
        "[Run %d] Best Val Acc %.2f | Val Macro-F1 %.2f | epoch=%d",
        run_id + 1, format_pct(best_metrics["val_acc"]),
        format_pct(best_metrics["val_macro_f1"]), int(best_epoch),
    )
    if "test_acc" in best_metrics:
        logger.info(
            "[Run %d] Test Acc %.2f | Test Macro-F1 %.2f",
            run_id + 1, format_pct(best_metrics["test_acc"]),
            format_pct(best_metrics["test_macro_f1"]),
        )
    return {**best_metrics, "_run_metadata": run_metadata}


def run_nc(
    cfg,
    data: MAGData,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, tuple[float, float]]:
    if data.y is None or data.train_idx is None or data.val_idx is None or data.test_idx is None:
        raise ValueError("NC data must contain y/train_idx/val_idx/test_idx")
    _resolve_training_mode(cfg)
    evaluate_test = bool(cfg.task.get("evaluate_test", True))
    if bool(cfg.task.get("mask_test_labels", False)):
        if evaluate_test:
            raise ValueError("task.mask_test_labels requires task.evaluate_test=false")
        data = copy.copy(data)
        data.y = data.y.clone()
        data.y[data.test_idx] = -100
    eval_labels = _resolve_nc_eval_labels(
        data,
        fixed_labels=cfg.task.get("macro_f1_labels"),
    )
    run_results = []
    for run_id in range(int(cfg.num_runs)):
        seed = int(cfg.seed) + run_id
        set_seed(seed)
        run_results.append(_run_single_nc(cfg, data, device, logger, run_id, seed, eval_labels))

    run_metrics_path = cfg.task.get("run_metrics_path")
    if run_metrics_path:
        path = Path(str(run_metrics_path)).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "protocol_version": str(cfg.task.get("protocol_version", "")),
                    "base_seed": int(cfg.seed),
                    "run_seeds": [int(cfg.seed) + run_id for run_id in range(int(cfg.num_runs))],
                    "aggregation": "mean ± population std (ddof=0)",
                    "runs": [
                        {
                            "run_id": run_id,
                            "seed": int(cfg.seed) + run_id,
                            "metrics": {key: float(value) for key, value in item.items() if not key.startswith("_")},
                            "metadata": item.get("_run_metadata", {}),
                        }
                        for run_id, item in enumerate(run_results)
                    ],
                },
                handle,
                indent=2,
            )
    keys = ["val_acc", "val_macro_f1"]
    if bool(cfg.task.get("evaluate_test", True)):
        keys.extend(["test_acc", "test_macro_f1"])
    names = {
        "val_acc": "Val Accuracy", "val_macro_f1": "Val Macro-F1",
        "test_acc": "Test Accuracy", "test_macro_f1": "Test Macro-F1",
    }
    output: dict[str, tuple[float, float]] = {}
    logger.info("============================================================")
    logger.info("Final Results over %d runs", int(cfg.num_runs))
    for key in keys:
        mean, std = mean_std([item[key] for item in run_results])
        output[key] = (mean, std)
        logger.info("%s: %.2f ± %.2f", names[key], format_pct(mean), format_pct(std))
    logger.info("============================================================")
    return output


def evaluate_nc_checkpoint(
    cfg,
    data: MAGData,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, float]:
    """Recompute Validation, then evaluate Test exactly once from a saved run."""
    checkpoint_path = Path(str(cfg.task.checkpoint_path)).expanduser()
    validation_path = Path(str(cfg.task.validation_metrics_path)).expanduser()
    final_path = Path(str(cfg.task.final_test_metrics_path)).expanduser()
    status_path = Path(str(cfg.task.paper_run_status_path)).expanduser()
    if final_path.exists():
        raise FileExistsError(f"Final-test results already exist: {final_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if int(status.get("test_evaluation_count", 0)) != 0:
        raise RuntimeError("Test evaluation has already been completed for this run")
    if status.get("status") not in {"validation_complete", "final_test_running"}:
        raise RuntimeError(f"run is not ready for final Test evaluation: {status.get('status')}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("task") != "nc":
        raise ValueError("checkpoint task is not NC")
    if checkpoint.get("dataset") != str(cfg.dataset.name) or checkpoint.get("model") != str(cfg.model.name):
        raise ValueError("checkpoint dataset/model identity differs from the requested run")
    if int(checkpoint.get("seed", -1)) != int(cfg.seed):
        raise ValueError("checkpoint training seed differs from the requested run")
    if checkpoint.get("protocol_version") != str(cfg.task.protocol_version):
        raise ValueError("checkpoint protocol version differs from the current config")
    run_metadata = checkpoint.get("run_metadata", {})
    current_split_hash = data.info.get("nc_split_sha256", data.info.get("node_split_sha256"))
    if not current_split_hash or run_metadata.get("split_sha256") != current_split_hash:
        raise ValueError("checkpoint NC split fingerprint differs from the loaded split")
    current_split_seed = data.info.get("nc_split_seed")
    if current_split_seed is not None and int(run_metadata.get("split_seed", -1)) != int(current_split_seed):
        raise ValueError("checkpoint NC split seed differs from the loaded split")

    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    if checkpoint.get("data_info") != data_info:
        raise ValueError("checkpoint data dimensions differ from the loaded NC dataset")
    # Reproduce fixed, non-parameter initialization for legacy checkpoints
    # whose model_state predates MMGCN's registered id_embedding buffer.
    set_seed(int(cfg.seed))
    model = build_model(cfg, data_info)
    if bool(getattr(model, "requires_global_semantic_candidates", False)):
        cache_path = cfg.semantic_graph.get("resolved_cache_path")
        if not cache_path:
            raise ValueError("MRAGE NC final evaluation requires a validated semantic cache")
        model.load_semantic_cache(
            str(cache_path),
            expected_dataset=str(cfg.dataset.name),
            expected_feature_fingerprint=cfg.semantic_graph.get("feature_fingerprint"),
            expected_candidate_fingerprint=cfg.semantic_graph.get("candidate_fingerprint"),
            expected_v7b_fingerprint=cfg.semantic_graph.get("v7b_fingerprint"),
            expected_identity_fingerprint=cfg.semantic_graph.get("identity_fingerprint"),
        )
        for key in (
            "semantic_candidate_fingerprint", "semantic_feature_fingerprint",
            "semantic_graph_identity_fingerprint", "semantic_v7b_fingerprint",
        ):
            saved = run_metadata.get(key)
            current = getattr(model, key, None)
            if saved is not None and saved != current:
                raise ValueError(f"checkpoint semantic graph identity differs for {key}")
    model = model.to(device)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    model_state = checkpoint["model_state"]
    if str(cfg.model.name).lower() == "mmgcn" and "id_embedding" not in model_state:
        incompatible = model.load_state_dict(model_state, strict=False)
        if set(incompatible.missing_keys) != {"id_embedding"} or incompatible.unexpected_keys:
            raise RuntimeError(
                "legacy MMGCN checkpoint is missing unexpected model state: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        logger.info("Restored legacy MMGCN fixed id_embedding deterministically from seed %d", int(cfg.seed))
    else:
        load_state_dict_cpu(model, model_state)
    load_state_dict_cpu(classifier, checkpoint["head_state"])

    eval_labels = _resolve_nc_eval_labels(
        data, fixed_labels=cfg.task.get("macro_f1_labels", "all_classes")
    )
    z = infer_all_embeddings(
        model,
        data,
        device,
        _uses_graph_encoder(cfg),
        int(cfg.task.inference_batch_size),
        resolve_inference_mode(cfg),
    )
    validation_raw = _evaluate_split(
        classifier, z, data.y, data.val_idx, device,
        int(cfg.task.inference_batch_size), eval_labels,
    )
    validation = {
        "val_acc": validation_raw["acc"],
        "val_macro_f1": validation_raw["macro_f1"],
    }
    saved_metrics = checkpoint.get("metrics", {})
    expected_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    for metric in ("val_acc", "val_macro_f1"):
        expected = float(saved_metrics[metric])
        if metric in expected_validation.get("metrics", {}):
            if not validation_metrics_match(expected, expected_validation["metrics"][metric]):
                raise ValueError(f"saved validation record disagrees with checkpoint for {metric}")
        if not validation_metrics_match(expected, validation[metric]):
            raise ValueError(
                f"recomputed Validation {metric}={validation[metric]} differs from checkpoint {expected}"
            )

    z = infer_all_embeddings(
        model,
        data,
        device,
        _uses_graph_encoder(cfg),
        int(cfg.task.inference_batch_size),
        resolve_inference_mode(cfg),
    )
    test_raw = _evaluate_split(
        classifier, z, data.y, data.test_idx, device,
        int(cfg.task.inference_batch_size), eval_labels,
    )
    test = {
        "test_acc": test_raw["acc"],
        "test_macro_f1": test_raw["macro_f1"],
    }
    metrics = {**validation, **test}
    payload = {
        "dataset": str(cfg.dataset.name),
        "task": "nc",
        "model": str(cfg.model.name),
        "seed": int(cfg.seed),
        "split_id": current_split_hash,
        "protocol_version": str(cfg.task.protocol_version),
        "best_epoch": int(checkpoint["epoch"]),
        "selection_metric": "val_acc",
        "metrics": metrics,
        "test_evaluation_count": 1,
    }
    write_json_atomic(final_path, payload)
    status.update(
        status="final_test_complete",
        test_evaluation_count=1,
        final_test_metrics_path=str(final_path),
    )
    write_json_atomic(status_path, status)
    logger.info("Final Test metrics: %s", metrics)
    return metrics
