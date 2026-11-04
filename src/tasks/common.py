from __future__ import annotations

import copy
import math

import torch
from omegaconf import ListConfig


def validation_metrics_match(
    saved_value: float,
    recomputed_value: float,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> bool:
    """Compare reloaded Validation metrics with practical floating-point tolerance."""
    return math.isclose(
        float(saved_value), float(recomputed_value), rel_tol=rel_tol, abs_tol=abs_tol
    )


# The LP final-Test phase recomputes Validation in a separate process. CUDA
# index_add_/scatter reductions are not bitwise deterministic, so a handful of
# rank-boundary flips can move Hits@k/MRR by ~1e-5 even when the checkpoint is
# identical. The exact checkpoint file is verified separately by SHA256; this
# tolerance only absorbs rank-boundary noise. 1e-3 = 0.1 percentage point.
RANKING_VALIDATION_METRIC_ABS_TOL = 1.0e-3


def ranking_validation_metrics_match(saved_value: float, recomputed_value: float) -> bool:
    """Compare LP rank metrics while tolerating small nondeterministic drift."""
    return validation_metrics_match(
        saved_value,
        recomputed_value,
        abs_tol=RANKING_VALIDATION_METRIC_ABS_TOL,
    )


def clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def load_state_dict_cpu(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    module.load_state_dict(copy.deepcopy(state_dict))


def build_optimizer(parameters, cfg, model=None) -> torch.optim.Optimizer:
    """Build a task optimizer while honoring an imported model preset."""
    name = str(cfg.task.get("optimizer", "adamw")).strip().lower()
    lr = float(cfg.model.get("lr", cfg.task.lr))
    weight_decay = float(cfg.model.get("weight_decay", cfg.task.weight_decay))
    parameter_list = list(parameters)
    optimizer_parameters: list[torch.nn.Parameter] | list[dict] = parameter_list

    no_decay_names = frozenset(
        str(parameter_name)
        for parameter_name in getattr(model, "no_weight_decay_parameter_names", ())
    )
    if no_decay_names:
        named_model_parameters = dict(model.named_parameters())
        missing_names = no_decay_names - named_model_parameters.keys()
        if missing_names:
            raise ValueError(
                "Model declared unknown no-weight-decay parameters: "
                + ", ".join(sorted(missing_names))
            )
        no_decay_ids = {id(named_model_parameters[key]) for key in no_decay_names}
        included_ids = {id(parameter) for parameter in parameter_list}
        if no_decay_ids - included_ids:
            raise ValueError("Model no-weight-decay parameters are absent from the optimizer input")
        optimizer_parameters = [
            {"params": [p for p in parameter_list if id(p) not in no_decay_ids]},
            {
                "params": [p for p in parameter_list if id(p) in no_decay_ids],
                "weight_decay": 0.0,
            },
        ]

    if name == "adamw":
        return torch.optim.AdamW(optimizer_parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(optimizer_parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"task.optimizer must be adam|adamw, got {name!r}")


def optimizer_metadata(optimizer: torch.optim.Optimizer) -> dict:
    """Return the effective optimizer and per-group LR/decay actually in use."""
    groups = [
        {
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "parameter_count": sum(int(parameter.numel()) for parameter in group["params"]),
        }
        for group in optimizer.param_groups
    ]
    return {"optimizer": type(optimizer).__name__, "parameter_groups": groups}


def log_optimizer(logger, optimizer: torch.optim.Optimizer) -> dict:
    metadata = optimizer_metadata(optimizer)
    formatted_groups = "; ".join(
        f"group{index}: lr={group['learning_rate']:g}, wd={group['weight_decay']:g}"
        for index, group in enumerate(metadata["parameter_groups"])
    )
    logger.info("Effective optimizer: %s | %s", metadata["optimizer"], formatted_groups)
    return metadata


def scheduler_step(cfg, optimizer: torch.optim.Optimizer, epoch: int, total_epochs: int) -> None:
    """Apply the optional generic warmup-cosine schedule; the frozen default is null."""
    name = str(cfg.task.get("scheduler", "null")).strip().lower()
    if name in {"null", "", "none"}:
        return
    if name != "warmup_cosine":
        raise ValueError(f"task.scheduler must be null|warmup_cosine, got {name!r}")
    warmup = max(int(cfg.task.get("scheduler_warmup_epochs", 10)), 1)
    base_lr = float(cfg.model.get("lr", cfg.task.lr))
    final_lr = float(cfg.task.get("scheduler_min_lr", 1e-5))
    if epoch <= warmup:
        lr = base_lr * epoch / warmup
    else:
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        lr = final_lr + 0.5 * (base_lr - final_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr


def resolve_num_neighbors(cfg) -> list[int]:
    """Resolve sampler fanouts to exactly the configured encoder depth."""
    num_layers = int(cfg.model.get("num_layers", 1))
    if num_layers < 1:
        raise ValueError(f"model.num_layers must be >= 1, got {num_layers}")

    raw_neighbors = cfg.task.get("num_neighbors", -1)
    if isinstance(raw_neighbors, str):
        raw_neighbors = raw_neighbors.strip()
        if raw_neighbors.startswith("[") and raw_neighbors.endswith("]"):
            values = [int(item.strip()) for item in raw_neighbors[1:-1].split(",") if item.strip()]
        else:
            values = [int(raw_neighbors)]
    elif isinstance(raw_neighbors, (list, tuple, ListConfig)):
        values = [int(value) for value in raw_neighbors]
    else:
        values = [int(raw_neighbors)]

    if not values:
        raise ValueError("task.num_neighbors must contain at least one value")
    if len(values) >= num_layers:
        return values[:num_layers]
    return values + [values[-1]] * (num_layers - len(values))
