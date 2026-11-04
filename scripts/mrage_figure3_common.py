"""Shared utilities for the MRAGE Figure 3 mechanism analysis.

The analysis is deliberately read-only: it builds the formal ``mrage`` model,
restores a frozen Full checkpoint, and runs evaluation-mode forwards.  No
training, Test evaluation, or model/data mutation is performed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from src.data import load_mag_data
from src.data.semantic_graph_manager import ensure_semantic_graph_cache
from src.models import build_model
from src.tasks.common import load_state_dict_cpu
from src.utils.seeds import set_seed


DATASETS = ("Grocery", "Toys", "Reddit-S", "ele-fashion", "Movies")
DEFAULT_SEEDS = (42, 43, 44)
ADOPTED_SELECTION = (
    PROJECT_ROOT / "experiments/nc5_test_exploration_20261031/adopted_selection.json"
)
DEFAULT_CACHE_ROOT = Path("/hdd1/DataInHere/YHF/mrage/artifacts/semantic_graphs")
DEFAULT_EXTRA_ROOTS = (Path("/hdd1/DataInHere/YHF/mrage/outputs/mrage_figure3_movies_full"),)

# Forward order in `Model.forward`: two private calls per modality followed by
# two collaborative physical calls.  The labels are assigned by call order,
# not by modality/relation IDs (which are identical for private and
# collaborative physical).
CONTEXT_ORDER = (
    "private_text_physical",
    "private_text_semantic",
    "private_visual_physical",
    "private_visual_semantic",
    "collaborative_text_physical",
    "collaborative_visual_physical",
)
EXPECTED_IDS = ((0, 0), (0, 1), (1, 0), (1, 1), (0, 0), (1, 0))
CONTEXT_INDEX = {name: index for index, name in enumerate(CONTEXT_ORDER)}

PRIVATE_PHYSICAL = ("private_text_physical", "private_visual_physical")
PRIVATE_SEMANTIC = ("private_text_semantic", "private_visual_semantic")
COLLABORATIVE_PHYSICAL = (
    "collaborative_text_physical",
    "collaborative_visual_physical",
)

INTERVENTION_GROUPS = {
    "private_physical": PRIVATE_PHYSICAL,
    "private_semantic": PRIVATE_SEMANTIC,
    "collaborative_physical": COLLABORATIVE_PHYSICAL,
    "all_contexts": CONTEXT_ORDER,
}

# The literal initializer in src/models/mrage.py SharedExpertBank.__init__.
INITIAL_THETA = torch.tensor(
    [
        [0.62, -0.27, 0.18, -0.09],
        [-0.31, 0.57, -0.11, 0.19],
        [0.22, 0.16, -0.53, 0.29],
        [-0.18, -0.23, 0.37, 0.46],
    ],
    dtype=torch.float32,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: str | Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalized_coefficients(theta: torch.Tensor, max_order: int) -> torch.Tensor:
    """Reproduce ``SharedExpertBank.coefficients`` for a given maximum order."""
    if theta.ndim != 2:
        raise ValueError("theta must have shape [experts, orders]")
    if max_order < 0 or max_order >= theta.size(1):
        raise ValueError(f"max_order must lie in [0,{theta.size(1) - 1}]")
    order_mask = torch.arange(theta.size(1), device=theta.device) <= int(max_order)
    raw = theta * order_mask.to(theta.dtype)[None, :]
    normalized = raw / (raw.abs().sum(dim=-1, keepdim=True) + 1.0e-8)
    # Return only the orders that participate in this relation.  Higher-order
    # raw values are masked to zero and are not part of the effective filter.
    return normalized[:, : int(max_order) + 1]


def coefficient_rows(
    *,
    dataset: str,
    training_seed: int,
    checkpoint: str,
    theta: torch.Tensor,
) -> list[dict]:
    rows: list[dict] = []
    for relation, max_order in (("physical", 3), ("semantic", 1)):
        initial = normalized_coefficients(INITIAL_THETA, max_order)
        trained = normalized_coefficients(theta.detach().cpu(), max_order)
        delta = trained - initial
        for expert_id in range(4):
            for order in range(max_order + 1):
                rows.append(
                    {
                        "dataset": dataset,
                        "training_seed": int(training_seed),
                        "checkpoint": checkpoint,
                        "relation": relation,
                        "expert_id": expert_id + 1,
                        "order": order,
                        "initialized_coefficient": float(initial[expert_id, order]),
                        "trained_coefficient": float(trained[expert_id, order]),
                        "coefficient_delta": float(delta[expert_id, order]),
                    }
                )
    return rows


def coefficient_diversity_rows(
    *,
    dataset: str,
    training_seed: int,
    checkpoint: str,
    theta: torch.Tensor,
) -> list[dict]:
    rows: list[dict] = []
    for relation, max_order in (("physical", 3), ("semantic", 1)):
        initial = normalized_coefficients(INITIAL_THETA, max_order)
        trained = normalized_coefficients(theta.detach().cpu(), max_order)
        for expert_id in range(4):
            rows.append(
                {
                    "dataset": dataset,
                    "training_seed": int(training_seed),
                    "checkpoint": checkpoint,
                    "metric": "l2_change_from_initialization",
                    "relation": relation,
                    "expert_id": expert_id + 1,
                    "expert_j": "",
                    "context": "",
                    "value": float((trained[expert_id] - initial[expert_id]).norm()),
                    "n_active": "",
                    "n_sampled": "",
                }
            )
            rows.append(
                {
                    "dataset": dataset,
                    "training_seed": int(training_seed),
                    "checkpoint": checkpoint,
                    "metric": "trained_coefficient_l2",
                    "relation": relation,
                    "expert_id": expert_id + 1,
                    "expert_j": "",
                    "context": "",
                    "value": float(trained[expert_id].norm()),
                    "n_active": "",
                    "n_sampled": "",
                }
            )
        for first in range(4):
            for second in range(first + 1, 4):
                distance = float((trained[first] - trained[second]).norm())
                cosine = float(
                    F.cosine_similarity(
                        trained[first].unsqueeze(0),
                        trained[second].unsqueeze(0),
                        dim=-1,
                        eps=1.0e-8,
                    ).item()
                )
                initial_distance = float((initial[first] - initial[second]).norm())
                initial_cosine = float(
                    F.cosine_similarity(
                        initial[first].unsqueeze(0),
                        initial[second].unsqueeze(0),
                        dim=-1,
                        eps=1.0e-8,
                    ).item()
                )
                for metric, value in (
                    ("pairwise_initial_coefficient_l2", initial_distance),
                    ("pairwise_initial_coefficient_cosine", initial_cosine),
                    ("pairwise_coefficient_l2", distance),
                    ("pairwise_coefficient_cosine", cosine),
                ):
                    rows.append(
                        {
                            "dataset": dataset,
                            "training_seed": int(training_seed),
                            "checkpoint": checkpoint,
                            "metric": metric,
                            "relation": relation,
                            "expert_id": first + 1,
                            "expert_j": second + 1,
                            "context": "",
                            "value": value,
                            "n_active": "",
                            "n_sampled": "",
                        }
                    )
    return rows


def primary_subset(context: str) -> str:
    """Name of the primary active set for a routing context."""
    return "semantic_support" if "semantic" in context else "physical_support"


def top2_statistics(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Top-2 routing statistics over the first dimension of ``logits``."""
    if logits.ndim != 2 or logits.size(1) != 4:
        raise ValueError("router logits must have shape [N,4]")
    dense = torch.softmax(logits.float(), dim=-1)
    top_logits, top_ids = torch.topk(logits, 2, dim=-1)
    top_weights = torch.softmax(top_logits.float(), dim=-1)
    route_weights = torch.zeros_like(dense).scatter(1, top_ids, top_weights)
    selected = torch.zeros_like(dense, dtype=torch.bool).scatter(
        1, top_ids, torch.ones_like(top_ids, dtype=torch.bool)
    )
    combinations = top_ids[:, 0] * 4 + top_ids[:, 1]
    unique, counts = torch.unique(combinations, return_counts=True)
    most_common_index = int(counts.argmax())
    most_common_combo = unique[most_common_index].detach()
    node_route_std = route_weights.std(dim=0, unbiased=False)
    return {
        "mean_top2_weight": route_weights.mean(dim=0),
        "top2_selection_frequency": selected.float().mean(dim=0),
        "n_top2_combinations": torch.tensor(int(unique.numel())),
        "most_common_top2_combo": most_common_combo,
        "most_common_top2_combo_label": (
            f"E{int(most_common_combo) // 4 + 1}->E{int(most_common_combo) % 4 + 1}"
        ),
        "most_common_share": counts[most_common_index].float()
        / float(combinations.numel()),
        "node_route_std": node_route_std,
        "mean_node_route_std": node_route_std.mean(),
        "n_nodes": torch.tensor(int(combinations.numel())),
    }


def routing_stats(
    logits: torch.Tensor,
    support: torch.Tensor,
    cross_context: torch.Tensor | None,
) -> dict[str, Any]:
    active = support.to(torch.bool)
    total_nodes = int(logits.size(0))
    n_active = int(active.sum().item())
    result: dict[str, Any] = {
        "n_total_nodes": total_nodes,
        "n_active": n_active,
        "active_ratio": n_active / max(total_nodes, 1),
        "cross_nonzero_count": 0,
        "cross_nonzero_ratio": 0.0,
        "primary": None,
        "cross": None,
    }
    def to_cpu(value: Any) -> Any:
        return value.detach().cpu() if torch.is_tensor(value) else value

    if n_active:
        stats = top2_statistics(logits[active])
        result["primary"] = {key: to_cpu(value) for key, value in stats.items()}
    if cross_context is not None:
        cross_mask = cross_context.abs().sum(dim=-1) > 0
        cross_active = active & cross_mask
        n_cross = int(cross_active.sum().item())
        result["cross_nonzero_count"] = n_cross
        result["cross_nonzero_ratio"] = n_cross / max(total_nodes, 1)
        if n_cross:
            stats = top2_statistics(logits[cross_active])
            result["cross"] = {key: to_cpu(value) for key, value in stats.items()}
    return result


def jensen_shannon_divergence(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.float().flatten()
    second = second.float().flatten()
    first = first / first.sum().clamp_min(1.0e-12)
    second = second / second.sum().clamp_min(1.0e-12)
    midpoint = 0.5 * (first + second)
    eps = 1.0e-12

    def kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left * ((left + eps) / (right + eps)).log()).sum()

    return float(0.5 * kl(first, midpoint) + 0.5 * kl(second, midpoint))


@dataclass
class CheckpointRecord:
    dataset: str
    training_seed: int
    run_dir: Path
    checkpoint: Path
    config: Path
    configuration: str = ""
    source: str = ""

    @property
    def key(self) -> str:
        return f"{self.dataset}/seed-{self.training_seed}"


def discover_adopted_checkpoints(
    *,
    datasets: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    extra_roots: Iterable[str | Path] | None = None,
) -> list[CheckpointRecord]:
    """Locate Full MRAGE checkpoints from the adopted selection manifest."""
    selection = json.loads(ADOPTED_SELECTION.read_text(encoding="utf-8"))
    wanted_datasets = set(datasets) if datasets else set(DATASETS)
    wanted_seeds = set(int(value) for value in (seeds or DEFAULT_SEEDS))
    records: list[CheckpointRecord] = []
    for dataset, payload in selection.get("by_dataset", {}).items():
        if dataset not in wanted_datasets:
            continue
        directories = payload.get("source_directories", [])
        for directory in directories:
            run_dir = Path(directory)
            match = None
            for part in run_dir.parts:
                if part.startswith("seed-"):
                    try:
                        match = int(part.split("-", 1)[1])
                    except ValueError:
                        match = None
            if match is None or match not in wanted_seeds:
                continue
            checkpoint = run_dir / "best_validation.pt"
            config = run_dir / "config_resolved.yaml"
            if not checkpoint.is_file() or not config.is_file():
                continue
            records.append(
                CheckpointRecord(
                    dataset=dataset,
                    training_seed=match,
                    run_dir=run_dir,
                    checkpoint=checkpoint,
                    config=config,
                    configuration=str(payload.get("configuration", "")),
                    source=str(run_dir),
                )
            )
    seen = {str(record.checkpoint) for record in records}
    for record in discover_extra_checkpoints(
        roots=extra_roots if extra_roots is not None else DEFAULT_EXTRA_ROOTS,
        datasets=wanted_datasets,
        seeds=wanted_seeds,
    ):
        if str(record.checkpoint) not in seen:
            records.append(record)
            seen.add(str(record.checkpoint))
    records.sort(key=lambda record: (DATASETS.index(record.dataset), record.training_seed))
    return records


def discover_extra_checkpoints(
    *,
    roots: Iterable[str | Path],
    datasets: Iterable[str],
    seeds: Iterable[int],
) -> list[CheckpointRecord]:
    """Discover Figure-3-only checkpoints outside the adopted-selection JSON."""
    wanted_datasets = set(datasets)
    wanted_seeds = set(int(value) for value in seeds)
    records: list[CheckpointRecord] = []
    for root in roots:
        root_path = Path(root).expanduser()
        if not root_path.is_dir():
            continue
        for checkpoint in sorted(
            root_path.glob("*/nc/mrage/*/seed-*/best_validation.pt")
        ):
            seed_dir = checkpoint.parent
            try:
                seed = int(seed_dir.name.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            if seed not in wanted_seeds:
                continue
            config = seed_dir / "config_resolved.yaml"
            if not config.is_file():
                continue
            cfg = OmegaConf.load(config)
            dataset = str(cfg.dataset.name)
            if dataset not in wanted_datasets:
                continue
            records.append(
                CheckpointRecord(
                    dataset=dataset,
                    training_seed=seed,
                    run_dir=seed_dir,
                    checkpoint=checkpoint,
                    config=config,
                    configuration="figure3_full_movies_validation",
                    source=str(seed_dir),
                )
            )
    return records


def resolve_checkpoint_records(
    *,
    dataset: str | None,
    seeds: Iterable[int],
    run_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    all_datasets: bool = False,
    extra_roots: Iterable[str | Path] | None = None,
) -> list[CheckpointRecord]:
    if run_dir is not None or checkpoint is not None:
        if run_dir is None:
            checkpoint_path = Path(checkpoint).expanduser().resolve()
            run_dir_path = checkpoint_path.parent
        else:
            run_dir_path = Path(run_dir).expanduser().resolve()
            checkpoint_path = (
                Path(checkpoint).expanduser().resolve()
                if checkpoint is not None
                else run_dir_path / "best_validation.pt"
            )
        config_path = run_dir_path / "config_resolved.yaml"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"config not found: {config_path}")
        if dataset is None:
            explicit_cfg = OmegaConf.load(config_path)
            dataset = str(explicit_cfg.dataset.name)
        return [
            CheckpointRecord(
                dataset=str(dataset),
                training_seed=int(seeds[0]),
                run_dir=run_dir_path,
                checkpoint=checkpoint_path,
                config=config_path,
                configuration="explicit",
                source=str(run_dir_path),
            )
        ]
    datasets = DATASETS if all_datasets else ((dataset,) if dataset else DATASETS)
    return discover_adopted_checkpoints(
        datasets=datasets, seeds=seeds, extra_roots=extra_roots
    )


@dataclass
class ModelBundle:
    record: CheckpointRecord
    cfg: Any
    checkpoint: dict[str, Any]
    data: Any
    model: nn.Module
    head: nn.Module
    cache_metadata: dict[str, Any]
    x: torch.Tensor
    edge_index: torch.Tensor
    device: torch.device
    checkpoint_sha256: str
    baseline: dict[str, float] = field(default_factory=dict)


def load_model_bundle(
    record: CheckpointRecord,
    *,
    device: torch.device,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    cache_mode: str = "existing",
) -> ModelBundle:
    cfg = OmegaConf.load(record.config)
    checkpoint = torch.load(record.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("task") != "nc":
        raise ValueError(f"checkpoint task is not NC: {record.checkpoint}")
    if str(checkpoint.get("model")) != "mrage":
        raise ValueError(f"checkpoint model is not mrage: {record.checkpoint}")
    if str(checkpoint.get("dataset")) != record.dataset:
        raise ValueError(
            f"checkpoint dataset {checkpoint.get('dataset')!r} != {record.dataset!r}"
        )
    if int(checkpoint.get("seed", -1)) != int(record.training_seed):
        raise ValueError(
            f"checkpoint seed {checkpoint.get('seed')} != {record.training_seed}"
        )
    if str(cfg.model.name).lower() != "mrage":
        raise ValueError(f"resolved config model is not mrage: {record.config}")

    data = load_mag_data(cfg, "nc", 42)
    set_seed(int(checkpoint["seed"]))
    model = build_model(cfg, checkpoint["data_info"])

    cache_config = OmegaConf.create(OmegaConf.to_container(cfg.semantic_graph, resolve=True))
    cache_config.mode = str(cache_mode)
    cache_config.cache_dir = str(Path(cache_root).expanduser().resolve())
    semantic_cache = ensure_semantic_graph_cache(
        dataset=str(data.name),
        task="nc",
        text_features=data.x_t,
        visual_features=data.x_i,
        physical_edge_index=data.edge_index,
        config=cache_config,
        project_root=PROJECT_ROOT,
    )
    cache_metadata = model.load_semantic_cache(
        str(semantic_cache.path),
        expected_dataset=str(data.name),
        expected_feature_fingerprint=str(semantic_cache.metadata["feature_fingerprint"]),
        expected_candidate_fingerprint=str(
            semantic_cache.metadata["candidate_fingerprint"]
        ),
        expected_v7b_fingerprint=str(semantic_cache.metadata["v7b_fingerprint"]),
        expected_identity_fingerprint=str(semantic_cache.identity_fingerprint),
    )
    model = model.to(device)
    load_state_dict_cpu(model, checkpoint["model_state"])
    head = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    load_state_dict_cpu(head, checkpoint["head_state"])
    model.eval()
    head.eval()
    model._batch_n_id = None

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    return ModelBundle(
        record=record,
        cfg=cfg,
        checkpoint=checkpoint,
        data=data,
        model=model,
        head=head,
        cache_metadata=dict(cache_metadata or {}),
        x=x,
        edge_index=edge_index,
        device=device,
        checkpoint_sha256=sha256_file(record.checkpoint),
    )


def forward_full(bundle: ModelBundle) -> torch.Tensor:
    """One evaluation-mode full-graph forward; returns node embeddings."""
    bundle.model._batch_n_id = None
    embeddings, _, _, _, _ = bundle.model(bundle.x, bundle.edge_index)
    return embeddings


def validation_metrics(
    embeddings: torch.Tensor,
    head: nn.Module,
    data: Any,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor]:
    from sklearn.metrics import f1_score

    val_idx = data.val_idx.to(device)
    labels = data.y.to(device)[val_idx]
    logits = head(embeddings[val_idx].float())
    ce = float(F.cross_entropy(logits, labels).item())
    predictions = logits.argmax(dim=-1)
    accuracy = float((predictions == labels).float().mean().item())
    macro_f1 = float(
        f1_score(
            labels.detach().cpu().numpy(),
            predictions.detach().cpu().numpy(),
            labels=list(range(int(data.num_classes))),
            average="macro",
            zero_division=0,
        )
    )
    if not all(math.isfinite(value) for value in (ce, accuracy, macro_f1)):
        raise FloatingPointError("non-finite validation metric")
    return (
        {
            "val_ce": ce,
            "val_acc": accuracy,
            "val_macro_f1": macro_f1,
        },
        predictions.detach().cpu(),
    )


class ContextCapture:
    """Capture expert outputs/router logits and optionally shuffle router rows."""

    def __init__(
        self,
        model: nn.Module,
        *,
        capture_experts: bool = False,
        capture_routing: bool = False,
        shuffle_capable: bool = False,
        max_nodes: int = 4096,
        sample_seed: int = 2026,
    ):
        self.model = model
        self.capture_experts = bool(capture_experts)
        self.capture_routing = bool(capture_routing)
        self.shuffle_capable = bool(shuffle_capable)
        self.max_nodes = int(max_nodes)
        self.sample_seed = int(sample_seed)
        self.original_filter = None
        self.filter_handle = None
        self.router_handle = None
        self.call_index = 0
        self.current_context: str | None = None
        self.current_support: torch.Tensor | None = None
        self.current_cross: torch.Tensor | None = None
        self.expert_records: dict[str, dict[str, Any]] = {}
        self.routing_records: dict[str, dict[str, Any]] = {}
        self.shuffle_group: tuple[str, ...] = ()
        self.shuffle_enabled = False
        self.shuffle_mode = "none"
        self.shuffle_generator = torch.Generator().manual_seed(0)
        self.last_shuffle_metadata: list[dict[str, Any]] = []

    def install(self) -> "ContextCapture":
        model = self.model
        self.original_filter = model._filter

        original = self.original_filter

        def wrapped_filter(
            module,
            basis,
            support,
            current,
            relation_id,
            modality_id,
            cross_context,
        ):
            index = self.call_index
            if index >= len(CONTEXT_ORDER):
                raise RuntimeError(
                    f"unexpected extra filter call (index={index}); expected 6"
                )
            expected = EXPECTED_IDS[index]
            if (int(modality_id), int(relation_id)) != expected:
                raise RuntimeError(
                    "unexpected filter call order: "
                    f"index={index}, got ({modality_id},{relation_id}), "
                    f"expected {expected}"
                )
            self.current_context = CONTEXT_ORDER[index]
            self.current_support = support
            self.current_cross = cross_context
            self.call_index += 1
            return original(basis, support, current, relation_id, modality_id, cross_context)

        model._filter = types.MethodType(wrapped_filter, model)
        if self.capture_experts:
            self.filter_handle = model.filter_bank.register_forward_hook(
                self._expert_hook
            )
        if self.capture_routing or self.shuffle_capable:
            self.router_handle = model.router.register_forward_hook(self._router_hook)
        return self

    def begin_forward(
        self,
        *,
        shuffle_group: Iterable[str] | None = None,
        shuffle_seed: int | None = None,
        shuffle_mode: str = "none",
    ) -> None:
        self.call_index = 0
        self.current_context = None
        self.current_support = None
        self.current_cross = None
        self.shuffle_group = tuple(shuffle_group or ())
        self.shuffle_enabled = shuffle_mode in {"shuffle", "identity"}
        self.shuffle_mode = shuffle_mode
        self.last_shuffle_metadata = []
        if shuffle_seed is not None:
            self.shuffle_generator = torch.Generator().manual_seed(int(shuffle_seed))

    def end_forward(self) -> None:
        if self.call_index != len(CONTEXT_ORDER):
            raise RuntimeError(
                f"expected {len(CONTEXT_ORDER)} filter calls, saw {self.call_index}"
            )

    def _sample_active(self, active: torch.Tensor) -> torch.Tensor:
        if active.numel() <= self.max_nodes:
            return active
        generator = torch.Generator().manual_seed(self.sample_seed)
        order = torch.randperm(active.numel(), generator=generator)[: self.max_nodes]
        return active[order.to(active.device)]

    def _expert_hook(self, module, inputs, output):
        if self.current_context is None:
            return None
        outputs, _, _, _ = output
        support = self.current_support
        if support is None:
            return None
        active = support.to(torch.bool).nonzero(as_tuple=False).flatten()
        n_active = int(active.numel())
        if n_active == 0:
            self.expert_records[self.current_context] = {
                "n_active": 0,
                "n_sampled": 0,
                "cosine": {},
                "mean_norm": {},
            }
            return None
        sampled = self._sample_active(active)
        values = outputs[:, sampled, :].detach().float()
        norms = values.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        normalized = values / norms
        cosine: dict[tuple[int, int], float] = {}
        for first in range(4):
            for second in range(first + 1, 4):
                cosine[(first, second)] = float(
                    (normalized[first] * normalized[second]).sum(dim=-1).mean().item()
                )
        self.expert_records[self.current_context] = {
            "n_active": n_active,
            "n_sampled": int(sampled.numel()),
            "cosine": cosine,
            "mean_norm": {
                expert_id: float(values[expert_id].norm(dim=-1).mean().item())
                for expert_id in range(4)
            },
        }
        return None

    def _router_hook(self, module, inputs, output):
        if self.current_context is None:
            return output
        context = self.current_context
        if self.shuffle_enabled and context in self.shuffle_group:
            support = self.current_support
            if support is None:
                return output
            active = support.to(torch.bool)
            if context in COLLABORATIVE_PHYSICAL and self.current_cross is not None:
                active = active & (self.current_cross.abs().sum(dim=-1) > 0)
            indices = active.nonzero(as_tuple=False).flatten()
            if indices.numel() >= 2:
                if self.shuffle_mode == "identity":
                    permutation = torch.arange(
                        indices.numel(), device=indices.device
                    )
                elif self.shuffle_mode == "shuffle":
                    permutation = torch.randperm(
                        indices.numel(), generator=self.shuffle_generator
                    ).to(indices.device)
                else:
                    permutation = None
                if permutation is not None:
                    shuffled = output.clone()
                    shuffled[indices] = output[indices[permutation]]
                    changed = int(
                        (permutation != torch.arange(indices.numel(), device=indices.device))
                        .sum()
                        .item()
                    )
                    self.last_shuffle_metadata.append(
                        {
                            "context": context,
                            "n_active": int(indices.numel()),
                            "n_changed": changed,
                        }
                    )
                    return shuffled
        if self.capture_routing and context not in self.routing_records:
            self.routing_records[context] = routing_stats(
                output.detach(), self.current_support, self.current_cross
            )
        return output

    def remove(self) -> None:
        if self.filter_handle is not None:
            self.filter_handle.remove()
            self.filter_handle = None
        if self.router_handle is not None:
            self.router_handle.remove()
            self.router_handle = None
        if self.original_filter is not None:
            self.model._filter = self.original_filter
            self.original_filter = None


def routing_rows(
    *,
    dataset: str,
    training_seed: int,
    checkpoint: str,
    records: dict[str, dict[str, Any]],
) -> tuple[list[dict], list[dict], dict[str, torch.Tensor]]:
    pattern_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    mean_distributions: dict[str, torch.Tensor] = {}
    for context in CONTEXT_ORDER:
        record = records.get(context)
        if record is None:
            continue
        primary_name = primary_subset(context)
        for subset_name, stats in (
            (primary_name, record.get("primary")),
            ("cross_nonzero", record.get("cross")),
        ):
            if stats is None:
                continue
            mean_weights = stats["mean_top2_weight"].float()
            frequencies = stats["top2_selection_frequency"].float()
            node_route_std = stats.get("node_route_std")
            mean_distributions[f"{context}:{subset_name}"] = mean_weights
            for expert_id in range(4):
                pattern_rows.append(
                    {
                        "dataset": dataset,
                        "training_seed": int(training_seed),
                        "checkpoint": checkpoint,
                        "context": context,
                        "subset": subset_name,
                        "expert_id": expert_id + 1,
                        "mean_top2_weight": float(mean_weights[expert_id]),
                        "top2_selection_frequency": float(frequencies[expert_id]),
                        "active_node_count": int(record["n_active"]),
                        "active_node_ratio": float(record["active_ratio"]),
                        "cross_nonzero_count": int(record["cross_nonzero_count"]),
                        "cross_nonzero_ratio": float(record["cross_nonzero_ratio"]),
                        "mean_node_route_std": float(
                            node_route_std[expert_id]
                            if node_route_std is not None
                            else stats["mean_node_route_std"]
                        ),
                    }
                )
            diagnostic_rows.append(
                {
                    "dataset": dataset,
                    "training_seed": int(training_seed),
                    "checkpoint": checkpoint,
                    "context": context,
                    "subset": subset_name,
                    "active_node_count": int(record["n_active"]),
                    "active_node_ratio": float(record["active_ratio"]),
                    "n_top2_combinations": int(stats["n_top2_combinations"]),
                    "most_common_top2_combo": int(stats["most_common_top2_combo"]),
                    "most_common_top2_combo_label": str(
                        stats["most_common_top2_combo_label"]
                    ),
                    "most_common_share": float(stats["most_common_share"]),
                    "mean_node_route_std": float(stats["mean_node_route_std"]),
                    "cross_nonzero_count": int(record["cross_nonzero_count"]),
                    "cross_nonzero_ratio": float(record["cross_nonzero_ratio"]),
                }
            )
    for row in diagnostic_rows:
        key = f"{row['context']}:{row['subset']}"
        reference = mean_distributions.get(key)
        for other_context in CONTEXT_ORDER:
            other_key = f"{other_context}:{primary_subset(other_context)}"
            other = mean_distributions.get(other_key)
            row[f"jsd_vs_{other_context}"] = (
                jensen_shannon_divergence(reference, other)
                if reference is not None and other is not None
                else ""
            )
    return pattern_rows, diagnostic_rows, mean_distributions


def checkpoint_manifest_row(bundle: ModelBundle, metrics: dict[str, float]) -> dict[str, Any]:
    checkpoint = bundle.checkpoint
    return {
        "dataset": bundle.record.dataset,
        "training_seed": int(bundle.record.training_seed),
        "checkpoint": str(bundle.record.checkpoint),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "config": str(bundle.record.config),
        "configuration": bundle.record.configuration,
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "saved_val_acc": float(checkpoint.get("metrics", {}).get("val_acc", float("nan"))),
        "saved_val_macro_f1": float(
            checkpoint.get("metrics", {}).get("val_macro_f1", float("nan"))
        ),
        "recomputed_val_ce": float(metrics["val_ce"]),
        "recomputed_val_acc": float(metrics["val_acc"]),
        "recomputed_val_macro_f1": float(metrics["val_macro_f1"]),
        "semantic_candidate_fingerprint": bundle.checkpoint.get("run_metadata", {}).get(
            "semantic_candidate_fingerprint"
        ),
        "semantic_v7b_fingerprint": bundle.checkpoint.get("run_metadata", {}).get(
            "semantic_v7b_fingerprint"
        ),
        "device": str(bundle.device),
        "cache_identity": bundle.cache_metadata.get("identity_fingerprint"),
    }


def set_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {name}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device
