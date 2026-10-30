from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts.collect_results import collect
from scripts.run_paper_benchmark import _compose_config, _config_hash, _prepare_run
from src.data import MAGData
from src.data.loaders import _load_magb
from src.data.splits import generate_nc_split
from src.tasks.lp import _evaluate_split as evaluate_lp_split
from src.tasks.nc import _resolve_nc_eval_labels
from src.tasks.common import build_optimizer, optimizer_metadata
from src.utils.run_artifacts import write_json_atomic
from src.utils.seeds import set_seed
from src.utils.timing import PhaseProfiler


def test_nc_split_uses_fixed_split_seed_independent_of_training_seed(tmp_path, monkeypatch):
    n = 60
    labels = torch.arange(n) % 3
    split_path = tmp_path / "Movies_nc_seed42_train0.6_val0.2.pt"
    torch.save(generate_nc_split(labels, 0.6, 0.2, 42), split_path)
    edge_index = torch.stack((torch.arange(n), (torch.arange(n) + 1) % n))
    monkeypatch.setattr(
        "src.data.loaders._load_dgl_graph",
        lambda path: (edge_index, labels, n),
    )
    monkeypatch.setattr(
        "src.data.loaders._load_numpy_feature",
        lambda path, dtype="float32": torch.ones((n, 2 if "image" in str(path) else 3)),
    )
    cfg = OmegaConf.create(
        {
            "dataset_split_seed": 42,
            "dataset": {
                "name": "Movies", "source": "magb", "graph_path": str(tmp_path / "g.pt"),
                "image_feat_path": str(tmp_path / "image.npy"),
                "text_feat_path": str(tmp_path / "text.npy"),
                "nc_split_path": str(split_path), "nc_train_ratio": 0.6,
                "nc_val_ratio": 0.2, "num_classes": 3, "feature_dtype": "float32",
                "make_undirected": True, "add_self_loops": False,
                "auto_generate_splits": False,
            },
        }
    )
    loaded = [_load_magb(cfg, "nc", train_seed) for train_seed in (42, 43, 44)]
    assert len({data.info["nc_split_path"] for data in loaded}) == 1
    assert len({data.info["nc_split_sha256"] for data in loaded}) == 1
    for attr in ("train_idx", "val_idx", "test_idx"):
        assert all(torch.equal(getattr(loaded[0], attr), getattr(data, attr)) for data in loaded[1:])

    initial_weights = []
    for train_seed in (42, 43, 44):
        set_seed(train_seed)
        initial_weights.append(nn.Linear(3, 5).weight.detach().clone())
    assert not torch.equal(initial_weights[0], initial_weights[1])
    assert not torch.equal(initial_weights[1], initial_weights[2])


def test_nc_macro_f1_class_ids_do_not_depend_on_test_labels():
    labels = torch.tensor([0, 1, 0, 1, 2, 2])
    data = MAGData(
        name="tiny", source="synthetic", task="nc", x=torch.ones(6, 2),
        edge_index=torch.empty((2, 0), dtype=torch.long), y=labels,
        train_idx=torch.tensor([0, 1]), val_idx=torch.tensor([2, 3]),
        test_idx=torch.tensor([4, 5]), num_nodes=6, num_classes=3,
    )
    first = _resolve_nc_eval_labels(data)
    data.y[data.test_idx] = 1
    second = _resolve_nc_eval_labels(data)
    assert first == second == [0, 1, 2]
    assert _resolve_nc_eval_labels(data, fixed_labels=[0, 2]) == [0, 2]


class _PairScorer(nn.Module):
    def score_pairs(self, source, target):
        return (source * target).sum(dim=-1)

    def forward(self, pair_feature):
        return pair_feature.sum(dim=-1)


def test_lp_validation_scoring_is_no_grad_and_matches_reference():
    z = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]],
        requires_grad=True,
    )
    predictor = _PairScorer()
    split = {
        "source_node": torch.tensor([0, 3]),
        "target_node": torch.tensor([1, 2]),
        "target_node_neg": torch.tensor([[1, 2, 4], [0, 1, 2]]),
    }
    with torch.enable_grad():
        actual = evaluate_lp_split(z, predictor, split, torch.device("cpu"), 1)

    ranks = []
    for source, target, negatives in zip(
        split["source_node"], split["target_node"], split["target_node_neg"]
    ):
        pos = predictor.score_pairs(z[source : source + 1], z[target : target + 1])
        neg = predictor(z[source : source + 1].unsqueeze(1) * z[negatives].unsqueeze(0)).view(1, -1)
        ranks.append(1 + int((neg >= pos.view(-1, 1)).sum()))
    expected = {
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "hits@1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "hits@3": sum(rank <= 3 for rank in ranks) / len(ranks),
        "hits@10": sum(rank <= 10 for rank in ranks) / len(ranks),
    }
    assert actual == expected
    assert z.grad is None
    assert all(parameter.grad is None for parameter in predictor.parameters())


def test_disabled_profiler_does_not_synchronize_cuda(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "synchronize",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected sync")),
    )
    profiler = PhaseProfiler(False, torch.device("cuda:0"))
    profiler.start_epoch()
    with profiler.measure("phase"):
        pass
    assert profiler.finish_epoch() == 0.0


def test_optimizer_preset_precedence_and_mrage_task_lr():
    parameter = nn.Parameter(torch.ones(2))
    cfg = OmegaConf.create(
        {
            "task": {"optimizer": "adamw", "lr": 1e-4, "weight_decay": 2e-4},
            "model": {"name": "mrage"},
        }
    )
    optimizer = build_optimizer([parameter], cfg)
    metadata = optimizer_metadata(optimizer)
    assert metadata["parameter_groups"][0]["learning_rate"] == 1e-4
    assert metadata["parameter_groups"][0]["weight_decay"] == 2e-4

    cfg.model.lr = 5e-4
    cfg.model.weight_decay = 3e-4
    optimizer = build_optimizer([parameter], cfg)
    metadata = optimizer_metadata(optimizer)
    assert metadata["parameter_groups"][0]["learning_rate"] == 5e-4
    assert metadata["parameter_groups"][0]["weight_decay"] == 3e-4


def test_paper_runner_uses_fixed_nc_split_and_seed_independent_config_hash(tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(
        dataset="Movies", task="nc", model="mrage", device="cuda:0",
        experiment_config=None, override=[], split_seed=42, seeds=[42, 43, 44],
    )
    nc42, nc43 = _compose_config(args, 42), _compose_config(args, 43)
    assert nc42.dataset.nc_split_path == nc43.dataset.nc_split_path
    assert "seed42" in nc42.dataset.nc_split_path
    assert _config_hash(nc42) == _config_hash(nc43)

    args.task = "lp"
    lp42, lp43 = _compose_config(args, 42), _compose_config(args, 43)
    assert lp42.dataset.lp_split_path != lp43.dataset.lp_split_path
    assert _config_hash(lp42) == _config_hash(lp43)
    cfg, path, metadata = _prepare_run(
        args, 42, tmp_path / "lp-run", _config_hash(lp42), "source-sha",
    )
    assert cfg.task.evaluate_test is False
    assert cfg.task.phase == "train"
    assert cfg.task.checkpoint_path.endswith("best_validation.pt")
    assert "mask_test_labels" not in cfg.task
    assert path.is_file()
    assert _config_hash(OmegaConf.load(path)) == _config_hash(lp42)
    assert metadata["expected_seeds"] == [42, 43, 44]


def _write_run(root: Path, name: str, *, status: str, split: str, config: str, seed: int):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    metadata = {
        "dataset": "Movies", "task": "nc", "model": "mrage",
        "training_seed": seed, "expected_seeds": [42, 43, 44],
        "split_id": split, "protocol_version": "nc_v1", "config_hash": config,
        "best_epoch": 12,
    }
    write_json_atomic(run_dir / "run_metadata.json", metadata)
    count = 1 if status == "final_test_complete" else 0
    write_json_atomic(
        run_dir / "status.json",
        {"status": status, "training_seed": seed, "test_evaluation_count": count},
    )
    if count:
        write_json_atomic(
            run_dir / "final_test_metrics.json",
            {
                "dataset": "Movies", "task": "nc", "model": "mrage", "seed": seed,
                "split_id": split, "protocol_version": "nc_v1", "best_epoch": 12,
                "test_evaluation_count": 1,
                "metrics": {"val_acc": 0.8, "test_acc": 0.7, "test_macro_f1": 0.6},
            },
        )
    return run_dir


def test_collector_excludes_incomplete_runs_and_keeps_split_config_groups(tmp_path):
    _write_run(tmp_path, "done42", status="final_test_complete", split="split-a", config="cfg-a", seed=42)
    _write_run(tmp_path, "valid43", status="validation_complete", split="split-a", config="cfg-a", seed=43)
    _write_run(tmp_path, "done44-other", status="final_test_complete", split="split-b", config="cfg-b", seed=44)
    runs, summaries, statuses = collect(tmp_path)
    test_accuracy = [row for row in runs if row["metric_name"] == "test_acc"]
    assert len(test_accuracy) == 2
    assert len(summaries) == 4
    group_a = next(row for row in summaries if row["split_id"] == "split-a" and row["metric_name"] == "test_acc")
    assert group_a["num_valid_runs"] == 1
    assert group_a["expected_seeds"] == "42;43;44"
    assert group_a["missing_seeds"] == "43;44"
    assert len(statuses) == 3
