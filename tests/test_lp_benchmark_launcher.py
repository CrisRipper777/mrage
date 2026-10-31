from __future__ import annotations

import importlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from scripts.run_paper_benchmark import _compose_config


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BASELINES = (
    "gcn", "sage", "mmgcn", "mgat", "dip", "dgf", "dmgc", "rolemag",
    "lgmrec", "graphmore",
)


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("run_lp_baseline_benchmark")


@pytest.mark.parametrize("mrage_fails", [False, True])
def test_lp_execution_order_and_failure_gate(launcher, monkeypatch, tmp_path, mrage_fails):
    commands = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.stdout = io.StringIO("mock run completed\n")
            self.command = command

        def wait(self):
            models = self.command[self.command.index("--models") + 1:self.command.index("--seeds")]
            return int(mrage_fails and models == ["mrage"])

    matrix = importlib.import_module("run_baseline_matrix")
    monkeypatch.setattr(matrix.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(sys, "argv", ["run_lp_baseline_benchmark.py", "--output-root", str(tmp_path), "--resume"])
    assert launcher.main() == int(mrage_fails)
    assert len(commands) == (2 if mrage_fails else 4)
    for index, command in enumerate(commands):
        assert command[command.index("--dataset") + 1] == launcher.DATASETS[index % 2]
        models = command[command.index("--models") + 1:command.index("--seeds")]
        assert models == (["mrage"] if index < 2 else list(EXPECTED_BASELINES))
        assert command[command.index("--seeds") + 1:command.index("--device")] == ["42", "43", "44"]
        assert command[command.index("--device") + 1] == "cuda:1"
        assert command[command.index("--stage") + 1] == "all"
        assert "--resume" in command
        assert "--experiment-config" not in command
    mrage_manifest = json.loads((tmp_path / "_manifests/lp_mrage_benchmark.json").read_text())
    assert mrage_manifest["benchmark"] == "mrage_lp_matrix"
    assert mrage_manifest["status"] == ("failed" if mrage_fails else "complete")
    assert (tmp_path / "_manifests/lp_baseline_benchmark.json").exists() is not mrage_fails


def test_lp_dry_run_has_no_side_effects(launcher, monkeypatch, tmp_path, capsys):
    output_root = tmp_path / "dry-run"
    monkeypatch.setattr(sys, "argv", ["run_lp_baseline_benchmark.py", "--dry-run", "--output-root", str(output_root)])
    assert launcher.main() == 0
    commands = [line for line in capsys.readouterr().out.splitlines() if line.startswith("+ ")]
    assert len(commands) == 4
    assert not output_root.exists()


def test_lp_configs_keep_mrage_defaults_and_official_splits(launcher):
    assert launcher.BASELINE_MODELS == EXPECTED_BASELINES
    default_model = OmegaConf.load(ROOT / "configs/model/mrage.yaml")
    default_task = OmegaConf.load(ROOT / "configs/task/lp.yaml")
    for dataset in launcher.DATASETS:
        splits = set()
        for model in ("mrage", *launcher.BASELINE_MODELS):
            args = SimpleNamespace(
                dataset=dataset, task="lp", model=model, device="cuda:1",
                experiment_config=None, override=list(launcher.PROTOCOL_OVERRIDES),
                split_seed=42,
            )
            for seed in (42, 43, 44):
                cfg = _compose_config(args, seed)
                splits.add(cfg.dataset.edge_split_path)
                assert cfg.seed == seed
                assert cfg.task.name == "lp"
                if model == "mrage":
                    assert OmegaConf.to_container(cfg.model) == OmegaConf.to_container(default_model)
                    assert OmegaConf.to_container(cfg.task) == OmegaConf.to_container(default_task)
        assert len(splits) == 1
        assert Path(next(iter(splits))).name == "lp-edge-split.pt"
