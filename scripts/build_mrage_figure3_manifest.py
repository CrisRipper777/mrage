#!/usr/bin/env python3
"""Assemble the reproducibility manifest for the Figure 3 analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mrage_figure3_common import (  # noqa: E402
    ADOPTED_SELECTION,
    DATASETS,
    DEFAULT_CACHE_ROOT,
    DEFAULT_SEEDS,
    discover_adopted_checkpoints,
    sha256_file,
    write_json,
)


CODE_FILES = (
    "scripts/mrage_figure3_common.py",
    "scripts/analyze_mrage_experts.py",
    "scripts/analyze_mrage_router.py",
    "scripts/analyze_mrage_router_shuffle.py",
    "scripts/plot_mrage_figure3.py",
    "scripts/build_mrage_figure3_manifest.py",
    "tests/test_mrage_figure3_analysis.py",
)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/mrage_figure3_mechanism/data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments/mrage_figure3_mechanism/manifest.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-nodes", type=int, default=4096)
    parser.add_argument("--shuffle-repeats", type=int, default=10)
    parser.add_argument("--shuffle-seed-base", type=int, default=1000)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    experts = _read_json(data_dir / "manifest_experts.json")
    router = _read_json(data_dir / "manifest_router.json")
    shuffle = _read_json(data_dir / "manifest_shuffle.json")
    records = discover_adopted_checkpoints(
        datasets=DATASETS, seeds=DEFAULT_SEEDS
    )
    checkpoints = []
    for record in records:
        run_metadata_path = record.run_dir / "run_metadata.json"
        run_metadata = _read_json(run_metadata_path)
        config_hash = run_metadata.get("config_hash")
        checkpoint = {
            "dataset": record.dataset,
            "training_seed": int(record.training_seed),
            "checkpoint": str(record.checkpoint),
            "checkpoint_sha256": sha256_file(record.checkpoint),
            "config": str(record.config),
            "config_sha256": sha256_file(record.config),
            "run_metadata": str(run_metadata_path) if run_metadata_path.is_file() else None,
            "config_hash": config_hash,
            "configuration": record.configuration,
        }
        checkpoints.append(checkpoint)
    code_hashes = {
        relative: sha256_file(PROJECT_ROOT / relative) for relative in CODE_FILES
    }
    payload = {
        "figure": "Figure 3 — Mechanistic Analysis of Shared Adaptive Graph Experts",
        "branch": "codex/mrage-figure3-mechanism",
        "baseline_commit": "dce34f8",
        "manifest_git_head": _git_head(),
        "adopted_selection": str(ADOPTED_SELECTION),
        "analysis_config": {
            "datasets": list(DATASETS),
            "training_seeds": list(DEFAULT_SEEDS),
            "device": args.device,
            "max_nodes_for_expert_output_similarity": int(args.max_nodes),
            "shuffle_repeats": int(args.shuffle_repeats),
            "shuffle_seed_base": int(args.shuffle_seed_base),
            "intervention_groups": [
                "private_physical",
                "private_semantic",
                "collaborative_physical",
                "all_contexts",
            ],
            "contexts": [
                "private_text_physical",
                "private_text_semantic",
                "private_visual_physical",
                "private_visual_semantic",
                "collaborative_text_physical",
                "collaborative_visual_physical",
            ],
            "semantic_cache_root": str(DEFAULT_CACHE_ROOT),
            "validation_only": True,
            "test_evaluated": False,
            "retraining": False,
        },
        "code_sha256": code_hashes,
        "checkpoints": checkpoints,
        "analysis_runs": {
            "experts": experts,
            "router": router,
            "shuffle": shuffle,
        },
        "notes": [
            "Movies is skipped because the adopted historical Full run has no saved checkpoint.",
            "All metrics are Validation-only; the official Test path was not called.",
            "No model weights, semantic caches, or data tensors were modified.",
            "Shuffle trials permute complete four-dimensional router-logit rows within active nodes.",
        ],
    }
    write_json(args.output, payload)
    print(f"Wrote {args.output} with {len(checkpoints)} checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
