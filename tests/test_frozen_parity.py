from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from src.data.semantic_graph_manager import ensure_semantic_graph_cache


EXPECTED_SOURCE_COMMIT = "8f1f9c490907583b003beb73b72e873d62f16c3d"


def _max_errors(reference: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    if reference.shape != actual.shape:
        raise AssertionError(f"shape mismatch: {tuple(reference.shape)} != {tuple(actual.shape)}")
    if reference.numel() == 0:
        return 0.0, 0.0
    if not reference.is_floating_point():
        if not torch.equal(reference, actual):
            raise AssertionError("integer/bool tensors are not identical")
        return 0.0, 0.0
    delta = (reference.double() - actual.double()).abs()
    max_abs = float(delta.max())
    max_rel = float((delta / reference.double().abs().clamp_min(1.0e-12)).max())
    return max_abs, max_rel


def _compare_tensor_dict(name: str, reference: dict, actual: dict, *, rtol=2.0e-5, atol=2.0e-6):
    if set(reference) != set(actual):
        raise AssertionError(
            f"{name} keys differ: reference-only={sorted(set(reference)-set(actual))}, "
            f"target-only={sorted(set(actual)-set(reference))}"
        )
    max_abs = max_rel = 0.0
    for key in reference:
        ref, got = reference[key], actual[key]
        if ref is None or got is None:
            if ref is not None or got is not None:
                raise AssertionError(f"{name}.{key} has a gradient-presence mismatch")
            continue
        abs_error, rel_error = _max_errors(ref, got)
        max_abs, max_rel = max(max_abs, abs_error), max(max_rel, rel_error)
        if ref.is_floating_point() and not torch.allclose(ref, got, rtol=rtol, atol=atol):
            raise AssertionError(
                f"{name}.{key} mismatch: max_abs={abs_error:.8g}, max_rel={rel_error:.8g}"
            )
    return max_abs, max_rel


def _make_inputs(cache_root: Path, project_root: Path, source_root: Path):
    torch.set_num_threads(1)
    n = 24
    generator = torch.Generator().manual_seed(1729)
    text = torch.randn(n, 7, generator=generator)
    visual = torch.randn(n, 5, generator=generator)
    x = torch.cat((text, visual), dim=1)
    nodes = torch.arange(n)
    near = torch.stack((nodes, (nodes + 1) % n))
    far = torch.stack((nodes, (nodes + 3) % n))
    train_graph = torch.cat((near, near.flip(0)), dim=1).contiguous()
    physical = torch.cat((near, far, near.flip(0), far.flip(0)), dim=1).contiguous()
    config = {
        "mode": "auto",
        "knn_k": 8,
        "cache_dir": str(cache_root),
        "validate": True,
        "faiss_seed": 2026,
        "faiss_threads": 1,
    }
    nc_cache = ensure_semantic_graph_cache(
        dataset="Parity", task="nc", text_features=text, visual_features=visual,
        physical_edge_index=physical, config=config, project_root=project_root,
    )
    lp_cache = ensure_semantic_graph_cache(
        dataset="Parity", task="lp", text_features=text, visual_features=visual,
        physical_edge_index=train_graph, config=config, project_root=project_root,
    )
    payload = {
        "x": x,
        "text": text,
        "visual": visual,
        "physical": physical,
        "train_graph": train_graph,
        "labels": torch.arange(n, dtype=torch.long) % 3,
        "train_idx": torch.arange(18, dtype=torch.long),
        "data_info": {"input_dim": 12, "num_nodes": n, "num_classes": 3, "text_dim": 7, "visual_dim": 5},
        "nc_cache_path": str(nc_cache.path),
        "lp_cache_path": str(lp_cache.path),
        "feature_fingerprint": nc_cache.metadata["feature_fingerprint"],
        "candidate_fingerprint": nc_cache.metadata["candidate_fingerprint"],
        "nc_v7b_fingerprint": nc_cache.metadata["v7b_fingerprint"],
        "lp_v7b_fingerprint": lp_cache.metadata["v7b_fingerprint"],
        "nc_identity_fingerprint": nc_cache.identity_fingerprint,
        "lp_identity_fingerprint": lp_cache.identity_fingerprint,
        "lp_edge_label_index": torch.tensor([[0, 1, 0, 3], [1, 2, 12, 19]], dtype=torch.long),
        "lp_edge_label": torch.tensor([1.0, 1.0, 0.0, 0.0]),
    }
    input_path = cache_root / "parity_inputs.pt"
    torch.save(payload, input_path)
    return input_path


def _run_worker(side: str, project_root: Path, source_root: Path, input_path: Path, output_path: Path):
    worker = project_root / "tests/frozen_parity_worker.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root if side == "reference" else project_root)
    completed = subprocess.run(
        [sys.executable, str(worker), side, str(project_root), str(source_root), str(input_path), str(output_path)],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{side} parity worker failed (exit {completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else project_root.parent / "model_research"
    if not source_root.is_dir():
        raise FileNotFoundError(f"source repository not found: {source_root}")
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_SOURCE_COMMIT:
        raise RuntimeError(f"expected source HEAD {EXPECTED_SOURCE_COMMIT}, got {commit}")

    with tempfile.TemporaryDirectory(prefix="mrage-frozen-parity-") as temporary:
        temp_root = Path(temporary)
        input_path = _make_inputs(temp_root / "semantic_graphs", project_root, source_root)
        reference_path, target_path = temp_root / "reference.pt", temp_root / "target.pt"
        _run_worker("reference", project_root, source_root, input_path, reference_path)
        _run_worker("target", project_root, source_root, input_path, target_path)
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        target = torch.load(target_path, map_location="cpu", weights_only=False)

        reports = []
        for name in ("initial_model_state", "initial_classifier_state"):
            left, right = reference["nc"][name], target["nc"][name]
            extra = set(right) - set(left)
            missing = set(left) - set(right)
            if name == "initial_model_state":
                if extra != {"semantic_graph_identity_state"} or missing:
                    raise AssertionError(f"unexpected state key migration: extra={extra}, missing={missing}")
                right = {key: right[key] for key in left}
            elif extra or missing:
                raise AssertionError(f"classifier state keys differ: extra={extra}, missing={missing}")
            reports.append((name, *_compare_tensor_dict(name, left, right, rtol=0.0, atol=0.0)))

        for label in ("eval",):
            reports.append((f"NC {label}", *_compare_tensor_dict(label, reference["nc"][label], target["nc"][label])))
        for name in ("training_loss",):
            abs_error, rel_error = _max_errors(reference["nc"][name], target["nc"][name])
            if not torch.allclose(reference["nc"][name], target["nc"][name], rtol=2.0e-5, atol=2.0e-6):
                raise AssertionError(f"NC {name} mismatch: max_abs={abs_error:.8g}, max_rel={rel_error:.8g}")
            reports.append((f"NC {name}", abs_error, rel_error))
        for name in ("gradients", "updated_model_state", "updated_classifier_state"):
            left, right = reference["nc"][name], target["nc"][name]
            if name == "updated_model_state":
                right = {key: right[key] for key in left}
            reports.append((f"NC {name}", *_compare_tensor_dict(name, left, right, rtol=5.0e-5, atol=5.0e-6)))

        reports.append(("LP sampled context", *_compare_tensor_dict(
            "LP context", reference["lp"]["context"], target["lp"]["context"], rtol=0.0, atol=0.0
        )))
        for name in ("node_embeddings", "predictions"):
            left, right = reference["lp"][name], target["lp"][name]
            abs_error, rel_error = _max_errors(left, right)
            if not torch.allclose(left, right, rtol=2.0e-5, atol=2.0e-6):
                raise AssertionError(f"LP {name} mismatch: max_abs={abs_error:.8g}, max_rel={rel_error:.8g}")
            reports.append((f"LP {name}", abs_error, rel_error))
        if reference["lp"]["sampled_context_metadata"] != target["lp"]["sampled_context_metadata"]:
            raise AssertionError("LP sampled context metadata differs")

        print(f"Source commit: {commit}")
        print("Shared model state keys: identical; target adds semantic_graph_identity_state for strict checkpoint binding")
        for name, max_abs, max_rel in reports:
            print(f"{name}: max_abs={max_abs:.8g}, max_rel={max_rel:.8g}")
        print("PASS: NC forward/loss/gradient/update and LP sampled context/predictions match")


if __name__ == "__main__":
    main()
