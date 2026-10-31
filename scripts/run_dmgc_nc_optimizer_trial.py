#!/usr/bin/env python3
"""Run DMGC's NC optimizer trial with the shared task-level LR and decay."""

from __future__ import annotations

from pathlib import Path

from run_baseline_matrix import _launch_matrix, _parse_common_args
from run_nc_baseline_benchmark import DATASETS, PROTOCOL_OVERRIDES, PROTOCOL_DESCRIPTION


TRIAL_OVERRIDES = PROTOCOL_OVERRIDES + (
    # build_optimizer honors model-level presets, so set these explicitly to
    # test the task-level 1e-3 / 1e-4 values against DMGC's official preset.
    "model.lr=0.001",
    "model.weight_decay=0.0001",
    # Keep the original DMGC objective fixed while testing the optimizer.
    "model.lambda_cm=1.0",
)


def main() -> int:
    args = _parse_common_args(__doc__ or "Run the DMGC NC optimizer trial.")
    if args.output_root == Path("outputs/baseline_benchmark"):
        args.output_root = Path("outputs/dmgc_nc_task_lr001_wd1e4_20261031")

    description = dict(PROTOCOL_DESCRIPTION)
    description["optimizer"] = (
        "AdamW with DMGC LR=1e-3 and weight decay=1e-4 explicitly overridden from the "
        "OpenMAG preset (5e-3 / 1e-5); task-level settings are 1e-3 / 1e-4."
    )
    description["selection"] += " DMGC validation checkpoint selection remains by Accuracy."

    return _launch_matrix(
        task="nc",
        datasets=DATASETS,
        models=("dmgc",),
        protocol_version="unified_full_graph_nc_method_objectives_v3_dmgc_task_optimizer_trial",
        protocol_overrides=TRIAL_OVERRIDES,
        protocol_description=description,
        args=args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
