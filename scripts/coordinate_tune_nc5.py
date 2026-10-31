#!/usr/bin/env python3
"""Sequential single-parameter MRAGE screening on three representative NC datasets."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.run_artifacts import write_json_atomic


def full_candidate(study: Path, manifest, name: str):
    base = OmegaConf.create({
        "model": OmegaConf.to_container(OmegaConf.load(study / "base_model.yaml")),
        "task": OmegaConf.to_container(OmegaConf.load(study / "base_task_nc.yaml")),
    })
    return OmegaConf.merge(base, manifest.get("common_overrides", {}), manifest.candidates[name].overrides)


def signature(cfg) -> str:
    return json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True)


def scores(study: Path, names: list[str], datasets: list[str]) -> list[dict]:
    path = study / "trials.json"
    runs = json.loads(path.read_text())["runs"] if path.exists() else []
    rows = []
    for name in names:
        found = {r["dataset"]: r for r in runs if r["candidate"] == name and r["seed"] == 42
                 and r["phase"] == "validation" and r["status"] == "validation_complete"
                 and r["dataset"] in datasets}
        row = {"candidate": name, "num_datasets": len(found),
               "datasets": {d: {k: found[d][k] for k in ("val_acc", "val_macro_f1", "best_epoch")}
                            for d in datasets if d in found}}
        if len(found) == len(datasets):
            row.update(mean_val_acc=statistics.fmean(r["val_acc"] for r in found.values()),
                       mean_val_macro_f1=statistics.fmean(r["val_macro_f1"] for r in found.values()))
        rows.append(row)
    return rows


def save_summary(study: Path, manifest) -> None:
    rows = scores(study, list(manifest.candidates), list(manifest.representatives))
    rows.sort(key=lambda r: (r.get("mean_val_acc", -1), r.get("mean_val_macro_f1", -1)), reverse=True)
    columns = ["candidate", "num_datasets", "mean_val_acc", "mean_val_macro_f1", *manifest.representatives]
    with (study / "screening_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row.get(k, "") for k in columns if k not in manifest.representatives},
                             **{d: row["datasets"].get(d, {}).get("val_acc", "") for d in manifest.representatives}})
    write_json_atomic(study / "screening_summary.json", {"candidates": rows})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=ROOT / "experiments/nc5_tuning_20261030")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--include-capacity", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    study = args.study_dir.resolve()
    manifest = OmegaConf.load(study / "study.yaml")
    decisions_path = study / "axis_decisions.json"
    decisions = json.loads(decisions_path.read_text())["decisions"] if decisions_path.exists() else []
    anchor = decisions[-1]["winner"] if decisions else "baseline"
    completed_axes = {decision["parameter"] for decision in decisions}
    axes = [(stage, parameter, list(values)) for stage, parameters in manifest.parameter_schedule.items()
            if stage != "representation_last" or args.include_capacity
            for parameter, values in parameters.items()]
    for index, (stage, parameter, values) in enumerate(axes):
        if parameter in completed_axes:
            continue
        if (study / "STOP_AFTER_CURRENT").exists():
            print("STOP_AFTER_CURRENT exists; no further axis launched.", flush=True)
            return 0
        print(f"AXIS {index + 1}/{len(axes)} {parameter} anchor={anchor}", flush=True)
        current = full_candidate(study, manifest, anchor)
        signatures = {signature(full_candidate(study, manifest, name)): name for name in manifest.candidates}
        names = [anchor]
        for value in values:
            candidate_cfg = OmegaConf.create(OmegaConf.to_container(current))
            OmegaConf.update(candidate_cfg, parameter, value)
            existing = signatures.get(signature(candidate_cfg))
            if existing:
                if existing not in names:
                    names.append(existing)
                continue
            label = str(value).replace(".", "p").replace("-", "m")
            name = f"a{index:02d}_{parameter.split('.')[-1]}_{label}"
            overrides = OmegaConf.create(OmegaConf.to_container(manifest.candidates[anchor].overrides))
            OmegaConf.update(overrides, parameter, value, force_add=True)
            manifest.candidates[name] = {
                "description": f"Change only {parameter} to {value} from {anchor}.",
                "stage": stage, "anchor": anchor, "changed_parameter": parameter,
                "changed_value": value, "overrides": OmegaConf.to_container(overrides),
            }
            signatures[signature(candidate_cfg)] = name
            names.append(name)
        if args.dry_run:
            print({"parameter": parameter, "candidates": names, "datasets": list(manifest.representatives)})
            continue
        OmegaConf.save(manifest, study / "study.yaml")
        axis_dir = study / "axis_logs"
        axis_dir.mkdir(exist_ok=True)
        command = [sys.executable, "scripts/tune_nc5.py", "--study-dir", str(study),
                   "--devices", *args.devices, "--candidates", *names,
                   "--datasets", "ele-fashion", "Movies", "Grocery", "--seeds", "42"]
        with (axis_dir / f"{index:02d}_{parameter}.log").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"command": command, "at_unix": time.time()}) + "\n")
            log.flush()
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if (study / "STOP_AFTER_CURRENT").exists():
            save_summary(study, manifest)
            return 0
        measured = scores(study, names, list(manifest.representatives))
        complete = [row for row in measured if "mean_val_acc" in row]
        if not complete:
            raise RuntimeError(f"no complete candidate at axis {parameter}; see {axis_dir}")
        winner = max(complete, key=lambda row: (row["mean_val_acc"], row["mean_val_macro_f1"]))
        decision = {
            "stage": stage, "parameter": parameter, "values": values,
            "anchor_before": anchor, "winner": winner["candidate"],
            "selection": "highest unweighted mean Validation Accuracy across representatives; Macro-F1 breaks ties",
            "scores": measured, "runner_exit_code": result.returncode, "at_unix": time.time(),
        }
        decisions.append(decision)
        write_json_atomic(decisions_path, {"decisions": decisions})
        anchor = winner["candidate"]
        save_summary(study, manifest)
        print(f"SELECT {parameter}: {anchor} mean_val_acc={winner['mean_val_acc']:.6f}", flush=True)
    if args.dry_run:
        return 0
    write_json_atomic(study / "screening_complete.json", {
        "anchor": anchor, "num_axes": len(decisions), "capacity_included": args.include_capacity,
        "test_used_for_selection": False, "at_unix": time.time(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
