#!/usr/bin/env python3
"""Collect Hydra results.json files into one CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = (args.output or root / "summary.csv").expanduser().resolve()

    records = []
    metric_names = set()
    for path in sorted(root.rglob("results.json")):
        if path.resolve() == output:
            continue
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        row = {"output_dir": str(path.parent)}
        for metric, values in payload.items():
            row[f"{metric}_mean"] = values.get("mean")
            row[f"{metric}_std"] = values.get("std")
            metric_names.add(f"{metric}_mean")
            metric_names.add(f"{metric}_std")
        records.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["output_dir", *sorted(metric_names)]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    print(f"Collected {len(records)} runs into {output}")


if __name__ == "__main__":
    main()
