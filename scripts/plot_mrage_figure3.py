#!/usr/bin/env python3
"""Build Figure 3 panels from the real mechanism-analysis CSVs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "experiments/mrage_figure3_mechanism/data"
DEFAULT_FIGURES = PROJECT_ROOT / "experiments/mrage_figure3_mechanism/figures"

DATASETS = ("Grocery", "Toys", "Reddit-S", "ele-fashion")
CONTEXTS = (
    "private_text_physical",
    "private_text_semantic",
    "private_visual_physical",
    "private_visual_semantic",
    "collaborative_text_physical",
    "collaborative_visual_physical",
)
CONTEXT_LABELS = {
    "private_text_physical": "Private Text–Physical",
    "private_text_semantic": "Private Text–Semantic",
    "private_visual_physical": "Private Visual–Physical",
    "private_visual_semantic": "Private Visual–Semantic",
    "collaborative_text_physical": "Collaborative Text–Physical",
    "collaborative_visual_physical": "Collaborative Visual–Physical",
}
GROUPS = (
    ("private_physical", "Private\nPhysical"),
    ("private_semantic", "Private\nSemantic"),
    ("collaborative_physical", "Collaborative\nPhysical"),
    ("all_contexts", "All\nContexts"),
)
DATASET_COLORS = {
    "Grocery": "#2a78d6",
    "Toys": "#eb6834",
    "Reddit-S": "#1baf7a",
    "ele-fashion": "#eda100",
}
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list(
    "mrage_diverging", ["#2a78d6", "#f0efec", "#e34948"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "mrage_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def read_csv(name: str) -> list[dict]:
    with (args.data / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.6,
            "axes.edgecolor": MUTED,
            "xtick.color": SECONDARY,
            "ytick.color": SECONDARY,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )


def _annotate_heatmap(ax, values: np.ndarray, *, norm, cmap, fmt="{:.2f}", fontsize=6.5):
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "white" if luminance < 0.55 else INK
            ax.text(
                column,
                row,
                fmt.format(value),
                ha="center",
                va="center",
                color=color,
                fontsize=fontsize,
            )


def _coefficient_matrices() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    rows = [
        row
        for row in read_csv("expert_coefficients.csv")
        if row["dataset"] == "Grocery"
    ]
    result: dict[str, np.ndarray] = {}
    for relation, width in (("physical", 4), ("semantic", 2)):
        values = np.zeros((4, width), dtype=float)
        for expert in range(4):
            for order in range(width):
                selected = [
                    float(row["trained_coefficient"])
                    for row in rows
                    if row["relation"] == relation
                    and int(row["expert_id"]) == expert + 1
                    and int(row["order"]) == order
                ]
                values[expert, order] = statistics.fmean(selected)
        result[relation] = values
    diversity = [
        row
        for row in read_csv("expert_diversity.csv")
        if row["dataset"] == "Grocery"
        and row["metric"] == "l2_change_from_initialization"
    ]
    l2 = {
        relation: statistics.fmean(
            float(row["value"])
            for row in diversity
            if row["relation"] == relation
        )
        for relation in ("physical", "semantic")
    }
    return result["physical"], result["semantic"], l2


def _routing_matrix() -> np.ndarray:
    rows = [
        row
        for row in read_csv("routing_patterns.csv")
        if row["dataset"] == "Grocery"
        and row["subset"]
        == ("semantic_support" if "semantic" in row["context"] else "physical_support")
    ]
    values = np.zeros((len(CONTEXTS), 4), dtype=float)
    for context_index, context in enumerate(CONTEXTS):
        for expert in range(4):
            selected = [
                float(row["mean_top2_weight"])
                for row in rows
                if row["context"] == context and int(row["expert_id"]) == expert + 1
            ]
            values[context_index, expert] = statistics.fmean(selected)
    return values


def _routing_by_dataset() -> dict[str, np.ndarray]:
    rows = read_csv("routing_patterns.csv")
    result = {}
    for dataset in DATASETS:
        values = np.zeros((len(CONTEXTS), 4), dtype=float)
        for context_index, context in enumerate(CONTEXTS):
            subset = "semantic_support" if "semantic" in context else "physical_support"
            for expert in range(4):
                selected = [
                    float(row["mean_top2_weight"])
                    for row in rows
                    if row["dataset"] == dataset
                    and row["context"] == context
                    and row["subset"] == subset
                    and int(row["expert_id"]) == expert + 1
                ]
                values[context_index, expert] = statistics.fmean(selected)
        result[dataset] = values
    return result


def _shuffle_summary() -> dict[tuple[str, str], dict[str, float]]:
    rows = read_csv("router_shuffle_dataset_summary.csv")
    result = {}
    for row in rows:
        result[(row["dataset"], row["intervention_group"])] = {
            "delta_ce": float(row["mean_delta_val_ce"]),
            "delta_ce_std": float(row["std_delta_val_ce"]),
            "delta_acc": float(row["mean_delta_val_acc"]),
            "delta_acc_std": float(row["std_delta_val_acc"]),
            "disagreement": float(row["mean_prediction_disagreement"]),
        }
    return result


def _format_heatmap(
    ax,
    values: np.ndarray,
    *,
    title: str,
    row_labels: list[str],
    column_labels: list[str],
    norm,
    cmap,
    fmt: str = "{:.2f}",
    title_size: float = 8.3,
):
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_xticklabels(column_labels, fontsize=6.5)
    ax.set_yticklabels(row_labels, fontsize=6.5)
    ax.set_title(title, loc="left", pad=5, fontweight="bold", fontsize=title_size)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _annotate_heatmap(ax, values, norm=norm, cmap=cmap, fmt=fmt, fontsize=6.2)
    return image


def plot_panel_a(ax_physical, ax_semantic, cax, *, show_l2_note: bool = True):
    physical, semantic, l2 = _coefficient_matrices()
    norm = TwoSlopeNorm(vmin=-0.75, vcenter=0.0, vmax=0.75)
    image = _format_heatmap(
        ax_physical,
        physical,
        title="(a) Physical filter coefficients",
        row_labels=[f"E{i}" for i in range(1, 5)],
        column_labels=["0", "1", "2", "3"],
        norm=norm,
        cmap=DIVERGING,
    )
    _format_heatmap(
        ax_semantic,
        semantic,
        title="Semantic filter coefficients",
        row_labels=[f"E{i}" for i in range(1, 5)],
        column_labels=["0", "1"],
        norm=norm,
        cmap=DIVERGING,
    )
    ax_physical.set_ylabel("Shared expert", fontsize=7.5)
    ax_physical.set_xlabel("Polynomial order", fontsize=7.0, labelpad=2)
    ax_semantic.set_xlabel("Polynomial order", fontsize=7.0, labelpad=2)
    colorbar = plt.colorbar(image, cax=cax, orientation="horizontal")
    colorbar.set_label(
        "Effective coefficient (blue negative, red positive)", fontsize=6.4
    )
    colorbar.ax.tick_params(labelsize=6)
    colorbar.outline.set_linewidth(0.4)
    if show_l2_note:
        cax.text(
            0.5,
            -3.2,
            f"Mean L2 change from initialization: physical {l2['physical']:.3f}, "
            f"semantic {l2['semantic']:.3f}",
            transform=cax.transAxes,
            ha="center",
            va="top",
            fontsize=6.1,
            color=SECONDARY,
        )
    return image


def plot_panel_b(ax, cax):
    values = _routing_matrix()
    norm = Normalize(vmin=0.0, vmax=0.85)
    image = _format_heatmap(
        ax,
        values,
        title="(b) Mean Top-2 routing weight",
        row_labels=[CONTEXT_LABELS[context] for context in CONTEXTS],
        column_labels=[f"E{i}" for i in range(1, 5)],
        norm=norm,
        cmap=SEQUENTIAL,
        fmt="{:.2f}",
        title_size=8.3,
    )
    ax.set_xlabel("Expert", fontsize=7.0, labelpad=2)
    colorbar = plt.colorbar(image, cax=cax, orientation="vertical")
    colorbar.set_label("Mean Top-2 weight", fontsize=6.4)
    colorbar.ax.tick_params(labelsize=6)
    colorbar.outline.set_linewidth(0.4)
    return image


def _grouped_bars(ax, summary, metric: str, error: str, ylabel: str, show_xlabels: bool):
    x = np.arange(len(GROUPS))
    width = 0.18
    for dataset_index, dataset in enumerate(DATASETS):
        means, errors = [], []
        for group_name, _ in GROUPS:
            entry = summary.get((dataset, group_name), {})
            means.append(entry.get(metric, 0.0))
            errors.append(entry.get(error, 0.0))
        offsets = x + (dataset_index - (len(DATASETS) - 1) / 2) * width
        bars = ax.bar(
            offsets,
            means,
            width=width,
            color=DATASET_COLORS[dataset],
            edgecolor=SURFACE,
            linewidth=0.5,
            yerr=errors,
            error_kw={
                "elinewidth": 0.55,
                "capsize": 1.2,
                "capthick": 0.55,
                "ecolor": MUTED,
            },
            label=dataset,
            zorder=3,
        )
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            if abs(mean) < 8.0e-4:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (0.0012 if height >= 0 else -0.0012),
                f"{mean:.3f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=4.7,
                color=SECONDARY,
            )
    ax.axhline(0.0, color=MUTED, linewidth=0.6, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in GROUPS], fontsize=6.4)
    if not show_xlabels:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel(ylabel, fontsize=7.0)
    ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=0)
    ax.tick_params(axis="y", labelsize=6.1)
    ax.set_axisbelow(True)
    ax.margins(x=0.06)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_panel_c(ax_ce, ax_acc):
    summary = _shuffle_summary()
    _grouped_bars(ax_ce, summary, "delta_ce", "delta_ce_std", "Δ Validation CE", False)
    _grouped_bars(ax_acc, summary, "delta_acc", "delta_acc_std", "Δ Validation Acc", True)
    ax_acc.set_xlabel("Intervention context", fontsize=7.0, labelpad=3)
    ax_ce.set_title("(c) Frozen router-shuffle sensitivity", loc="left", pad=5, fontweight="bold")


def _dataset_legend(fig, anchor=(0.5, 1.0)):
    handles = [
        Patch(facecolor=DATASET_COLORS[dataset], label=dataset) for dataset in DATASETS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=anchor,
        ncol=4,
        frameon=False,
        fontsize=6.2,
        handlelength=1.0,
        columnspacing=1.2,
    )


def _save(fig, stem: str):
    fig.savefig(args.figures / f"{stem}.png", dpi=300)
    fig.savefig(args.figures / f"{stem}.pdf")
    plt.close(fig)


def render_individual_panels():
    fig = plt.figure(figsize=(5.6, 2.7))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.08], width_ratios=[1.0, 0.62], wspace=0.65, hspace=0.32)
    ax_physical = fig.add_subplot(grid[0, 0])
    ax_semantic = fig.add_subplot(grid[0, 1])
    cax = fig.add_subplot(grid[1, :])
    plot_panel_a(ax_physical, ax_semantic, cax)
    _save(fig, "figure3a")

    fig = plt.figure(figsize=(4.8, 3.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.04)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    plot_panel_b(ax, cax)
    _save(fig, "figure3b")

    fig = plt.figure(figsize=(5.6, 3.8))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.55)
    ax_ce = fig.add_subplot(grid[0])
    ax_acc = fig.add_subplot(grid[1])
    plot_panel_c(ax_ce, ax_acc)
    _dataset_legend(fig, anchor=(0.5, 1.0))
    fig.text(
        0.01,
        0.005,
        "Frozen Validation Diagnostic — No Retraining",
        fontsize=6.2,
        color=SECONDARY,
        ha="left",
    )
    _save(fig, "figure3c")


def render_combined():
    fig = plt.figure(figsize=(7.2, 6.3))
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.95, 1.10, 0.90],
        hspace=0.44,
        left=0.115,
        right=0.965,
        top=0.94,
        bottom=0.075,
    )
    top = outer[0].subgridspec(2, 2, height_ratios=[1.0, 0.09], width_ratios=[1.0, 0.62], wspace=0.62, hspace=0.25)
    ax_a1 = fig.add_subplot(top[0, 0])
    ax_a2 = fig.add_subplot(top[0, 1])
    cax_a = fig.add_subplot(top[1, :])
    mid = outer[1].subgridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.03)
    ax_b = fig.add_subplot(mid[0, 0])
    cax_b = fig.add_subplot(mid[0, 1])
    bottom = outer[2].subgridspec(1, 2, wspace=0.55)
    ax_ce = fig.add_subplot(bottom[0, 0])
    ax_acc = fig.add_subplot(bottom[0, 1])
    plot_panel_a(ax_a1, ax_a2, cax_a, show_l2_note=False)
    plot_panel_b(ax_b, cax_b)
    plot_panel_c(ax_ce, ax_acc)
    _dataset_legend(fig, anchor=(0.5, 1.0))
    fig.text(
        0.01,
        0.005,
        "Frozen Validation Diagnostic — No Retraining",
        fontsize=6.2,
        color=SECONDARY,
        ha="left",
    )
    _save(fig, "figure3")


def render_routing_by_dataset():
    matrices = _routing_by_dataset()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    norm = Normalize(vmin=0.0, vmax=0.85)
    for ax, dataset in zip(axes.flat, DATASETS):
        _format_heatmap(
            ax,
            matrices[dataset],
            title=dataset,
            row_labels=[CONTEXT_LABELS[context] for context in CONTEXTS],
            column_labels=[f"E{i}" for i in range(1, 5)],
            norm=norm,
            cmap=SEQUENTIAL,
            fmt="{:.2f}",
        )
    fig.text(
        0.5,
        0.005,
        "Mean Top-2 routing weight, seeds 42–44; fixed color scale 0–0.85",
        ha="center",
        fontsize=6.5,
        color=SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "figure3b_by_dataset")


def main() -> int:
    global args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    args.data = args.data.expanduser().resolve()
    args.figures = args.figures.expanduser().resolve()
    args.figures.mkdir(parents=True, exist_ok=True)
    style()
    render_individual_panels()
    render_combined()
    render_routing_by_dataset()
    print(f"Wrote Figure 3 panels to {args.figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
