#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot real-data precision-recall curves from saved grouped ECDF artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "ecdf_results"
OUTPUT_DIR = ROOT / "paper_figures"

DATASET_ORDER = ["smd", "swat"]
DATASET_TITLES = {"smd": "SMD", "swat": "SWaT"}
VARIANT_ORDER = ["none", "mlp", "mlp_mean", "attn_mean"]
VARIANT_COLORS = {
    "none": "#264653",
    "mlp": "#2a9d8f",
    "mlp_mean": "#e9c46a",
    "attn_mean": "#c45a2d",
}
SEED_LINESTYLES = {
    0: "solid",
    1: "dashed",
    2: "dashdot",
    3: (0, (1, 1)),
    4: (0, (3, 1, 1, 1)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smd-summary", type=Path, default=None)
    parser.add_argument("--swat-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def _find_latest_summary(dataset: str) -> Path:
    matches = sorted(
        ARTIFACT_ROOT.glob(f"grouped_ecdf_{dataset}_*/grouped_ecdf_{dataset}_results.json")
    )
    if not matches:
        raise FileNotFoundError(f"No grouped ECDF summary found for {dataset} under {ARTIFACT_ROOT}")
    return matches[-1]


def _resolve_curve_path(summary_path: Path, curve_artifact: str) -> Path:
    curve_path = Path(curve_artifact)
    if curve_path.is_absolute():
        return curve_path
    summary_relative = (summary_path.parent / curve_path)
    if summary_relative.exists():
        return summary_relative.resolve()
    return (ROOT / curve_path).resolve()


def _load_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _variant_runs(summary: dict, variant: str) -> list[dict]:
    return sorted(
        (row for row in summary["per_seed"] if row["variant"] == variant),
        key=lambda row: row["seed"],
    )


def _interpolate_precision_curve(curve_path: Path, recall_grid: np.ndarray) -> np.ndarray:
    """Project a saved PR curve onto a shared recall grid for seed aggregation."""
    curve = np.load(curve_path)
    recall = np.asarray(curve["recall"], dtype=np.float64)
    precision = np.asarray(curve["precision"], dtype=np.float64)

    # sklearn returns recall in descending order; reverse it for interpolation.
    recall = recall[::-1]
    precision = precision[::-1]

    unique_recall, unique_indices = np.unique(recall, return_index=True)
    unique_precision = precision[unique_indices]
    return np.interp(
        recall_grid,
        unique_recall,
        unique_precision,
        left=unique_precision[0],
        right=unique_precision[-1],
    )


def _summarize_variant_curves(
    summary: dict,
    summary_path: Path,
    variant: str,
    recall_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    runs = _variant_runs(summary, variant)
    if not runs:
        return None

    interpolated = []
    aucprs = []
    for row in runs:
        curve_path = _resolve_curve_path(summary_path, row["curve_artifact"])
        interpolated.append(_interpolate_precision_curve(curve_path, recall_grid))
        aucprs.append(float(row["aucpr"]))

    precision_stack = np.vstack(interpolated)
    return (
        precision_stack.mean(axis=0),
        precision_stack.std(axis=0),
        float(np.mean(aucprs)),
        float(np.std(aucprs)),
    )


def _plot_variant_axis(
    ax: plt.Axes,
    dataset: str,
    variant: str,
    summary: dict,
    summary_path: Path,
    recall_grid: np.ndarray,
) -> None:
    variant_summary = _summarize_variant_curves(summary, summary_path, variant, recall_grid)
    runs = _variant_runs(summary, variant)
    if variant_summary is None or not runs:
        ax.set_visible(False)
        return

    _precision_mean, _precision_std, aucpr_mean, aucpr_std = variant_summary
    color = VARIANT_COLORS[variant]

    for row in runs:
        curve_path = _resolve_curve_path(summary_path, row["curve_artifact"])
        curve = np.load(curve_path)
        ax.plot(
            curve["recall"],
            curve["precision"],
            color=color,
            linewidth=1.4,
            linestyle=SEED_LINESTYLES.get(int(row["seed"]), "solid"),
            alpha=0.85,
        )

    ax.set_xlabel("Recall")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.set_title(variant)
    ax.text(
        0.03,
        0.04,
        f"AUCPR {aucpr_mean:.3f} +/- {aucpr_std:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9, "pad": 2},
    )


def plot_pr_curves(smd_summary_path: Path, swat_summary_path: Path, output_dir: Path) -> None:
    summaries = {
        "smd": _load_summary(smd_summary_path),
        "swat": _load_summary(swat_summary_path),
    }
    recall_grid = np.linspace(0.0, 1.0, 501)

    fig, axes = plt.subplots(
        len(DATASET_ORDER),
        len(VARIANT_ORDER),
        figsize=(17.4, 7.6),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.11, top=0.92, wspace=0.18, hspace=0.24)
    summary_paths = {
        "smd": smd_summary_path,
        "swat": swat_summary_path,
    }
    for row, dataset in enumerate(DATASET_ORDER):
        for col, variant in enumerate(VARIANT_ORDER):
            ax = axes[row, col]
            _plot_variant_axis(
                ax,
                dataset,
                variant,
                summaries[dataset],
                summary_paths[dataset],
                recall_grid,
            )
            if col == 0:
                ax.set_ylabel(f"{DATASET_TITLES[dataset]}\nPrecision")

    seed_handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.6, linestyle=SEED_LINESTYLES[seed], label=f"seed {seed}")
        for seed in sorted(SEED_LINESTYLES)
    ]
    fig.legend(
        handles=seed_handles,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.855, 0.5),
        ncol=1,
        title="Seed",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"realdata_pr_curves.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    smd_summary = args.smd_summary or _find_latest_summary("smd")
    swat_summary = args.swat_summary or _find_latest_summary("swat")
    plot_pr_curves(smd_summary.resolve(), swat_summary.resolve(), args.output_dir)
    print(f"Saved precision-recall figures to {args.output_dir}")


if __name__ == "__main__":
    main()
