#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate manuscript-ready figures from saved experiment artifacts.

This script builds five figures in the requested order:
    1. Real-data aggregate grouped bars (F1 and AUCPR on SMD/SWaT)
    2. Real-data per-seed AUCPR curves from the manuscript tables
    3. Real-data per-seed heatmaps (seed × variant for F1/AUCPR)
    4. Synthetic anomaly-type delta bars (Fusion - Baseline AUCPR)
    5. Feature-similarity heatmap with cluster boundaries from Stage 1

The first three figures are sourced directly from the manuscript LaTeX tables so
the values match the current paper text. The fourth figure reuses the feature
selection pipeline on real training data to visualize the lagged Spearman
similarity matrix and redundancy clusters.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import (
    DEFAULT_PARAMS_SMD,
    DEFAULT_PARAMS_SWAT,
    MACHINE,
    SEQUENCE_LENGTH,
    SMD_DRIVE,
    SWAT_ATTACK_DATASET,
    SWAT_NORMAL_DATASET,
    SWAT_VAL_RATIO,
)
from data_loader import (
    detect_binary_features,
    load_smd_data,
    load_swat_data,
    preprocess_data,
    standardize_continuous_features,
)
from feature_selection import compute_feature_selection_similarity_diagnostics


ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
OUTPUT_DIR = ROOT / "paper_figures"

VARIANT_ORDER = ["none", "mlp", "mlp_mean", "attn_mean"]
VARIANT_LABELS = {
    "none": "none",
    "mlp": "mlp",
    "mlp_mean": "mlp_mean",
    "attn_mean": "attn_mean",
}
VARIANT_TICK_LABELS = {
    "none": "none",
    "mlp": "mlp",
    "mlp_mean": "mlp\nmean",
    "attn_mean": "attn\nmean",
}
DATASET_ORDER = ["SMD", "SWaT"]
DATASET_COLORS = {"SMD": "#2a6f97", "SWaT": "#c45a2d"}
VARIANT_COLORS = {
    "none": "#264653",
    "mlp": "#2a9d8f",
    "mlp_mean": "#e9c46a",
    "attn_mean": "#c45a2d",
}
METRIC_LABELS = {"f1": "F1", "aucpr": "AUCPR"}
POSITIVE_COLOR = "#2a9d8f"
NEGATIVE_COLOR = "#c44536"
HEATMAP_CMAP = "YlOrRd"
SIMILARITY_CMAP = "magma"

TABLE_LABELS = {
    "SMD": {"aggregate": "tab:smd-aggregate", "per_seed": "tab:smd-perseed"},
    "SWaT": {"aggregate": "tab:swat-aggregate", "per_seed": "tab:swat-perseed"},
}

FLOAT_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


def _strip_latex(text: str) -> str:
    """Remove lightweight LaTeX wrappers while keeping numeric content."""
    cleaned = text.strip()
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", cleaned)
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("\\", "")
    cleaned = cleaned.replace("{", "")
    cleaned = cleaned.replace("}", "")
    cleaned = cleaned.replace("\u00a0", " ")
    return " ".join(cleaned.split())


def _extract_numbers(text: str) -> list[float]:
    return [float(token) for token in FLOAT_RE.findall(_strip_latex(text))]


def _parse_mean_std(cell: str) -> tuple[float, float]:
    numbers = _extract_numbers(cell)
    if len(numbers) < 2:
        raise ValueError(f"Expected mean±std cell, got: {cell}")
    return numbers[0], numbers[1]


def _parse_single_value(cell: str) -> float:
    numbers = _extract_numbers(cell)
    if not numbers:
        raise ValueError(f"Expected numeric cell, got: {cell}")
    return numbers[0]


def _variant_key(raw_variant: str) -> str:
    cleaned = _strip_latex(raw_variant)
    cleaned = cleaned.replace(" (Baseline)", "")
    cleaned = cleaned.replace(" ", "")
    return cleaned


def _extract_table_block(text: str, label: str) -> str:
    marker = f"\\label{{{label}}}"
    start = text.index(marker)
    end = text.index("\\end{table}", start)
    return text[start:end]


def load_realdata_results(tex_path: Path) -> tuple[dict, dict]:
    """Parse aggregate and per-seed grouped results from realdata.tex."""
    text = tex_path.read_text(encoding="utf-8")
    aggregate: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    per_seed: dict[str, dict[str, np.ndarray]] = {}

    aggregate_row_re = re.compile(
        r"^(none \(Baseline\)|mlp|mlp\\_mean|attn\\_mean)\s*&\s*(.+?)\s*&\s*(.+?)\s*&\s*(.+?)\\\\$",
        re.MULTILINE,
    )
    seed_row_re = re.compile(
        r"^(\d+)\s*&\s*(.+?)\s*&\s*(.+?)\s*&\s*(.+?)\s*&\s*(.+?)\\\\$",
        re.MULTILINE,
    )

    for dataset in DATASET_ORDER:
        aggregate_block = _extract_table_block(text, TABLE_LABELS[dataset]["aggregate"])
        aggregate[dataset] = {}
        for variant_raw, f1_cell, aucpr_cell, score_sep_cell in aggregate_row_re.findall(aggregate_block):
            variant = _variant_key(variant_raw)
            aggregate[dataset][variant] = {
                "f1": _parse_mean_std(f1_cell),
                "aucpr": _parse_mean_std(aucpr_cell),
                "score_sep": _parse_mean_std(score_sep_cell),
            }

        per_seed_block = _extract_table_block(text, TABLE_LABELS[dataset]["per_seed"])
        f1_section = per_seed_block.split("\\textit{F1}", 1)[1].split("\\textit{AUCPR}", 1)[0]
        aucpr_section = per_seed_block.split("\\textit{AUCPR}", 1)[1]

        dataset_seed_results = {
            "f1": np.zeros((5, len(VARIANT_ORDER)), dtype=float),
            "aucpr": np.zeros((5, len(VARIANT_ORDER)), dtype=float),
        }
        for metric_name, section in [("f1", f1_section), ("aucpr", aucpr_section)]:
            for seed_str, *cells in seed_row_re.findall(section):
                seed = int(seed_str)
                for col, cell in enumerate(cells):
                    dataset_seed_results[metric_name][seed, col] = _parse_single_value(cell)
        per_seed[dataset] = dataset_seed_results

    return aggregate, per_seed


def load_synthetic_deltas(tex_path: Path) -> list[dict[str, object]]:
    """Parse anomaly-type AUCPR deltas from the synthetic experiment table."""
    text = tex_path.read_text(encoding="utf-8")
    block = _extract_table_block(text, "tab:synth-pertype")

    rows: list[dict[str, object]] = []
    current_category = ""
    row_re = re.compile(r"^\\quad\s*(.+?)\s*&\s*(.+?)\s*&\s*(.+?)\s*&\s*(.+?)\\\\$")

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if "Spatial anomalies" in line:
            current_category = "Spatial"
            continue
        if "Temporal anomalies" in line:
            current_category = "Temporal"
            continue
        if "Control anomalies" in line:
            current_category = "Control"
            continue

        match = row_re.match(line)
        if not match:
            continue
        anomaly_name, baseline_cell, fusion_cell, _winner_cell = match.groups()
        if "---" in baseline_cell or "---" in fusion_cell:
            continue
        baseline = _parse_single_value(baseline_cell)
        fusion = _parse_single_value(fusion_cell)
        rows.append(
            {
                "category": current_category,
                "anomaly": _strip_latex(anomaly_name),
                "baseline": baseline,
                "fusion": fusion,
                "delta": fusion - baseline,
            }
        )

    return rows


def _save_figure(fig: plt.Figure, stem: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_realdata_aggregate_bars(aggregate: dict, output_dir: Path) -> None:
    """Render grouped bar charts for real-data aggregate F1 and AUCPR."""
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8), constrained_layout=True)
    x = np.arange(len(VARIANT_ORDER))
    width = 0.36

    for ax, metric in zip(axes, ["f1", "aucpr"]):
        all_means = [aggregate[dataset][variant][metric][0] for dataset in DATASET_ORDER for variant in VARIANT_ORDER]
        all_stds = [aggregate[dataset][variant][metric][1] for dataset in DATASET_ORDER for variant in VARIANT_ORDER]
        y_min = max(0.0, min(m - s for m, s in zip(all_means, all_stds)) - 0.03)
        y_max = min(1.0, max(m + s for m, s in zip(all_means, all_stds)) + 0.06)
        label_pad = 0.018 * (y_max - y_min)

        for idx, dataset in enumerate(DATASET_ORDER):
            offset = (idx - 0.5) * width
            means = [aggregate[dataset][variant][metric][0] for variant in VARIANT_ORDER]
            stds = [aggregate[dataset][variant][metric][1] for variant in VARIANT_ORDER]
            bars = ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=4,
                label=dataset,
                color=DATASET_COLORS[dataset],
                alpha=0.92,
                edgecolor="black",
                linewidth=0.6,
            )
            for bar, mean, std in zip(bars, means, stds):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean + std + label_pad,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    clip_on=False,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_TICK_LABELS[v] for v in VARIANT_ORDER])
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"Real-data {METRIC_LABELS[metric]}")
        ax.set_ylim(y_min, y_max)
        ax.margins(x=0.08)
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Grouped fusion variants on SMD and SWaT", fontsize=14, y=1.05)
    _save_figure(fig, "realdata_aggregate_bars", output_dir)


def plot_realdata_seed_heatmaps(per_seed: dict, output_dir: Path) -> None:
    """Render seed-by-variant heatmaps for F1 and AUCPR on SMD and SWaT."""
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), constrained_layout=True)
    metric_limits = {
        metric: (
            min(per_seed[dataset][metric].min() for dataset in DATASET_ORDER),
            max(per_seed[dataset][metric].max() for dataset in DATASET_ORDER),
        )
        for metric in ("f1", "aucpr")
    }

    for row, dataset in enumerate(DATASET_ORDER):
        for col, metric in enumerate(("f1", "aucpr")):
            ax = axes[row, col]
            values = per_seed[dataset][metric]
            vmin, vmax = metric_limits[metric]
            image = ax.imshow(values, cmap=HEATMAP_CMAP, vmin=vmin, vmax=vmax, aspect="auto")

            row_max = values.argmax(axis=1)
            for seed in range(values.shape[0]):
                for variant_idx in range(values.shape[1]):
                    value = values[seed, variant_idx]
                    text_color = "white" if value < (vmin + vmax) / 2 else "black"
                    ax.text(
                        variant_idx,
                        seed,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color,
                        fontweight="bold" if variant_idx == row_max[seed] else "normal",
                    )

            ax.set_xticks(np.arange(len(VARIANT_ORDER)))
            ax.set_xticklabels([VARIANT_TICK_LABELS[v] for v in VARIANT_ORDER], fontsize=9)
            ax.set_yticks(np.arange(5))
            ax.set_yticklabels([str(seed) for seed in range(5)])
            ax.set_ylabel("Seed")
            ax.set_title(f"{dataset} {METRIC_LABELS[metric]}")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle("Per-seed grouped-model performance", fontsize=14)
    _save_figure(fig, "realdata_per_seed_heatmaps", output_dir)


def _export_realdata_aucpr_points(per_seed: dict, output_dir: Path) -> None:
    """Save parsed per-seed AUCPR values for downstream inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASET_ORDER:
        aucpr_values = per_seed[dataset]["aucpr"]
        for seed in range(aucpr_values.shape[0]):
            for variant_idx, variant in enumerate(VARIANT_ORDER):
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "variant": variant,
                        "aucpr": float(aucpr_values[seed, variant_idx]),
                    }
                )

    export_path = output_dir / "realdata_aucpr_points.json"
    export_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def plot_realdata_aucpr_curves(per_seed: dict, output_dir: Path) -> None:
    """Render per-seed AUCPR trajectories from the manuscript tables.

    The real-data artifacts currently preserve AUCPR summary values per seed, not
    full precision-recall coordinates. This figure therefore visualizes the
    available AUCPR evaluation points for each variant across seeds.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True, sharey=True)
    seeds = np.arange(5)
    y_min = min(per_seed[dataset]["aucpr"].min() for dataset in DATASET_ORDER)
    y_max = max(per_seed[dataset]["aucpr"].max() for dataset in DATASET_ORDER)
    padding = max(0.03, 0.08 * (y_max - y_min))

    for ax, dataset in zip(axes, DATASET_ORDER):
        values = per_seed[dataset]["aucpr"]
        for variant_idx, variant in enumerate(VARIANT_ORDER):
            variant_values = values[:, variant_idx]
            ax.plot(
                seeds,
                variant_values,
                marker="o",
                linewidth=2.2,
                markersize=6,
                color=VARIANT_COLORS[variant],
                label=VARIANT_LABELS[variant],
            )
            for seed, value in zip(seeds, variant_values, strict=True):
                ax.text(
                    seed,
                    value + 0.012,
                    f"{value:.3f}",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    color=VARIANT_COLORS[variant],
                )

        ax.set_title(f"{dataset} AUCPR by evaluation seed")
        ax.set_xlabel("Seed")
        ax.set_xticks(seeds)
        ax.set_xlim(seeds[0] - 0.15, seeds[-1] + 0.15)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(max(0.0, y_min - padding), min(1.0, y_max + padding))

    axes[0].set_ylabel("AUCPR")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Grouped real-data AUCPR curves from per-seed manuscript values", fontsize=14, y=1.08)
    _save_figure(fig, "realdata_aucpr_curves", output_dir)
    _export_realdata_aucpr_points(per_seed, output_dir)


def plot_synthetic_delta_bars(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Render horizontal bars for Fusion-Baseline AUCPR deltas."""
    fig, ax = plt.subplots(figsize=(11.6, 5.9), constrained_layout=True)
    positions = []
    labels = []
    deltas = []
    colors = []
    boundaries = []
    current_y = 0.0
    previous_category = None

    for row in rows:
        if previous_category is not None and row["category"] != previous_category:
            current_y += 0.6
            boundaries.append(current_y - 0.3)
        positions.append(current_y)
        labels.append(str(row["anomaly"]))
        delta = float(row["delta"])
        deltas.append(delta)
        colors.append(POSITIVE_COLOR if delta >= 0 else NEGATIVE_COLOR)
        current_y += 1.0
        previous_category = str(row["category"])

    bars = ax.barh(positions, deltas, color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(0.0, color="black", linewidth=1.0)
    for boundary in boundaries:
        ax.axhline(boundary, color="0.6", linewidth=0.8, linestyle="--")

    delta_min = min(deltas)
    delta_max = max(deltas)
    x_range = max(delta_max - delta_min, 0.08)
    x_left = delta_min - max(0.05, 0.35 * x_range)
    x_right = delta_max + max(0.04, 0.18 * x_range)
    label_x = x_left + 0.012
    ax.set_xlim(x_left, x_right)

    for bar, delta in zip(bars, deltas):
        x = bar.get_width()
        ha = "left" if delta >= 0 else "right"
        offset = 0.006 if delta >= 0 else -0.006
        ax.text(
            x + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{delta:+.3f}",
            va="center",
            ha=ha,
            fontsize=9,
            clip_on=False,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("AUCPR delta (Fusion - Baseline)")
    ax.set_title("Synthetic anomaly-type gains from posterior fusion")
    ax.grid(axis="x", alpha=0.25)

    category_to_rows: dict[str, list[float]] = {}
    for position, row in zip(positions, rows):
        category_to_rows.setdefault(str(row["category"]), []).append(position)
    for category, category_positions in category_to_rows.items():
        ax.text(
            label_x,
            np.mean(category_positions),
            category,
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )

    _save_figure(fig, "synthetic_anomaly_delta_bars", output_dir)


def _normalize_binary_with_train_stats(train_data: np.ndarray, binary_indices: set[int]) -> np.ndarray:
    """Normalize two-valued channels to [0, 1] using train-only statistics."""
    for idx in sorted(binary_indices):
        col_min = train_data[:, idx].min()
        col_max = train_data[:, idx].max()
        if col_max > col_min:
            train_data[:, idx] = (train_data[:, idx] - col_min) / (col_max - col_min)
    return train_data


def _prepare_swat_continuous_train() -> np.ndarray:
    """Recreate the SWaT train-only continuous feature matrix used for FS."""
    metric_tensor, _metric_test_tensor, _true_anomalies = load_swat_data(
        SWAT_NORMAL_DATASET,
        SWAT_ATTACK_DATASET,
    )
    split_idx = int(len(metric_tensor) * (1.0 - SWAT_VAL_RATIO))
    train_series = metric_tensor[:split_idx].astype(np.float32, copy=True)

    binary_feature_indices = detect_binary_features(train_series)
    all_indices = set(range(train_series.shape[1]))
    continuous_indices = sorted(all_indices - binary_feature_indices)
    if binary_feature_indices:
        train_series = _normalize_binary_with_train_stats(train_series, binary_feature_indices)
    train_series, _ = standardize_continuous_features(train_series, [], continuous_indices)
    return train_series[:, continuous_indices]


def _prepare_smd_train() -> np.ndarray:
    """Load and preprocess the SMD training split used in grouped experiments."""
    metric_train, _metric_test, _true_anomalies = load_smd_data(MACHINE, SMD_DRIVE)
    return preprocess_data(metric_train.astype(np.float32))


def plot_feature_similarity_heatmap(dataset: str, output_dir: Path) -> None:
    """Render a Stage-1 similarity heatmap with cluster boundary overlays."""
    if dataset == "swat":
        train_data = _prepare_swat_continuous_train()
        params = DEFAULT_PARAMS_SWAT
        title_prefix = "SWaT"
    else:
        train_data = _prepare_smd_train()
        params = DEFAULT_PARAMS_SMD
        title_prefix = "SMD"

    lag_penalty = params.get("lag_penalty_lambda")
    if lag_penalty == 0:
        lag_penalty = None

    diagnostics = compute_feature_selection_similarity_diagnostics(
        train_data,
        sequence_length=SEQUENCE_LENGTH,
        corr_threshold=params["corr_threshold"],
        lag_penalty_lambda=lag_penalty,
    )
    similarity = diagnostics["similarity_matrix"]
    cluster_labels = diagnostics["cluster_labels"]
    kept_indices = diagnostics["kept_feature_indices"]
    if similarity.size == 0:
        raise RuntimeError(f"No dynamic features available to plot for {dataset.upper()}")

    order = np.argsort(cluster_labels, kind="stable")
    ordered_similarity = similarity[np.ix_(order, order)]
    ordered_clusters = cluster_labels[order]
    ordered_features = kept_indices[order]
    boundaries = np.where(np.diff(ordered_clusters) != 0)[0] + 0.5

    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    image = ax.imshow(ordered_similarity, cmap=SIMILARITY_CMAP, vmin=0.0, vmax=1.0)
    for boundary in boundaries:
        ax.axhline(boundary, color="white", linewidth=1.0)
        ax.axvline(boundary, color="white", linewidth=1.0)

    tick_labels = [str(int(idx)) for idx in ordered_features]
    ax.set_xticks(np.arange(len(ordered_features)))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(ordered_features)))
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Continuous feature index (cluster-sorted)")
    ax.set_ylabel("Continuous feature index (cluster-sorted)")
    ax.set_title(
        f"{title_prefix} lagged Spearman feature similarity\n"
        f"{len(np.unique(cluster_labels))} clusters, threshold={params['corr_threshold']:.3f}"
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label="Similarity")
    _save_figure(fig, f"{dataset}_feature_similarity_heatmap", output_dir)


def generate_requested_figures(which: str, output_dir: Path, feature_dataset: str) -> None:
    """Generate the requested figure(s) in the required priority order."""
    aggregate, per_seed = load_realdata_results(EXPERIMENTS_DIR / "realdata.tex")
    synthetic_rows = load_synthetic_deltas(EXPERIMENTS_DIR / "synthetic.tex")

    tasks = [
        ("aggregate-bars", lambda: plot_realdata_aggregate_bars(aggregate, output_dir)),
        ("aucpr-curves", lambda: plot_realdata_aucpr_curves(per_seed, output_dir)),
        ("seed-heatmaps", lambda: plot_realdata_seed_heatmaps(per_seed, output_dir)),
        ("synthetic-delta-bars", lambda: plot_synthetic_delta_bars(synthetic_rows, output_dir)),
        (
            "feature-heatmap",
            lambda: plot_feature_similarity_heatmap(feature_dataset, output_dir),
        ),
    ]

    for name, task in tasks:
        if which in ("all", name):
            print(f"Generating {name}...")
            task()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=["all", "aggregate-bars", "aucpr-curves", "seed-heatmaps", "synthetic-delta-bars", "feature-heatmap"],
        default="all",
        help="Select one figure or generate all figures in sequence.",
    )
    parser.add_argument(
        "--feature-dataset",
        choices=["swat", "smd"],
        default="smd",
        help="Dataset used for the feature-similarity heatmap.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where figures will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_requested_figures(args.figure, args.output_dir, args.feature_dataset)
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
