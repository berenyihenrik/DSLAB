# -*- coding: utf-8 -*-
"""Visualize all 10 anomaly types from synthetic data generation."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from synthetic_data import (
    SyntheticConfig, ANOMALY_TYPES,
    generate_hidden_processes, generate_normal_series,
    inject_anomaly_segment,
)

# ── Settings ──────────────────────────────────────────────────────────
SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "anomaly_plots")
os.makedirs(OUT_DIR, exist_ok=True)
SINGLE_PLOT_TOP_TEXT_SIZE = 20
OVERVIEW_PLOT_TOP_TEXT_SIZE = 24

cfg = SyntheticConfig(test_T=3000, anomaly_len=(80, 80))
rng = np.random.default_rng(SEED)

# ── Generate a normal baseline series ─────────────────────────────────
hidden = generate_hidden_processes(cfg.test_T, cfg, rng)
normal_series = generate_normal_series(hidden, cfg, rng)

# Group metadata: name → column slice
GROUPS = {
    "G0 (base oscillator)": slice(0, 3),
    "G1 (lagged / transformed)": slice(3, 6),
    "G2 (regime + pulse)": slice(6, 9),
    "G3 (cross-group product)": slice(9, 12),
}

# Map each anomaly type to the group(s) it affects
AFFECTED = {
    "spatial_regime_desync_g1":      ["G1 (lagged / transformed)"],
    "spatial_routing_swap_g3":       ["G3 (cross-group product)"],
    "spatial_group_transplant_g1":   ["G1 (lagged / transformed)"],
    "spatial_sign_flip_g1":          ["G1 (lagged / transformed)"],
    "temporal_lag_shift_g1":         ["G1 (lagged / transformed)"],
    "temporal_phase_jump_g1":        ["G1 (lagged / transformed)"],
    "temporal_time_warp_g1":         ["G1 (lagged / transformed)"],
    "temporal_event_order_shift_g2": ["G2 (regime + pulse)"],
    "control_spike":                 list(GROUPS.keys()),
    "control_dropout":               list(GROUPS.keys()),
}

# Human-readable descriptions
DESCRIPTIONS = {
    "spatial_regime_desync_g1":      "Regime Desync – G1 receives inverted regime",
    "spatial_routing_swap_g3":       "Routing Swap – G3 route selection inverted",
    "spatial_group_transplant_g1":   "Group Transplant – G1 copied from distant window",
    "spatial_sign_flip_g1":          "Sign Flip – G1 channels negated",
    "temporal_lag_shift_g1":         "Lag Shift – G1 lag τ₁ changed from 3→7",
    "temporal_phase_jump_g1":        "Phase Jump – G1 phase offset +π/3",
    "temporal_time_warp_g1":         "Time Warp – G1 stretched/compressed ×0.7 or ×1.3",
    "temporal_event_order_shift_g2": "Event Order Shift – G2 pulses shifted ±6–10 steps",
    "control_spike":                 "Control: Spike – Gaussian envelope added to all channels",
    "control_dropout":               "Control: Dropout – All channels frozen to last value",
}

OVERVIEW_TITLES = {
    "spatial_regime_desync_g1":      "Regime desync (G1)",
    "spatial_routing_swap_g3":       "Routing swap (G3)",
    "spatial_group_transplant_g1":   "Group transplant (G1)",
    "spatial_sign_flip_g1":          "Sign flip (G1)",
    "temporal_lag_shift_g1":         "Lag shift (G1)",
    "temporal_phase_jump_g1":        "Phase jump (G1)",
    "temporal_time_warp_g1":         "Time warp (G1)",
    "temporal_event_order_shift_g2": "Event order shift (G2)",
    "control_spike":                 "Spike control (all)",
    "control_dropout":               "Dropout control (all)",
}


def plot_single_anomaly(anomaly_type, save_path):
    """Generate one figure showing normal vs. anomalous for a single type."""
    local_rng = np.random.default_rng(SEED + hash(anomaly_type) % 10000)
    hidden_local = generate_hidden_processes(cfg.test_T, cfg, local_rng)
    normal = generate_normal_series(hidden_local, cfg, local_rng)

    anom_series = normal.copy()
    anom_len = cfg.anomaly_len[0]
    start = cfg.test_T // 2 - anom_len // 2  # centre the anomaly
    inject_anomaly_segment(anom_series, hidden_local, anomaly_type, start, anom_len, cfg, local_rng)

    # Context window around the anomaly
    pad = 120
    lo = max(0, start - pad)
    hi = min(cfg.test_T, start + anom_len + pad)

    affected_groups = AFFECTED[anomaly_type]
    n_groups = len(affected_groups)

    fig, axes = plt.subplots(n_groups, 1, figsize=(14, 3.2 * n_groups), squeeze=False)
    fig.suptitle(
        DESCRIPTIONS[anomaly_type],
        fontsize=SINGLE_PLOT_TOP_TEXT_SIZE,
        fontweight="bold",
        y=1.03,
    )

    colors_normal = ["#2196F3", "#4CAF50", "#FF9800"]
    colors_anom   = ["#E53935", "#AB47BC", "#FF6F00"]

    for row, gname in enumerate(affected_groups):
        ax = axes[row, 0]
        cols = GROUPS[gname]
        t = np.arange(lo, hi)

        for j, (cn, ca) in enumerate(zip(colors_normal, colors_anom)):
            ch = cols.start + j
            ax.plot(t, normal[lo:hi, ch], color=cn, alpha=0.55, lw=1.0, label=f"normal ch{ch}")
            ax.plot(t, anom_series[lo:hi, ch], color=ca, alpha=0.85, lw=1.3, ls="--", label=f"anomalous ch{ch}")

        ax.axvspan(start, start + anom_len, color="red", alpha=0.08, label="anomaly window")
        ax.axvline(start, color="red", alpha=0.4, lw=0.8, ls=":")
        ax.axvline(start + anom_len, color="red", alpha=0.4, lw=0.8, ls=":")
        ax.set_ylabel(gname, fontsize=10)
        ax.set_xlim(lo, hi)
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        ax.grid(True, alpha=0.25)

    axes[-1, 0].set_xlabel("Time step", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


def plot_overview_grid(save_path):
    """Create a single overview figure with all 10 anomaly types in a grid."""
    fig = plt.figure(figsize=(20, 24))
    outer = gridspec.GridSpec(5, 2, hspace=0.35, wspace=0.25)

    for idx, anomaly_type in enumerate(ANOMALY_TYPES):
        local_rng = np.random.default_rng(SEED + hash(anomaly_type) % 10000)
        hidden_local = generate_hidden_processes(cfg.test_T, cfg, local_rng)
        normal = generate_normal_series(hidden_local, cfg, local_rng)

        anom_series = normal.copy()
        anom_len = cfg.anomaly_len[0]
        start = cfg.test_T // 2 - anom_len // 2
        inject_anomaly_segment(anom_series, hidden_local, anomaly_type, start, anom_len, cfg, local_rng)

        pad = 100
        lo = max(0, start - pad)
        hi = min(cfg.test_T, start + anom_len + pad)

        affected_groups = AFFECTED[anomaly_type]
        # Show first affected group's first channel for overview
        gname = affected_groups[0]
        cols = GROUPS[gname]

        ax = fig.add_subplot(outer[idx])
        t = np.arange(lo, hi)
        for j, (cn, ca) in enumerate(zip(
            ["#2196F3", "#4CAF50", "#FF9800"],
            ["#E53935", "#AB47BC", "#FF6F00"],
        )):
            ch = cols.start + j
            ax.plot(t, normal[lo:hi, ch], color=cn, alpha=0.5, lw=0.8)
            ax.plot(t, anom_series[lo:hi, ch], color=ca, alpha=0.85, lw=1.1, ls="--")

        ax.axvspan(start, start + anom_len, color="red", alpha=0.08)
        ax.axvline(start, color="red", alpha=0.35, lw=0.7, ls=":")
        ax.axvline(start + anom_len, color="red", alpha=0.35, lw=0.7, ls=":")
        ax.set_title(
            OVERVIEW_TITLES[anomaly_type],
            fontsize=OVERVIEW_PLOT_TOP_TEXT_SIZE,
            fontweight="bold",
        )
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")


if __name__ == "__main__":
    print("Generating individual anomaly plots...")
    for atype in ANOMALY_TYPES:
        fname = f"{atype}.png"
        plot_single_anomaly(atype, os.path.join(OUT_DIR, fname))

    print("\nGenerating overview grid...")
    plot_overview_grid(os.path.join(OUT_DIR, "overview_all_anomalies.png"))

    print(f"\nDone – {len(ANOMALY_TYPES) + 1} images saved to {OUT_DIR}/")
