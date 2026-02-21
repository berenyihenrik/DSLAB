# -*- coding: utf-8 -*-
"""Visualization functions for Optuna study results and anomaly attribution."""

import os
import numpy as np
import matplotlib.pyplot as plt
from optuna.trial import TrialState

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


def visualize_optuna_study(study, save_path=None, dataset_name="MSL", identifier="M-1"):
    """
    Generate visualizations for Optuna study results.
    
    Args:
        study: Optuna study object
        save_path: Optional path prefix to save figures
        dataset_name: Dataset name for titles (e.g., "MSL", "SMAP", "SMD")
        identifier: Channel or machine identifier for titles
    """
    # Create figure for optimization history
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    trial_values = [t.value for t in study.trials if t.value is not None]
    trial_numbers = [t.number for t in study.trials if t.value is not None]
    
    ax1.plot(trial_numbers, trial_values, 'bo-', alpha=0.6, label='Trial F1 Score')
    
    # Add best value line
    best_values = []
    current_best = 0
    for val in trial_values:
        if val > current_best:
            current_best = val
        best_values.append(current_best)
    ax1.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best F1 Score')
    
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('F1 Score')
    ax1.set_title(f'Optuna Optimization History - {dataset_name} {identifier}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(f'{save_path}_optimization_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create figure for KL weight analysis
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Extract parameter values from completed trials
    completed_trials = [t for t in study.trials if t.value is not None]
    
    if len(completed_trials) > 0:
        kl_weights = [t.params.get('kl_weight', None) for t in completed_trials]
        f1_scores = [t.value for t in completed_trials]
        
        # Filter out None values
        valid_data = [(kw, f1) for kw, f1 in zip(kl_weights, f1_scores) if kw is not None]
        
        if valid_data:
            kl_weights, f1_scores = zip(*valid_data)
            
            scatter = ax2.scatter(kl_weights, f1_scores, c=range(len(kl_weights)), 
                                  cmap='viridis', alpha=0.7, s=50)
            
            # Highlight best trial
            best_idx = np.argmax(f1_scores)
            ax2.scatter([kl_weights[best_idx]], [f1_scores[best_idx]], 
                       c='red', s=200, marker='*', edgecolors='black', 
                       linewidths=2, label=f'Best (kl_weight={kl_weights[best_idx]:.3f})')
            
            ax2.set_xlabel('KL Weight')
            ax2.set_ylabel('F1 Score')
            ax2.set_title(f'KL Weight vs Model Performance - {dataset_name} {identifier}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.colorbar(scatter, ax=ax2, label='Trial Number')
    
    if save_path:
        plt.savefig(f'{save_path}_weight_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Parameter distribution plot
    fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    param_names = ['kl_weight', 'hidden_dim', 'latent_dim', 'learning_rate', 
                   'batch_size', 'percentile_threshold']
    
    for idx, param_name in enumerate(param_names):
        if idx < len(axes):
            param_values = [t.params.get(param_name, None) for t in completed_trials]
            trial_f1s = [t.value for t in completed_trials]
            
            valid_data = [(p, f1) for p, f1 in zip(param_values, trial_f1s) if p is not None]
            
            if valid_data:
                param_vals, f1_vals = zip(*valid_data)
                axes[idx].scatter(param_vals, f1_vals, alpha=0.6)
                axes[idx].set_xlabel(param_name)
                axes[idx].set_ylabel('F1 Score')
                axes[idx].set_title(f'{param_name} vs F1 Score')
                axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(f'Parameter Analysis - {dataset_name} {identifier}')
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}_parameter_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_optuna_summary(study, dataset_name="MSL", identifier="M-1"):
    """
    Print a detailed summary of the Optuna study results.
    
    Args:
        study: Optuna study object
        dataset_name: Dataset name for display (e.g., "MSL", "SMAP", "SMD")
        identifier: Channel or machine identifier for display
    """
    print("\n" + "=" * 70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    print(f"\nDataset: {dataset_name}, Identifier: {identifier}")
    print(f"Total trials: {len(study.trials)}")
    
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    
    print(f"Completed trials: {len(completed)}")
    print(f"Pruned trials: {len(pruned)}")
    
    if completed:
        f1_scores = [t.value for t in completed]
        print(f"\nF1 Score Statistics:")
        print(f"  Best:   {max(f1_scores):.4f}")
        print(f"  Mean:   {np.mean(f1_scores):.4f}")
        print(f"  Std:    {np.std(f1_scores):.4f}")
        print(f"  Median: {np.median(f1_scores):.4f}")
        
        print(f"\nBest Hyperparameters:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        # Analyze KL weight importance
        kl_weights = [t.params.get('kl_weight', 0.1) for t in completed]
        weight_corr = np.corrcoef(kl_weights, f1_scores)[0, 1]
        print(f"\nKL Weight Analysis:")
        print(f"  Correlation with F1 Score: {weight_corr:.4f}")
        print(f"  Optimal kl_weight: {study.best_params.get('kl_weight', 'N/A')}")
    
    print("=" * 70)


def print_final_summary(dataset_name, identifier, best_params, f1, pa_results=None):
    """
    Print final model performance summary.
    
    Args:
        dataset_name: Dataset name (e.g., "MSL", "SMAP", "SMD")
        identifier: Channel or machine identifier
        best_params: Best hyperparameters dictionary
        f1: Point-wise F1 score
        pa_results: Point-adjust evaluation results (optional, for SMAP/MSL)
    """
    print("\n" + "=" * 70)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"\nDataset: {dataset_name}, Identifier: {identifier}")
    print(f"\nOptimized Hyperparameters:")
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print(f"\nPerformance Metrics:")
    print(f"  Point-wise F1 Score: {f1:.4f}")
    if pa_results is not None:
        print(f"  Point-Adjust F1 Score: {pa_results['f1']:.4f}")
        print(f"  Detected Anomaly Segments: {pa_results['true_positives']} / {pa_results['total_segments']}")
    print("=" * 70)


def plot_anomaly_attribution_heatmap(
    attribution,
    normalize="total",
    order="group",
    show_group_boundaries=True,
    kl_weight=0.1,
    cmap="inferno",
    save_path=None,
):
    """
    Plot a two-panel anomaly attribution figure:
      Top:    per-group bar chart (reconstruction + KL contributions)
      Bottom: (timestep × feature) heatmap of squared reconstruction error

    Args:
        attribution: Dict returned by extract_anomaly_attributions for one sample.
        normalize: How to normalize the heatmap.
                   "total"   — divide by total sum (shows relative contribution)
                   "feature" — normalize each feature column independently
                   "none"    — raw squared error values
        order: Column ordering. "group" keeps features clustered by encoder group;
               "sorted" uses the default sorted feature order.
        show_group_boundaries: Draw vertical lines between encoder groups.
        kl_weight: KL weight (for scaling KL bar to score units).
        cmap: Matplotlib/seaborn colormap name.
        save_path: If set, save figure to this path.

    Returns:
        fig: The matplotlib Figure object.
    """
    T, F = attribution["recon_sqerr_t_f"].shape
    err = attribution["recon_sqerr_t_f"].copy()

    if normalize == "total":
        denom = err.sum() + 1e-12
        err = err / denom
    elif normalize == "feature":
        denom = err.sum(axis=0, keepdims=True) + 1e-12
        err = err / denom

    feature_order = attribution["feature_order"]
    encoder_groups = attribution["encoder_groups"]

    col_perm = np.arange(F)
    group_boundaries = None

    if order == "group":
        feat_to_col = {fid: i for i, fid in enumerate(feature_order)}
        by_group_cols = []
        boundaries = []
        c = 0
        for g in encoder_groups:
            cols = [feat_to_col[fid] for fid in g if fid in feat_to_col]
            by_group_cols.extend(cols)
            c += len(cols)
            boundaries.append(c)
        col_perm = np.array(by_group_cols, dtype=int)
        group_boundaries = boundaries[:-1]

    err = err[:, col_perm]
    xlabels = [str(feature_order[i]) for i in col_perm]

    # --- Figure layout ---
    fig = plt.figure(figsize=(max(12, F * 0.4), 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.25)

    # Top panel: per-group contributions
    ax0 = fig.add_subplot(gs[0])
    recon_g = attribution["recon_contrib_by_group"]
    kld_g = attribution["kld_contrib_by_group"] * kl_weight
    n_groups = len(recon_g)
    ind = np.arange(n_groups)
    ax0.bar(ind, recon_g, label="Reconstruction", alpha=0.85, color="#e74c3c")
    ax0.bar(ind, kld_g, bottom=recon_g, label="KL divergence", alpha=0.7, color="#3498db")
    ax0.set_xticks(ind)
    ax0.set_xticklabels([f"Group {i}" for i in range(n_groups)])
    ax0.set_ylabel("Score contribution")
    ax0.set_title(
        f"Anomaly Attribution — window idx {attribution['index']}  "
        f"(score = {attribution['score_total']:.4f})"
    )
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(axis="y", alpha=0.3)

    # Bottom panel: (timestep × feature) heatmap
    ax1 = fig.add_subplot(gs[1])
    if _HAS_SEABORN:
        sns.heatmap(
            err, ax=ax1, cmap=cmap, cbar=True,
            xticklabels=xlabels,
            yticklabels=5,
        )
    else:
        im = ax1.imshow(err, aspect="auto", interpolation="nearest", cmap=cmap)
        plt.colorbar(im, ax=ax1)
        ax1.set_xticks(np.arange(F))
        ax1.set_xticklabels(xlabels, rotation=90)

    ax1.set_xlabel("Feature (original index)")
    ax1.set_ylabel("Timestep in window")

    if show_group_boundaries and group_boundaries is not None:
        for b in group_boundaries:
            ax1.axvline(b, color="white", linewidth=2, linestyle="--")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_top_anomaly_heatmaps(attributions, top_n=5, save_dir=None, **kwargs):
    """
    Plot heatmaps for the top-N highest-scoring anomalies.

    Args:
        attributions: List of attribution dicts from extract_anomaly_attributions.
        top_n: Number of top anomalies to plot.
        save_dir: Directory to save figures into. Each file is named
                  anomaly_heatmap_idx{index}.png. If None, figures are not saved.
        **kwargs: Additional keyword arguments passed to plot_anomaly_attribution_heatmap.

    Returns:
        List of matplotlib Figure objects.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    sorted_attrs = sorted(attributions, key=lambda a: a["score_total"], reverse=True)
    figs = []
    for attr in sorted_attrs[:top_n]:
        sp = None
        if save_dir is not None:
            sp = os.path.join(save_dir, f"anomaly_heatmap_idx{attr['index']}.png")
        fig = plot_anomaly_attribution_heatmap(attr, save_path=sp, **kwargs)
        figs.append(fig)
    return figs

def extract_group_error(attribution, group_id):
    """Extract the (T, n_features) error matrix for one encoder group."""
    feature_order = attribution["feature_order"]
    encoder_groups = attribution["encoder_groups"]
    recon_sqerr_t_f = attribution["recon_sqerr_t_f"]

    feat_to_col = {fid: i for i, fid in enumerate(feature_order)}
    group_fids = encoder_groups[group_id]
    cols = [feat_to_col[fid] for fid in group_fids if fid in feat_to_col]

    if not cols:
        return np.empty((recon_sqerr_t_f.shape[0], 0)), []

    err = recon_sqerr_t_f[:, cols]
    labels = [str(feature_order[c]) for c in cols]
    return err, labels

def detect_onsets(err, baseline_len=8, z_thresh=3.0, run_len=2, frac_max=0.2):
    """Detect onset timestep for each feature trace using robust z-scores."""
    if err.size == 0:
        return []

    T, n_features = err.shape
    baseline_len = min(baseline_len, T)
    baseline = err[:baseline_len]
    med = np.median(baseline, axis=0)
    mad = np.median(np.abs(baseline - med), axis=0)
    mad = np.where(mad < 1e-12, 1.0, mad)

    z = (err - med) / mad
    onsets = []
    for j in range(n_features):
        max_val = np.max(err[:, j]) if T > 0 else 0.0
        if max_val <= 0:
            onsets.append(None)
            continue
        mag_gate = frac_max * max_val
        onset_idx = None
        for t in range(T - run_len + 1):
            if np.all(z[t:t + run_len, j] >= z_thresh) and err[t, j] >= mag_gate:
                onset_idx = t
                break
        onsets.append(onset_idx)
    return onsets

def _smooth_traces(err, window=3):
    if window is None or window <= 1 or err.size == 0:
        return err
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(err, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), 0, padded)
    return smoothed

def plot_stacked_time_series_traces(
    attribution,
    group_id,
    normalize="feature",
    smooth_window=3,
    baseline_len=8,
    z_thresh=3.0,
    run_len=2,
    frac_max=0.2,
    offset=None,
    annotate_onsets=True,
    save_path=None,
):
    """Plot stacked time series traces for one encoder group."""
    err, labels = extract_group_error(attribution, group_id)
    err = _smooth_traces(err, window=smooth_window)

    if normalize == "feature":
        denom = err.sum(axis=0, keepdims=True) + 1e-12
        err = err / denom
    elif normalize == "total":
        denom = err.sum() + 1e-12
        err = err / denom

    onsets = detect_onsets(err, baseline_len, z_thresh, run_len, frac_max)

    T, n_features = err.shape
    fig, ax = plt.subplots(figsize=(12, max(4.0, 0.45 * n_features + 2.0)))

    if n_features == 0:
        ax.set_title(f"Group {group_id} — no features")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    if offset is None:
        peak = np.max(err) if err.size else 1.0
        offset = 1.15 * peak if peak > 0 else 1.0

    x = np.arange(T)
    for idx in range(n_features):
        y = err[:, idx] + (n_features - 1 - idx) * offset
        ax.plot(x, y, linewidth=1.4)
        if annotate_onsets and onsets[idx] is not None:
            ax.axvline(onsets[idx], color="black", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.scatter([onsets[idx]], [y[onsets[idx]]], color="black", s=12, zorder=4)

    ax.set_yticks([(n_features - 1 - idx) * offset for idx in range(n_features)])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Timestep in window")
    ax.set_title(
        f"Stacked Error Traces — group {group_id} (window idx {attribution['index']})"
    )
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlim(0, T - 1 if T > 0 else 0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig

def plot_top_group_traces(attributions, top_n=5, group_id=None, save_dir=None, **kwargs):
    """Plot stacked traces for the top-N anomalies and a selected group."""
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    sorted_attrs = sorted(attributions, key=lambda a: a["score_total"], reverse=True)
    figs = []
    for attr in sorted_attrs[:top_n]:
        sel_group = group_id
        if sel_group is None:
            sel_group = int(np.argmax(attr["recon_contrib_by_group"]))
        sp = None
        if save_dir is not None:
            sp = os.path.join(
                save_dir, f"stacked_traces_idx{attr['index']}_group{sel_group}.png"
            )
        fig = plot_stacked_time_series_traces(
            attr, sel_group, save_path=sp, **kwargs
        )
        figs.append(fig)
    return figs
