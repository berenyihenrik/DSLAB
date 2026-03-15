# -*- coding: utf-8 -*-
"""A/B test: baseline vs ResidualMLPFusion on synthetic grouped anomalies."""

import os
import random
from functools import partial

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import DATALOADER_WORKERS, DEVICE, PIN_MEMORY, USE_AMP
from data_loader import create_grouped_sequences
from evaluation import (
    compute_anomaly_scores_grouped,
    compute_threshold_from_baseline,
    fit_group_ecdf,
)
from feature_selection import split_features_by_groups
from models import LSTMVAE_Grouped
from synthetic_data import (
    ANOMALY_TYPES,
    SyntheticConfig,
    build_test_with_anomalies,
    generate_hidden_processes,
    generate_normal_series,
    standardize_train_test,
)
from training import loss_function_grouped, train_model_grouped


SEEDS = [0, 1, 2, 3, 4]
ENCODER_GROUPS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
HARD_TYPES = [
    "spatial_regime_desync_g1",
    "spatial_routing_swap_g3",
    "spatial_group_transplant_g1",
    "spatial_sign_flip_g1",
    "temporal_lag_shift_g1",
    "temporal_phase_jump_g1",
    "temporal_time_warp_g1",
    "temporal_event_order_shift_g2",
]
CONTROL_TYPES = ["control_spike", "control_dropout"]

HIDDEN_DIM = 64
LATENT_DIM = 4
NUM_LAYERS = 1
KL_WEIGHT = 0.01
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
NUM_EPOCHS = 40
PERCENTILE_THRESHOLD = 95


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _generate_synthetic_dataset(seed, cfg):
    rng = np.random.default_rng(seed)

    hidden_train = generate_hidden_processes(cfg.train_T, cfg, rng)
    train_normal = generate_normal_series(hidden_train, cfg, rng)

    hidden_test = generate_hidden_processes(cfg.test_T, cfg, rng)
    test_normal = generate_normal_series(hidden_test, cfg, rng)
    test_with_anom, labels, type_labels, segments = build_test_with_anomalies(
        test_normal, hidden_test, cfg, rng, n_segments=10
    )

    train_std, test_std = standardize_train_test(train_normal, test_with_anom)
    return train_std, test_std, labels, type_labels, segments


def _type_metrics(scores, threshold, adjusted_true, adjusted_type):
    normal_mask = adjusted_true == 0
    mask = normal_mask | (adjusted_type == 1)
    y = adjusted_type[mask]
    s = scores[mask]

    preds = (s > threshold).astype(int)
    f1 = f1_score(y, preds, zero_division=0)
    aucpr = average_precision_score(y, s) if len(np.unique(y)) > 1 else float("nan")

    pos = y == 1
    neg = y == 0
    sep = float(s[pos].mean() - s[neg].mean()) if pos.any() and neg.any() else float("nan")
    return f1, aucpr, sep


def run_single(seed, use_fusion):
    set_seed(seed)
    device = DEVICE
    cfg = SyntheticConfig()

    train_x, test_x, true_anomalies, type_labels, segments = _generate_synthetic_dataset(seed, cfg)

    data_groups_train = split_features_by_groups(train_x, ENCODER_GROUPS)
    data_groups_test = split_features_by_groups(test_x, ENCODER_GROUPS)
    seqs_train = create_grouped_sequences(data_groups_train, cfg.seq_len)
    seqs_test = create_grouped_sequences(data_groups_test, cfg.seq_len)
    train_data, val_data = train_test_split(seqs_train, test_size=0.3, random_state=42)

    num_workers = min(DATALOADER_WORKERS, max(0, (os.cpu_count() or 2) // 2))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() and PIN_MEMORY,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(seqs_test, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    tag = "fusion" if use_fusion else "baseline"
    print(f"\n{'='*72}")
    print(f"seed={seed} variant={tag} groups={len(ENCODER_GROUPS)}")
    print(f"{'='*72}")

    model = LSTMVAE_Grouped(
        encoder_groups=ENCODER_GROUPS,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        sequence_length=cfg.seq_len,
        num_layers=NUM_LAYERS,
        device=device,
        use_fusion=use_fusion,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = partial(loss_function_grouped, kl_weight=KL_WEIGHT)
    train_model_grouped(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        scheduler=None,
        num_epochs=NUM_EPOCHS,
        device=device,
        use_amp=USE_AMP,
    )

    baseline_ecdfs = fit_group_ecdf(model, val_loader, device)
    test_scores = compute_anomaly_scores_grouped(model, test_loader, device, baseline_ecdfs=baseline_ecdfs)
    threshold, _ = compute_threshold_from_baseline(
        model,
        val_loader,
        device,
        PERCENTILE_THRESHOLD,
        baseline_ecdfs=baseline_ecdfs,
    )

    adjusted_true = true_anomalies[cfg.seq_len - 1:]
    scores_arr = np.array(test_scores[: len(adjusted_true)])
    preds = (scores_arr > threshold).astype(int)
    f1 = f1_score(adjusted_true, preds, zero_division=0)
    aucpr = average_precision_score(adjusted_true, scores_arr)
    sep = float(scores_arr[adjusted_true == 1].mean() - scores_arr[adjusted_true == 0].mean())

    print(f"overall: F1={f1:.4f} AUCPR={aucpr:.4f} sep={sep:.4f} thr={threshold:.4f}")

    per_type = {}
    for anomaly_type in ANOMALY_TYPES:
        t = type_labels[anomaly_type][cfg.seq_len - 1:]
        t = t[: len(scores_arr)]
        tf1, taucpr, tsep = _type_metrics(scores_arr, threshold, adjusted_true, t)
        per_type[anomaly_type] = {
            "f1": tf1,
            "aucpr": taucpr,
            "score_sep": tsep,
        }
        print(f"  {anomaly_type:30s} F1={tf1:.4f} AUCPR={taucpr:.4f} sep={tsep:.4f}")

    return {
        "seed": seed,
        "variant": tag,
        "f1": f1,
        "aucpr": aucpr,
        "score_sep": sep,
        "threshold": threshold,
        "per_type": per_type,
        "segments": segments,
    }


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def main():
    results = []
    for seed in SEEDS:
        for use_fusion in [False, True]:
            results.append(run_single(seed, use_fusion))

    print("\n" + "=" * 86)
    print(f"{'Seed':>4}  {'Variant':>8}  {'F1':>7}  {'AUCPR':>7}  {'ScoreSep':>9}")
    print("-" * 86)
    for r in results:
        print(f"{r['seed']:>4}  {r['variant']:>8}  {r['f1']:7.4f}  {r['aucpr']:7.4f}  {r['score_sep']:9.4f}")

    for variant in ["baseline", "fusion"]:
        subset = [r for r in results if r["variant"] == variant]
        f1_mean, f1_std = _mean_std([r["f1"] for r in subset])
        auc_mean, auc_std = _mean_std([r["aucpr"] for r in subset])
        print(f"\n{variant:>8}  F1={f1_mean:.4f} ± {f1_std:.4f}  AUCPR={auc_mean:.4f} ± {auc_std:.4f}")

    print("\nPer-type AUCPR mean ± std")
    print("-" * 86)
    for anomaly_type in ANOMALY_TYPES:
        for variant in ["baseline", "fusion"]:
            vals = [
                r["per_type"][anomaly_type]["aucpr"]
                for r in results
                if r["variant"] == variant
            ]
            mu, sigma = _mean_std(vals)
            print(f"{anomaly_type:30s} {variant:>8s}: {mu:.4f} ± {sigma:.4f}")

    print("\nWin counts by anomaly type (AUCPR)")
    print("-" * 86)
    for anomaly_type in ANOMALY_TYPES:
        wins = 0
        for seed in SEEDS:
            b = next(r for r in results if r["seed"] == seed and r["variant"] == "baseline")
            f = next(r for r in results if r["seed"] == seed and r["variant"] == "fusion")
            wins += int(f["per_type"][anomaly_type]["aucpr"] > b["per_type"][anomaly_type]["aucpr"])
        print(f"{anomaly_type:30s} fusion wins {wins}/{len(SEEDS)}")

    hard_wins = 0
    control_deltas = []
    for seed in SEEDS:
        b = next(r for r in results if r["seed"] == seed and r["variant"] == "baseline")
        f = next(r for r in results if r["seed"] == seed and r["variant"] == "fusion")

        b_hard = np.nanmean([b["per_type"][t]["aucpr"] for t in HARD_TYPES])
        f_hard = np.nanmean([f["per_type"][t]["aucpr"] for t in HARD_TYPES])
        hard_wins += int(f_hard > b_hard)

        b_ctl = np.nanmean([b["per_type"][t]["aucpr"] for t in CONTROL_TYPES])
        f_ctl = np.nanmean([f["per_type"][t]["aucpr"] for t in CONTROL_TYPES])
        control_deltas.append(float(f_ctl - b_ctl))

    print("\nAcceptance diagnostics")
    print("-" * 86)
    print(f"Fusion wins on hard anomalies (seed-wise mean AUCPR): {hard_wins}/{len(SEEDS)}")
    print(f"Control AUCPR delta (fusion-baseline): mean={np.mean(control_deltas):.4f} std={np.std(control_deltas):.4f}")


if __name__ == "__main__":
    main()
