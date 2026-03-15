# -*- coding: utf-8 -*-
"""A/B test: fusion variants on SMD machine-1-1."""

import os
import random
import numpy as np
import torch
from functools import partial
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score

from config import (
    SMD_DRIVE, MACHINE, SEQUENCE_LENGTH, DEFAULT_PARAMS_SMD, DEVICE,
    NUM_EPOCHS, USE_AMP, DATALOADER_WORKERS, PIN_MEMORY,
)
from data_loader import load_smd_data, preprocess_data, create_grouped_sequences
from models import LSTMVAE_Grouped
from training import loss_function_grouped, train_model_grouped
from feature_selection import perform_feature_selection, split_features_by_groups
from evaluation import fit_group_ecdf, compute_anomaly_scores_grouped, compute_threshold_from_baseline


SEEDS = [0, 1, 2]
FUSION_TYPES = ["none", "mlp", "attn_mean"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_single(seed, fusion_type):
    """Train and evaluate one model. Returns dict of metrics."""
    set_seed(seed)
    device = DEVICE
    params = DEFAULT_PARAMS_SMD.copy()

    # --- data (deterministic given seed for train/val split) ---
    metric_train, metric_test, true_anomalies = load_smd_data(MACHINE, SMD_DRIVE)
    metric_train = preprocess_data(metric_train.astype(np.float32))
    metric_test = preprocess_data(metric_test.astype(np.float32))

    seq_len = SEQUENCE_LENGTH

    # Feature selection (deterministic given data)
    encoder_groups, _ = perform_feature_selection(
        metric_train, metric_train.shape[1], seq_len, device,
        corr_threshold=params.get("corr_threshold", 0.9),
        importance_percentile=params.get("importance_percentile", 50),
        lag_penalty_lambda=params.get("lag_penalty_lambda", 0),
    )

    data_groups_train = split_features_by_groups(metric_train, encoder_groups)
    data_groups_test = split_features_by_groups(metric_test, encoder_groups)

    seqs_train = create_grouped_sequences(data_groups_train, seq_len)
    seqs_test = create_grouped_sequences(data_groups_test, seq_len)

    train_data, val_data = train_test_split(seqs_train, test_size=0.3, random_state=42)

    num_workers = min(DATALOADER_WORKERS, max(0, (os.cpu_count() or 2) // 2))
    lk = dict(num_workers=num_workers,
              pin_memory=torch.cuda.is_available() and PIN_MEMORY,
              persistent_workers=num_workers > 0)
    bs = params["batch_size"]
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, **lk)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False, **lk)
    test_loader = DataLoader(seqs_test, batch_size=bs, shuffle=False, **lk)

    # --- model ---
    tag = fusion_type
    print(f"\n{'='*60}")
    print(f"  seed={seed}  variant={tag}  groups={len(encoder_groups)}")
    print(f"{'='*60}")

    model = LSTMVAE_Grouped(
        encoder_groups=encoder_groups,
        hidden_dim=params["hidden_dim"],
        latent_dim=params["latent_dim"],
        sequence_length=seq_len,
        num_layers=params["num_layers"],
        device=device,
        fusion_type=fusion_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=params["learning_rate"])
    kl_weight = params.get("kl_weight", 0.1)
    loss_fn = partial(loss_function_grouped, kl_weight=kl_weight)

    train_model_grouped(
        model, train_loader, val_loader,
        optimizer, loss_fn, scheduler=None,
        num_epochs=NUM_EPOCHS, device=device, use_amp=USE_AMP,
    )

    # --- evaluation with validation-calibrated threshold ---
    baseline_ecdfs = fit_group_ecdf(model, val_loader, device)
    test_scores = compute_anomaly_scores_grouped(
        model, test_loader, device, baseline_ecdfs=baseline_ecdfs)
    threshold, _ = compute_threshold_from_baseline(
        model, val_loader, device, params["percentile_threshold"],
        baseline_ecdfs=baseline_ecdfs)

    adjusted_true = true_anomalies[seq_len - 1:]
    scores_arr = np.array(test_scores[: len(adjusted_true)])
    preds = (scores_arr > threshold).astype(int)

    f1 = f1_score(adjusted_true, preds, zero_division=0)
    aucpr = average_precision_score(adjusted_true, scores_arr) if len(np.unique(adjusted_true)) > 1 else float("nan")

    # Score separation diagnostic
    normal_mask = adjusted_true == 0
    anom_mask = adjusted_true == 1
    sep = float(scores_arr[anom_mask].mean() - scores_arr[normal_mask].mean()) if anom_mask.any() else float("nan")

    print(f"  F1={f1:.4f}  AUCPR={aucpr:.4f}  score_sep={sep:.4f}  threshold={threshold:.4f}")

    return dict(seed=seed, variant=tag, f1=f1, aucpr=aucpr, score_sep=sep,
                threshold=threshold, n_params=n_params)


def main():
    results = []
    for seed in SEEDS:
        for fusion_type in FUSION_TYPES:
            res = run_single(seed, fusion_type)
            results.append(res)

    # --- summary table ---
    print("\n" + "=" * 72)
    print(f"{'Seed':>4}  {'Variant':>8}  {'F1':>7}  {'AUCPR':>7}  {'ScoreSep':>9}  {'Params':>9}")
    print("-" * 72)
    for r in results:
        print(f"{r['seed']:>4}  {r['variant']:>8}  {r['f1']:7.4f}  {r['aucpr']:7.4f}  "
              f"{r['score_sep']:9.4f}  {r['n_params']:>9,}")

    # Aggregate
    for variant in FUSION_TYPES:
        subset = [r for r in results if r["variant"] == variant]
        f1s = [r["f1"] for r in subset]
        aucprs = [r["aucpr"] for r in subset]
        print(f"\n  {variant:>8}  F1 mean={np.mean(f1s):.4f} ± {np.std(f1s):.4f}  "
              f"AUCPR mean={np.mean(aucprs):.4f} ± {np.std(aucprs):.4f}")

    # Decision
    base_f1s = [r["f1"] for r in results if r["variant"] == "none"]
    for candidate in [v for v in FUSION_TYPES if v != "none"]:
        cand_f1s = [r["f1"] for r in results if r["variant"] == candidate]
        wins = sum(c > b for c, b in zip(cand_f1s, base_f1s))
        print(f"\n  {candidate} wins {wins}/{len(SEEDS)} seeds on F1 vs none.")


if __name__ == "__main__":
    main()
