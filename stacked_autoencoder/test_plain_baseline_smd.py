# -*- coding: utf-8 -*-
"""Plain LSTM-VAE baseline on SMD (machine-1-1) for comparison with grouped architecture.

Uses a single encoder group containing all 38 features (no feature selection,
no grouping). Parameter count is matched to the grouped model (~5.9M) by
scaling hidden_dim to 454.

Mirrors test_fusion_ab.py protocol: 5 seeds, contiguous 70/30 train/val split,
NUM_EPOCHS=256, no subsampling.
"""

import os
import random
import numpy as np
import torch
from functools import partial
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, average_precision_score

from config import (
    SMD_DRIVE, MACHINE, SEQUENCE_LENGTH, DEFAULT_PARAMS_SMD, DEVICE,
    NUM_EPOCHS, USE_AMP, DATALOADER_WORKERS, PIN_MEMORY,
)
from data_loader import load_smd_data, preprocess_data, create_grouped_sequences
from models import LSTMVAE_Grouped
from training import loss_function_grouped, train_model_grouped
from feature_selection import split_features_by_groups
from evaluation import fit_group_ecdf, compute_anomaly_scores_grouped, compute_threshold_from_baseline


SEEDS = [0, 1, 2, 3, 4]

# Plain model hyperparams — param-matched to grouped (5,904,834 → 5,890,260)
PLAIN_HIDDEN_DIM = 454
PLAIN_LATENT_DIM = 13
PLAIN_NUM_LAYERS = 2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def contiguous_train_val_split(seqs_train, val_ratio=0.3):
    """Split sequence windows into contiguous train/val partitions."""
    n_total = len(seqs_train)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError("Validation ratio leaves no samples for training.")
    return seqs_train[:n_train], seqs_train[n_train:]


def run_single(seed, train_data, val_data, seqs_test,
               true_anomalies, encoder_groups, params, seq_len, device):
    """Train and evaluate one plain model. Returns dict of metrics."""
    set_seed(seed)

    num_workers = min(DATALOADER_WORKERS, max(0, (os.cpu_count() or 2) // 2))
    lk = dict(num_workers=num_workers,
              pin_memory=torch.cuda.is_available() and PIN_MEMORY,
              persistent_workers=num_workers > 0)
    bs = params["batch_size"]
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, **lk)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False, **lk)
    test_loader = DataLoader(seqs_test, batch_size=bs, shuffle=False, **lk)

    print(f"\n{'='*60}")
    print(f"  seed={seed}  variant=plain")
    print(f"{'='*60}")

    model = LSTMVAE_Grouped(
        encoder_groups=encoder_groups,
        hidden_dim=PLAIN_HIDDEN_DIM,
        latent_dim=PLAIN_LATENT_DIM,
        sequence_length=seq_len,
        num_layers=PLAIN_NUM_LAYERS,
        device=device,
        fusion_type="none",
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

    # Evaluation with validation-calibrated threshold
    baseline_ecdfs = fit_group_ecdf(model, val_loader, device)
    test_scores = compute_anomaly_scores_grouped(
        model, test_loader, device, baseline_ecdfs=baseline_ecdfs)
    threshold, _ = compute_threshold_from_baseline(
        model, val_loader, device, params["percentile_threshold"],
        baseline_ecdfs=baseline_ecdfs)

    adjusted_true = true_anomalies[seq_len - 1:]
    scores_arr = np.array(test_scores[:len(adjusted_true)])
    preds = (scores_arr > threshold).astype(int)

    f1 = f1_score(adjusted_true, preds, zero_division=0)
    aucpr = average_precision_score(adjusted_true, scores_arr) if len(np.unique(adjusted_true)) > 1 else float("nan")

    normal_mask = adjusted_true == 0
    anom_mask = adjusted_true == 1
    sep = float(scores_arr[anom_mask].mean() - scores_arr[normal_mask].mean()) if anom_mask.any() else float("nan")

    print(f"  F1={f1:.4f}  AUCPR={aucpr:.4f}  score_sep={sep:.4f}  threshold={threshold:.4f}")

    return dict(seed=seed, variant="plain", f1=f1, aucpr=aucpr, score_sep=sep,
                threshold=threshold, n_params=n_params)


def main():
    device = DEVICE
    params = DEFAULT_PARAMS_SMD.copy()
    seq_len = SEQUENCE_LENGTH

    set_seed(42)

    print(f"Using device: {device}")
    print(f"Plain model: hidden_dim={PLAIN_HIDDEN_DIM}, latent_dim={PLAIN_LATENT_DIM}, "
          f"num_layers={PLAIN_NUM_LAYERS}")

    metric_train, metric_test, true_anomalies = load_smd_data(MACHINE, SMD_DRIVE)
    metric_train = preprocess_data(metric_train.astype(np.float32))
    metric_test = preprocess_data(metric_test.astype(np.float32))

    n_features = metric_train.shape[1]
    encoder_groups = [list(range(n_features))]
    print(f"Plain model: 1 encoder group with all {n_features} features")

    data_groups_train = split_features_by_groups(metric_train, encoder_groups)
    data_groups_test = split_features_by_groups(metric_test, encoder_groups)
    seqs_train = create_grouped_sequences(data_groups_train, seq_len)
    seqs_test = create_grouped_sequences(data_groups_test, seq_len)

    train_data, val_data = contiguous_train_val_split(seqs_train, val_ratio=0.3)
    print(f"Contiguous split: train_windows={len(train_data)} val_windows={len(val_data)}")

    results = []
    for seed in SEEDS:
        res = run_single(
            seed=seed,
            train_data=train_data,
            val_data=val_data,
            seqs_test=seqs_test,
            true_anomalies=true_anomalies,
            encoder_groups=encoder_groups,
            params=params,
            seq_len=seq_len,
            device=device,
        )
        results.append(res)

    # Summary table
    print("\n" + "=" * 74)
    print("PLAIN LSTM-VAE RESULTS (SMD machine-1-1)")
    print(f"{'Seed':>4}  {'Variant':>10}  {'F1':>7}  {'AUCPR':>7}  {'ScoreSep':>9}  {'Params':>9}")
    print("-" * 74)
    for r in results:
        print(f"{r['seed']:>4}  {r['variant']:>10}  {r['f1']:7.4f}  {r['aucpr']:7.4f}  "
              f"{r['score_sep']:9.4f}  {r['n_params']:>9,}")

    # Aggregate
    f1s = np.array([r["f1"] for r in results], dtype=np.float64)
    aucprs = np.array([r["aucpr"] for r in results], dtype=np.float64)
    seps = np.array([r["score_sep"] for r in results], dtype=np.float64)
    print(
        f"\n  Aggregate: F1={f1s.mean():.4f} +/- {f1s.std():.4f}  "
        f"AUCPR={aucprs.mean():.4f} +/- {aucprs.std():.4f}  "
        f"Sep={seps.mean():.4f} +/- {seps.std():.4f}"
    )

    # Reference grouped results
    print("\n" + "=" * 74)
    print("GROUPED MODEL REFERENCE (from A/B test, 6 groups, same protocol)")
    print("-" * 74)
    print("  none (no fusion):  F1=0.4740+/-0.0089  AUCPR=0.7448+/-0.0073  Sep=9.9247+/-0.1279")
    print("  attn_mean:         F1=0.4804+/-0.0132  AUCPR=0.7423+/-0.0102  Sep=9.9145+/-0.3057")


if __name__ == "__main__":
    main()
