# -*- coding: utf-8 -*-
"""Plain LSTM-VAE baseline on SWaT for comparison with grouped architecture.

Uses a single encoder group containing all 51 features (no feature selection,
no grouping). Parameter count is matched to the grouped model (~1.3M) by
scaling hidden_dim to 315.

Runtime control (env vars):
- SWAT_TRAIN_SUBSAMPLE_RATIO (default 0.40)
- SWAT_VAL_SUBSAMPLE_RATIO (default 0.50)
- SWAT_TEST_SUBSAMPLE_RATIO (default 0.50)
- NUM_EPOCHS_OVERRIDE (default 30)
"""

import os
import random
from functools import partial
import contextlib
import io

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from config import (
    SWAT_NORMAL_DATASET, SWAT_ATTACK_DATASET, SWAT_VAL_RATIO,
    SEQUENCE_LENGTH, DEFAULT_PARAMS_SWAT, DEVICE,
    USE_AMP, DATALOADER_WORKERS, PIN_MEMORY,
)
from data_loader import (
    GroupedSequenceDataset, detect_binary_features,
    load_swat_data, standardize_continuous_features,
)
from evaluation import (
    compute_anomaly_scores_grouped, compute_threshold_from_baseline, fit_group_ecdf,
)
from models import LSTMVAE_Grouped
from training import loss_function_grouped, train_model_grouped


SEEDS = [0, 1, 2, 3, 4]

# Plain model hyperparams — param-matched to grouped (1,302,275 → 1,299,159)
PLAIN_HIDDEN_DIM = 315
PLAIN_LATENT_DIM = 24
PLAIN_NUM_LAYERS = 1

# Default subsampling to match grouped A/B runs
TRAIN_SUBSAMPLE_RATIO = float(os.getenv("SWAT_TRAIN_SUBSAMPLE_RATIO", "0.40"))
VAL_SUBSAMPLE_RATIO = float(os.getenv("SWAT_VAL_SUBSAMPLE_RATIO", "0.50"))
TEST_SUBSAMPLE_RATIO = float(os.getenv("SWAT_TEST_SUBSAMPLE_RATIO", "0.50"))
EPOCHS_OVERRIDE_RAW = os.getenv("NUM_EPOCHS_OVERRIDE", "30")
VERBOSE_TRAINING = os.getenv("SWAT_VERBOSE_TRAINING", "0") == "1"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _normalize_binary_with_train_stats(train_data, other_arrays, binary_indices):
    """Normalize two-valued features to [0,1] using train-only min/max."""
    for idx in sorted(binary_indices):
        col_min = train_data[:, idx].min()
        col_max = train_data[:, idx].max()
        if col_max > col_min:
            train_data[:, idx] = (train_data[:, idx] - col_min) / (col_max - col_min)
            for arr in other_arrays:
                arr[:, idx] = (arr[:, idx] - col_min) / (col_max - col_min)
    return train_data, other_arrays


def _validate_ratio(name, ratio):
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"{name} must be in (0, 1], got {ratio}")


def _strided_subsample(dataset, ratio, tag, return_indices=False):
    _validate_ratio(f"{tag} subsample ratio", ratio)
    n_total = len(dataset)
    if ratio >= 1.0:
        indices = np.arange(n_total, dtype=np.int64)
        if return_indices:
            return dataset, n_total, n_total, indices
        return dataset, n_total, n_total

    n_target = max(1, int(round(n_total * ratio)))
    stride = max(1, n_total // n_target)
    indices = np.arange(0, n_total, stride, dtype=np.int64)[:n_target]
    subset = Subset(dataset, indices.tolist())
    if return_indices:
        return subset, len(indices), n_total, indices
    return subset, len(indices), n_total


def _resolve_num_epochs():
    if EPOCHS_OVERRIDE_RAW is None:
        from config import NUM_EPOCHS
        return NUM_EPOCHS
    override = int(EPOCHS_OVERRIDE_RAW)
    if override <= 0:
        raise ValueError(f"NUM_EPOCHS_OVERRIDE must be > 0, got {override}")
    return override


def _prepare_swat(seq_len):
    """Load and preprocess SWaT data for plain (single-group) model."""
    metric_train, metric_test, true_anomalies = load_swat_data(
        SWAT_NORMAL_DATASET, SWAT_ATTACK_DATASET
    )

    split_idx = int(len(metric_train) * (1.0 - SWAT_VAL_RATIO))
    train_series = metric_train[:split_idx].astype(np.float32, copy=True)
    val_series = metric_train[split_idx:].astype(np.float32, copy=True)
    test_series = metric_test.astype(np.float32, copy=True)
    n_features = train_series.shape[1]
    print(
        f"SWaT contiguous split: train={len(train_series)}, val={len(val_series)}, "
        f"test={len(test_series)}, features={n_features}"
    )

    binary_feature_indices = detect_binary_features(train_series)
    all_indices = set(range(n_features))
    continuous_indices = sorted(all_indices - binary_feature_indices)
    print(
        f"SWaT feature split: {len(binary_feature_indices)} binary, "
        f"{len(continuous_indices)} continuous"
    )

    if binary_feature_indices:
        train_series, [val_series, test_series] = _normalize_binary_with_train_stats(
            train_series, [val_series, test_series], binary_feature_indices
        )

    train_series, [val_series, test_series] = standardize_continuous_features(
        train_series, [val_series, test_series], continuous_indices
    )

    # Single group: all features
    encoder_groups = [list(range(n_features))]
    # Mixed binary+continuous in one group → use MSE (not BCE)
    binary_group_flags = [False]

    print(f"Plain model: 1 encoder group with all {n_features} features")

    from feature_selection import split_features_by_groups
    data_groups_train = split_features_by_groups(train_series, encoder_groups)
    data_groups_val = split_features_by_groups(val_series, encoder_groups)
    data_groups_test = split_features_by_groups(test_series, encoder_groups)

    train_dataset = GroupedSequenceDataset(data_groups_train, seq_len)
    val_dataset = GroupedSequenceDataset(data_groups_val, seq_len)
    test_dataset = GroupedSequenceDataset(data_groups_test, seq_len)

    return (train_dataset, val_dataset, test_dataset, true_anomalies,
            encoder_groups, binary_group_flags)


def run_single(seed, train_dataset, val_dataset, test_dataset,
               true_anomalies, encoder_groups, binary_group_flags, params,
               seq_len, device, num_epochs):
    set_seed(seed)

    train_data, train_used, train_total = _strided_subsample(
        train_dataset, TRAIN_SUBSAMPLE_RATIO, "train"
    )
    val_data, val_used, val_total = _strided_subsample(
        val_dataset, VAL_SUBSAMPLE_RATIO, "val"
    )
    test_data, test_used, test_total, test_indices = _strided_subsample(
        test_dataset, TEST_SUBSAMPLE_RATIO, "test", return_indices=True
    )

    num_workers = min(DATALOADER_WORKERS, max(0, (os.cpu_count() or 2) // 2))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() and PIN_MEMORY,
        "persistent_workers": num_workers > 0,
    }
    bs = params["batch_size"]
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, **loader_kwargs)

    print(f"\n{'='*68}")
    print(
        f"seed={seed} variant=plain "
        f"train_windows={train_used}/{train_total} val_windows={val_used}/{val_total} "
        f"test_windows={test_used}/{test_total}"
    )
    print(f"{'='*68}")

    model = LSTMVAE_Grouped(
        encoder_groups=encoder_groups,
        hidden_dim=PLAIN_HIDDEN_DIM,
        latent_dim=PLAIN_LATENT_DIM,
        sequence_length=seq_len,
        num_layers=PLAIN_NUM_LAYERS,
        device=device,
        binary_group_flags=binary_group_flags,
        fusion_type="none",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn = partial(loss_function_grouped, kl_weight=params.get("kl_weight", 0.1))

    if VERBOSE_TRAINING:
        train_model_grouped(
            model, train_loader, val_loader, optimizer, loss_fn, scheduler=None,
            num_epochs=num_epochs, device=device, use_amp=USE_AMP,
        )
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            train_model_grouped(
                model, train_loader, val_loader, optimizer, loss_fn, scheduler=None,
                num_epochs=num_epochs, device=device, use_amp=USE_AMP,
            )

    baseline_ecdfs = fit_group_ecdf(model, val_loader, device)
    test_scores = compute_anomaly_scores_grouped(
        model, test_loader, device, baseline_ecdfs=baseline_ecdfs
    )
    threshold, _ = compute_threshold_from_baseline(
        model, val_loader, device, params["percentile_threshold"], baseline_ecdfs=baseline_ecdfs
    )

    adjusted_true_full = true_anomalies[seq_len - 1:]
    adjusted_true = adjusted_true_full[test_indices]
    scores_arr = np.asarray(test_scores[:len(adjusted_true)], dtype=np.float64)
    preds = (scores_arr > threshold).astype(int)

    f1 = f1_score(adjusted_true, preds, zero_division=0)
    aucpr = (
        average_precision_score(adjusted_true, scores_arr)
        if len(np.unique(adjusted_true)) > 1 else float("nan")
    )

    normal_mask = adjusted_true == 0
    anom_mask = adjusted_true == 1
    sep = (
        float(scores_arr[anom_mask].mean() - scores_arr[normal_mask].mean())
        if anom_mask.any() else float("nan")
    )

    print(f"F1={f1:.4f} AUCPR={aucpr:.4f} score_sep={sep:.4f} threshold={threshold:.4f}")
    return {
        "seed": seed,
        "variant": "plain",
        "f1": f1,
        "aucpr": aucpr,
        "score_sep": sep,
        "threshold": threshold,
        "n_params": n_params,
    }


def main():
    set_seed(42)
    device = DEVICE
    params = DEFAULT_PARAMS_SWAT.copy()
    seq_len = SEQUENCE_LENGTH
    num_epochs = _resolve_num_epochs()

    print(f"Using device: {device}")
    print(f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
    print(
        f"Subsampling: train_ratio={TRAIN_SUBSAMPLE_RATIO:.3f}, "
        f"val_ratio={VAL_SUBSAMPLE_RATIO:.3f}, "
        f"test_ratio={TEST_SUBSAMPLE_RATIO:.3f}, "
        f"epochs={num_epochs}"
    )
    print(f"Plain model: hidden_dim={PLAIN_HIDDEN_DIM}, latent_dim={PLAIN_LATENT_DIM}, "
          f"num_layers={PLAIN_NUM_LAYERS}")

    (
        train_dataset, val_dataset, test_dataset, true_anomalies,
        encoder_groups, binary_group_flags
    ) = _prepare_swat(seq_len)

    results = []
    for seed in SEEDS:
        results.append(
            run_single(
                seed=seed,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                true_anomalies=true_anomalies,
                encoder_groups=encoder_groups,
                binary_group_flags=binary_group_flags,
                params=params,
                seq_len=seq_len,
                device=device,
                num_epochs=num_epochs,
            )
        )

    print("\n" + "=" * 78)
    print("PLAIN LSTM-VAE RESULTS (SWaT)")
    print(f"{'Seed':>4}  {'Variant':>10}  {'F1':>7}  {'AUCPR':>7}  {'ScoreSep':>9}  {'Params':>9}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['seed']:>4}  {r['variant']:>10}  {r['f1']:7.4f}  {r['aucpr']:7.4f}  "
            f"{r['score_sep']:9.4f}  {r['n_params']:>9,}"
        )

    f1s = np.array([r["f1"] for r in results], dtype=np.float64)
    aucprs = np.array([r["aucpr"] for r in results], dtype=np.float64)
    seps = np.array([r["score_sep"] for r in results], dtype=np.float64)
    print(
        f"\n  Aggregate: F1={f1s.mean():.4f} +/- {f1s.std():.4f}  "
        f"AUCPR={aucprs.mean():.4f} +/- {aucprs.std():.4f}  "
        f"Sep={seps.mean():.4f} +/- {seps.std():.4f}"
    )

    print("\n" + "=" * 78)
    print("GROUPED MODEL REFERENCE (from A/B test, 15 groups, same subsampling)")
    print("-" * 78)
    print("  none (no fusion):  F1=0.4268+/-0.078  AUCPR=0.4044+/-0.180  Sep=2.93+/-1.15  params=1,302,275")
    print("  attn_mean:         F1=0.5660+/-0.037  AUCPR=0.6544+/-0.121  Sep=5.36+/-2.14  params=1,307,507")
    print("  Phase-2 (12 grp):  F1=0.628+/-0.030   AUCPR=0.602+/-0.109")

    if (
        TRAIN_SUBSAMPLE_RATIO < 1.0
        or VAL_SUBSAMPLE_RATIO < 1.0
        or TEST_SUBSAMPLE_RATIO < 1.0
    ):
        print("\nNote: Subsampling enabled; absolute metrics may differ from full-data runs.")


if __name__ == "__main__":
    main()
