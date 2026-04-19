# -*- coding: utf-8 -*-
"""A/B test: fusion variants with raw_kl scoring on SMD and SWaT.

Replicates the protocol of test_fusion_ab.py / test_fusion_ab_swat.py but
replaces the ECDF-based scoring pipeline with raw_kl:

    score = reconstruction_loss + kl_weight * KL_divergence

where reconstruction_loss is the sum of per-group losses (MSE for continuous,
BCE for binary) and KL is computed from the fused latent distribution.
The threshold is derived from the validation score percentile.

Results are saved to a JSON artifact for reproducibility.

Usage:
    python test_fusion_ab_rawkl.py --dataset smd
    python test_fusion_ab_rawkl.py --dataset swat
    NUM_EPOCHS_OVERRIDE=1 python test_fusion_ab_rawkl.py --dataset smd  # smoke test
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
from datetime import datetime
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from config import (
    DATALOADER_WORKERS,
    DEFAULT_PARAMS_SMD,
    DEFAULT_PARAMS_SWAT,
    DEVICE,
    MACHINE,
    NUM_EPOCHS,
    PIN_MEMORY,
    SEQUENCE_LENGTH,
    SMD_DRIVE,
    SWAT_ATTACK_DATASET,
    SWAT_NORMAL_DATASET,
    SWAT_VAL_RATIO,
    USE_AMP,
)
from data_loader import (
    GroupedSequenceDataset,
    create_grouped_sequences,
    detect_binary_features,
    load_smd_data,
    load_swat_data,
    preprocess_data,
    standardize_continuous_features,
)
from feature_selection import perform_feature_selection, split_features_by_groups
from models import LSTMVAE_Grouped
from training import loss_function_grouped, train_model_grouped

SEEDS = [0, 1, 2, 3, 4]
FUSION_TYPES = ["none", "mlp", "mlp_mean", "attn_mean"]

# SWaT subsampling ratios (match the prior ECDF runs)
SWAT_TRAIN_SUBSAMPLE = float(os.getenv("SWAT_TRAIN_SUBSAMPLE_RATIO", "0.40"))
SWAT_VAL_SUBSAMPLE = float(os.getenv("SWAT_VAL_SUBSAMPLE_RATIO", "0.50"))
SWAT_TEST_SUBSAMPLE = float(os.getenv("SWAT_TEST_SUBSAMPLE_RATIO", "0.50"))
SWAT_FS_SUBSAMPLE = float(os.getenv("SWAT_FS_SUBSAMPLE_RATIO", "0.15"))

EPOCHS_OVERRIDE_RAW = os.getenv("NUM_EPOCHS_OVERRIDE")
VERBOSE_TRAINING = os.getenv("VERBOSE_TRAINING", "0") == "1"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "rawkl_results")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader_kwargs() -> dict[str, Any]:
    num_workers = min(DATALOADER_WORKERS, max(0, (os.cpu_count() or 2) // 2))
    return {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() and PIN_MEMORY,
        "persistent_workers": num_workers > 0,
    }


def _resolve_num_epochs(dataset: str) -> int:
    if EPOCHS_OVERRIDE_RAW is not None:
        return int(EPOCHS_OVERRIDE_RAW)
    if dataset == "swat":
        return 30
    return NUM_EPOCHS


def _strided_subsample(dataset, ratio, tag, return_indices=False):
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


def _subsample_timesteps(data, ratio, seq_len):
    n_total = data.shape[0]
    if ratio >= 1.0:
        return data, n_total, n_total
    min_rows = seq_len + 1
    n_target = max(min_rows, int(round(n_total * ratio)))
    stride = max(1, n_total // n_target)
    sampled = data[::stride][:n_target]
    if sampled.shape[0] < min_rows:
        sampled = data[:min_rows]
    return sampled, sampled.shape[0], n_total


def _normalize_binary_with_train_stats(train_data, other_arrays, binary_indices):
    for idx in sorted(binary_indices):
        col_min = train_data[:, idx].min()
        col_max = train_data[:, idx].max()
        if col_max > col_min:
            train_data[:, idx] = (train_data[:, idx] - col_min) / (col_max - col_min)
            for arr in other_arrays:
                arr[:, idx] = (arr[:, idx] - col_min) / (col_max - col_min)
    return train_data, other_arrays


# ---- raw_kl scoring for grouped models ----

@torch.inference_mode()
def compute_grouped_rawkl_scores(
    model: LSTMVAE_Grouped,
    loader: DataLoader,
    device: torch.device,
    kl_weight: float,
) -> np.ndarray:
    """Compute per-window raw_kl anomaly scores for a grouped model.

    score = sum_groups(recon_loss_g) + kl_weight * KL

    Reconstruction uses sum reduction (matching vanilla raw_kl semantics).
    """
    model.eval()
    all_scores = []

    for batch in loader:
        x_groups = [torch.as_tensor(g, dtype=torch.float32).to(device, non_blocking=True) for g in batch]
        x_recon, mean, logvar = model(x_groups)
        batch_size = x_groups[0].shape[0]

        recon = torch.zeros(batch_size, device=device)
        for gi, (x_g, positions) in enumerate(zip(x_groups, model.group_positions)):
            x_recon_g = x_recon[:, :, positions]
            if model.binary_group_flags is not None and model.binary_group_flags[gi]:
                group_loss = nn.functional.binary_cross_entropy_with_logits(
                    x_recon_g, x_g, reduction="none"
                ).sum(dim=(1, 2))
            else:
                group_loss = (x_recon_g - x_g).pow(2).sum(dim=(1, 2))
            recon += group_loss

        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=1)
        scores = recon + kl_weight * kl
        all_scores.append(scores.cpu().numpy())

    return np.concatenate(all_scores)


# ---- data preparation ----

def prepare_smd(params, seq_len, device):
    metric_train, metric_test, true_anomalies = load_smd_data(MACHINE, SMD_DRIVE)
    metric_train = preprocess_data(metric_train.astype(np.float32))
    metric_test = preprocess_data(metric_test.astype(np.float32))

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

    n_total = len(seqs_train)
    n_val = max(1, int(round(n_total * 0.3)))
    n_train = n_total - n_val
    train_data, val_data = seqs_train[:n_train], seqs_train[n_train:]
    print(f"SMD contiguous split: train_windows={len(train_data)} val_windows={len(val_data)}")

    binary_group_flags = [False] * len(encoder_groups)

    return {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": seqs_test,
        "true_anomalies": true_anomalies,
        "encoder_groups": encoder_groups,
        "binary_group_flags": binary_group_flags,
        "train_subsample": 1.0,
        "val_subsample": 1.0,
        "test_subsample": 1.0,
        "metadata": {
            "n_features": metric_train.shape[1],
            "n_groups": len(encoder_groups),
            "train_windows": len(train_data),
            "val_windows": len(val_data),
            "test_windows": len(seqs_test),
        },
    }


def prepare_swat(params, seq_len, device):
    metric_train, metric_test, true_anomalies = load_swat_data(
        SWAT_NORMAL_DATASET, SWAT_ATTACK_DATASET
    )

    split_idx = int(len(metric_train) * (1.0 - SWAT_VAL_RATIO))
    train_series = metric_train[:split_idx].astype(np.float32, copy=True)
    val_series = metric_train[split_idx:].astype(np.float32, copy=True)
    test_series = metric_test.astype(np.float32, copy=True)
    print(f"SWaT contiguous split: train={len(train_series)} val={len(val_series)} test={len(test_series)}")

    binary_feature_indices = detect_binary_features(train_series)
    all_indices = set(range(train_series.shape[1]))
    continuous_indices = sorted(all_indices - binary_feature_indices)
    print(f"SWaT features: {len(binary_feature_indices)} binary, {len(continuous_indices)} continuous")

    if binary_feature_indices:
        train_series, [val_series, test_series] = _normalize_binary_with_train_stats(
            train_series, [val_series, test_series], binary_feature_indices
        )
    train_series, [val_series, test_series] = standardize_continuous_features(
        train_series, [val_series, test_series], continuous_indices
    )

    encoder_groups = []
    if continuous_indices:
        continuous_train = train_series[:, continuous_indices]
        fs_train, fs_used, fs_total = _subsample_timesteps(
            continuous_train, SWAT_FS_SUBSAMPLE, seq_len
        )
        print(f"Feature-selection rows: {fs_used}/{fs_total}")
        cont_groups_local, _ = perform_feature_selection(
            fs_train, fs_train.shape[1], seq_len, device,
            corr_threshold=params.get("corr_threshold", 0.9),
            importance_percentile=params.get("importance_percentile", 50),
        )
        encoder_groups.extend(
            sorted(continuous_indices[idx] for idx in group) for group in cont_groups_local
        )

    binary_group = sorted(binary_feature_indices)
    if binary_group:
        encoder_groups.append(binary_group)

    binary_feature_set = set(binary_feature_indices)
    binary_group_flags = [all(idx in binary_feature_set for idx in g) for g in encoder_groups]

    print(f"Encoder groups: {len(encoder_groups)}")
    for i, group in enumerate(encoder_groups):
        kind = "binary" if binary_group_flags[i] else "continuous"
        print(f"  Group {i}: {len(group)} features ({kind})")

    data_groups_train = split_features_by_groups(train_series, encoder_groups)
    data_groups_val = split_features_by_groups(val_series, encoder_groups)
    data_groups_test = split_features_by_groups(test_series, encoder_groups)

    train_dataset = GroupedSequenceDataset(data_groups_train, seq_len)
    val_dataset = GroupedSequenceDataset(data_groups_val, seq_len)
    test_dataset = GroupedSequenceDataset(data_groups_test, seq_len)

    return {
        "train_data": train_dataset,
        "val_data": val_dataset,
        "test_data": test_dataset,
        "true_anomalies": true_anomalies,
        "encoder_groups": encoder_groups,
        "binary_group_flags": binary_group_flags,
        "train_subsample": SWAT_TRAIN_SUBSAMPLE,
        "val_subsample": SWAT_VAL_SUBSAMPLE,
        "test_subsample": SWAT_TEST_SUBSAMPLE,
        "metadata": {
            "n_features": train_series.shape[1],
            "n_groups": len(encoder_groups),
            "binary_features": len(binary_feature_indices),
            "continuous_features": len(continuous_indices),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "test_windows": len(test_dataset),
        },
    }


# ---- single run ----

def run_single(
    seed: int,
    fusion_type: str,
    bundle: dict[str, Any],
    params: dict[str, Any],
    seq_len: int,
    device: torch.device,
    num_epochs: int,
) -> dict[str, Any]:
    set_seed(seed)

    train_data, train_used, train_total = _strided_subsample(
        bundle["train_data"], bundle["train_subsample"], "train"
    )
    val_data, val_used, val_total = _strided_subsample(
        bundle["val_data"], bundle["val_subsample"], "val"
    )
    test_data, test_used, test_total, test_indices = _strided_subsample(
        bundle["test_data"], bundle["test_subsample"], "test", return_indices=True
    )

    lk = _loader_kwargs()
    bs = params["batch_size"]
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, **lk)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False, **lk)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False, **lk)

    print(f"\n{'=' * 68}")
    print(
        f"seed={seed} variant={fusion_type} groups={len(bundle['encoder_groups'])} "
        f"score_mode=raw_kl "
        f"train={train_used}/{train_total} val={val_used}/{val_total} test={test_used}/{test_total}"
    )
    print(f"{'=' * 68}")

    model = LSTMVAE_Grouped(
        encoder_groups=bundle["encoder_groups"],
        hidden_dim=params["hidden_dim"],
        latent_dim=params["latent_dim"],
        sequence_length=seq_len,
        num_layers=params["num_layers"],
        device=device,
        binary_group_flags=bundle["binary_group_flags"],
        fusion_type=fusion_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=params["learning_rate"])
    kl_weight = params.get("kl_weight", 0.1)
    loss_fn = partial(loss_function_grouped, kl_weight=kl_weight)

    if VERBOSE_TRAINING:
        train_model_grouped(
            model, train_loader, val_loader, optimizer, loss_fn,
            scheduler=None, num_epochs=num_epochs, device=device, use_amp=USE_AMP,
        )
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            train_model_grouped(
                model, train_loader, val_loader, optimizer, loss_fn,
                scheduler=None, num_epochs=num_epochs, device=device, use_amp=USE_AMP,
            )

    # raw_kl scoring
    val_scores = compute_grouped_rawkl_scores(model, val_loader, device, kl_weight)
    test_scores = compute_grouped_rawkl_scores(model, test_loader, device, kl_weight)

    threshold = float(np.percentile(val_scores, params["percentile_threshold"]))

    adjusted_true_full = bundle["true_anomalies"][seq_len - 1:]
    adjusted_true = np.asarray(adjusted_true_full, dtype=np.int64)[np.asarray(test_indices, dtype=np.int64)]
    scores_arr = np.asarray(test_scores[: len(adjusted_true)], dtype=np.float64)
    preds = (scores_arr > threshold).astype(int)

    f1 = f1_score(adjusted_true, preds, zero_division=0)
    aucpr = (
        average_precision_score(adjusted_true, scores_arr)
        if len(np.unique(adjusted_true)) > 1
        else float("nan")
    )

    normal_mask = adjusted_true == 0
    anom_mask = adjusted_true == 1
    sep = float(scores_arr[anom_mask].mean() - scores_arr[normal_mask].mean()) if anom_mask.any() else float("nan")

    print(f"F1={f1:.4f} AUCPR={aucpr:.4f} score_sep={sep:.4f} threshold={threshold:.4f}")

    del model
    torch.cuda.empty_cache()

    return {
        "seed": seed,
        "variant": fusion_type,
        "f1": float(f1),
        "aucpr": float(aucpr),
        "score_sep": float(sep),
        "threshold": float(threshold),
        "n_params": n_params,
        "train_windows": int(train_used),
        "train_windows_total": int(train_total),
        "val_windows": int(val_used),
        "val_windows_total": int(val_total),
        "test_windows": int(test_used),
        "test_windows_total": int(test_total),
    }


# ---- main ----

def main():
    parser = argparse.ArgumentParser(description="Grouped fusion A/B test with raw_kl scoring")
    parser.add_argument("--dataset", choices=["smd", "swat"], required=True)
    args = parser.parse_args()

    set_seed(42)
    device = DEVICE
    seq_len = SEQUENCE_LENGTH
    num_epochs = _resolve_num_epochs(args.dataset)

    if args.dataset == "smd":
        params = DEFAULT_PARAMS_SMD.copy()
        bundle = prepare_smd(params, seq_len, device)
    else:
        params = DEFAULT_PARAMS_SWAT.copy()
        bundle = prepare_swat(params, seq_len, device)

    print(f"\nDevice: {device}")
    print(f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
    print(f"Epochs: {num_epochs}  Score mode: raw_kl  kl_weight: {params.get('kl_weight', 0.1)}")
    print(f"Fusion variants: {FUSION_TYPES}")
    print(f"Seeds: {SEEDS}\n")

    results = []
    for seed in SEEDS:
        for fusion_type in FUSION_TYPES:
            res = run_single(
                seed=seed,
                fusion_type=fusion_type,
                bundle=bundle,
                params=params,
                seq_len=seq_len,
                device=device,
                num_epochs=num_epochs,
            )
            results.append(res)

    # --- summary ---
    print("\n" + "=" * 82)
    print(f"GROUPED raw_kl RESULTS ({args.dataset.upper()})")
    print(f"{'Seed':>4}  {'Variant':>10}  {'F1':>7}  {'AUCPR':>7}  {'ScoreSep':>12}  {'Params':>9}")
    print("-" * 82)
    for r in results:
        print(
            f"{r['seed']:>4}  {r['variant']:>10}  {r['f1']:7.4f}  {r['aucpr']:7.4f}  "
            f"{r['score_sep']:12.4f}  {r['n_params']:>9,}"
        )

    print("\nAggregate (mean ± std):")
    agg = {}
    for variant in FUSION_TYPES:
        subset = [r for r in results if r["variant"] == variant]
        f1s = np.array([r["f1"] for r in subset], dtype=np.float64)
        aucprs = np.array([r["aucpr"] for r in subset], dtype=np.float64)
        seps = np.array([r["score_sep"] for r in subset], dtype=np.float64)
        agg[variant] = {
            "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
            "aucpr_mean": float(aucprs.mean()), "aucpr_std": float(aucprs.std()),
            "score_sep_mean": float(seps.mean()), "score_sep_std": float(seps.std()),
            "n_runs": len(subset),
        }
        print(
            f"  {variant:>10}  F1={f1s.mean():.4f} ± {f1s.std():.4f}  "
            f"AUCPR={aucprs.mean():.4f} ± {aucprs.std():.4f}  "
            f"Sep={seps.mean():.4f} ± {seps.std():.4f}"
        )

    base = {r["seed"]: r["f1"] for r in results if r["variant"] == "none"}
    print("\nF1 wins vs none:")
    for candidate in [v for v in FUSION_TYPES if v != "none"]:
        cand = {r["seed"]: r["f1"] for r in results if r["variant"] == candidate}
        wins = sum(cand[s] > base[s] for s in SEEDS)
        print(f"  {candidate}: {wins}/{len(SEEDS)} seeds")

    # --- save JSON artifact ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(OUTPUT_DIR, f"grouped_rawkl_{args.dataset}_{ts}_results.json")
    artifact = {
        "dataset": args.dataset,
        "score_mode": "raw_kl",
        "kl_weight": params.get("kl_weight", 0.1),
        "num_epochs": num_epochs,
        "params": {k: v for k, v in params.items() if not callable(v)},
        "metadata": bundle["metadata"],
        "per_seed": results,
        "aggregate": agg,
    }
    with open(outpath, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
