# -*- coding: utf-8 -*-
"""A/B test: fusion variants on SMD machine-1-1."""

import os
import random
import json
import numpy as np
import torch
from functools import partial
from datetime import datetime
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve

from config import (
    SMD_DRIVE, MACHINE, SEQUENCE_LENGTH, DEFAULT_PARAMS_SMD, DEVICE,
    NUM_EPOCHS, USE_AMP, DATALOADER_WORKERS, PIN_MEMORY,
)
from data_loader import load_smd_data, preprocess_data, create_grouped_sequences
from models import LSTMVAE_Grouped
from training import loss_function_grouped, train_model_grouped
from feature_selection import perform_feature_selection, split_features_by_groups
from evaluation import fit_group_ecdf, compute_anomaly_scores_grouped, compute_threshold_from_baseline


DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_FUSION_TYPES = ["none", "mlp", "mlp_mean", "attn_mean"]
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "ecdf_results")
EPOCHS_OVERRIDE_RAW = os.getenv("NUM_EPOCHS_OVERRIDE")
EXPORT_CURVE_ARTIFACTS = os.getenv("EXPORT_CURVE_ARTIFACTS", "1") == "1"


def _parse_int_list_env(name, default):
    raw = os.getenv(name)
    if not raw:
        return default
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    return values or default


def _parse_str_list_env(name, default):
    raw = os.getenv(name)
    if not raw:
        return default
    values = [token.strip() for token in raw.split(",") if token.strip()]
    return values or default


SEEDS = _parse_int_list_env("SEEDS_OVERRIDE", DEFAULT_SEEDS)
FUSION_TYPES = _parse_str_list_env("FUSION_TYPES_OVERRIDE", DEFAULT_FUSION_TYPES)


def _resolve_num_epochs():
    if EPOCHS_OVERRIDE_RAW is None:
        return NUM_EPOCHS
    override = int(EPOCHS_OVERRIDE_RAW)
    if override <= 0:
        raise ValueError(f"NUM_EPOCHS_OVERRIDE must be > 0, got {override}")
    return override


def _save_curve_artifact(output_dir, dataset, seed, variant, labels, scores, preds, threshold):
    """Persist per-run score data and PR coordinates for plotting."""
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    artifact_path = None
    if EXPORT_CURVE_ARTIFACTS:
        artifact_path = os.path.join(output_dir, f"{dataset}_{variant}_seed{seed}_curve.npz")
        np.savez_compressed(
            artifact_path,
            labels=np.asarray(labels, dtype=np.int8),
            scores=np.asarray(scores, dtype=np.float32),
            predictions=np.asarray(preds, dtype=np.int8),
            precision=np.asarray(precision, dtype=np.float32),
            recall=np.asarray(recall, dtype=np.float32),
            pr_thresholds=np.asarray(pr_thresholds, dtype=np.float32),
            threshold=np.asarray([threshold], dtype=np.float32),
        )
    if artifact_path is not None:
        artifact_path = os.path.basename(artifact_path)
    return precision, recall, artifact_path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def contiguous_train_val_split(seqs_train, val_ratio=0.3):
    """Split sequence windows into contiguous train/val partitions.

    This avoids leakage from random splitting of overlapping windows.
    """
    n_total = len(seqs_train)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError("Validation ratio leaves no samples for training.")
    return seqs_train[:n_train], seqs_train[n_train:]


def run_single(seed, fusion_type, train_data, val_data, seqs_test,
               true_anomalies, encoder_groups, params, seq_len, device,
               num_epochs, output_dir):
    """Train and evaluate one model. Returns dict of metrics."""
    set_seed(seed)

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
        num_epochs=num_epochs, device=device, use_amp=USE_AMP,
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

    precision, recall, artifact_path = _save_curve_artifact(
        output_dir=output_dir,
        dataset="smd",
        seed=seed,
        variant=tag,
        labels=adjusted_true,
        scores=scores_arr,
        preds=preds,
        threshold=threshold,
    )

    f1 = f1_score(adjusted_true, preds, zero_division=0)
    aucpr = average_precision_score(adjusted_true, scores_arr) if len(np.unique(adjusted_true)) > 1 else float("nan")

    # Score separation diagnostic
    normal_mask = adjusted_true == 0
    anom_mask = adjusted_true == 1
    sep = float(scores_arr[anom_mask].mean() - scores_arr[normal_mask].mean()) if anom_mask.any() else float("nan")

    print(f"  F1={f1:.4f}  AUCPR={aucpr:.4f}  score_sep={sep:.4f}  threshold={threshold:.4f}")

    return dict(seed=seed, variant=tag, f1=f1, aucpr=aucpr, score_sep=sep,
                threshold=threshold, n_params=n_params,
                train_windows=int(len(train_data)), val_windows=int(len(val_data)),
                test_windows=int(len(seqs_test)),
                curve_points=int(len(recall)), curve_artifact=artifact_path)


def main():
    device = DEVICE
    params = DEFAULT_PARAMS_SMD.copy()
    seq_len = SEQUENCE_LENGTH
    num_epochs = _resolve_num_epochs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_ROOT, f"grouped_ecdf_smd_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Keep feature grouping deterministic and shared across variants/seeds.
    set_seed(42)

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

    train_data, val_data = contiguous_train_val_split(seqs_train, val_ratio=0.3)
    print(f"Contiguous split: train_windows={len(train_data)} val_windows={len(val_data)}")
    print(f"Seeds: {SEEDS}")
    print(f"Fusion variants: {FUSION_TYPES}")
    print(f"Epochs: {num_epochs}")

    results = []
    for seed in SEEDS:
        for fusion_type in FUSION_TYPES:
            res = run_single(
                seed=seed,
                fusion_type=fusion_type,
                train_data=train_data,
                val_data=val_data,
                seqs_test=seqs_test,
                true_anomalies=true_anomalies,
                encoder_groups=encoder_groups,
                params=params,
                seq_len=seq_len,
                device=device,
                num_epochs=num_epochs,
                output_dir=output_dir,
            )
            results.append(res)

    # --- summary table ---
    print("\n" + "=" * 74)
    print(f"{'Seed':>4}  {'Variant':>10}  {'F1':>7}  {'AUCPR':>7}  {'ScoreSep':>9}  {'Params':>9}")
    print("-" * 74)
    for r in results:
        print(f"{r['seed']:>4}  {r['variant']:>10}  {r['f1']:7.4f}  {r['aucpr']:7.4f}  "
              f"{r['score_sep']:9.4f}  {r['n_params']:>9,}")

    # Aggregate
    aggregate = {}
    for variant in FUSION_TYPES:
        subset = [r for r in results if r["variant"] == variant]
        f1s = [r["f1"] for r in subset]
        aucprs = [r["aucpr"] for r in subset]
        seps = [r["score_sep"] for r in subset]
        aggregate[variant] = {
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "aucpr_mean": float(np.mean(aucprs)),
            "aucpr_std": float(np.std(aucprs)),
            "score_sep_mean": float(np.mean(seps)),
            "score_sep_std": float(np.std(seps)),
            "n_runs": len(subset),
        }
        print(f"\n  {variant:>10}  F1 mean={np.mean(f1s):.4f} +/- {np.std(f1s):.4f}  "
              f"AUCPR mean={np.mean(aucprs):.4f} +/- {np.std(aucprs):.4f}  "
              f"Sep mean={np.mean(seps):.4f} +/- {np.std(seps):.4f}")

    # Decision
    base_f1s = [r["f1"] for r in results if r["variant"] == "none"]
    for candidate in [v for v in FUSION_TYPES if v != "none"]:
        cand_f1s = [r["f1"] for r in results if r["variant"] == candidate]
        wins = sum(c > b for c, b in zip(cand_f1s, base_f1s))
        print(f"\n  {candidate} wins {wins}/{len(SEEDS)} seeds on F1 vs none.")

    summary_path = os.path.join(output_dir, "grouped_ecdf_smd_results.json")
    payload = {
        "dataset": "smd",
        "score_mode": "ecdf",
        "num_epochs": num_epochs,
        "params": {k: v for k, v in params.items() if not callable(v)},
        "metadata": {
            "machine": MACHINE,
            "num_features": int(metric_train.shape[1]),
            "num_groups": int(len(encoder_groups)),
            "train_windows": int(len(train_data)),
            "val_windows": int(len(val_data)),
            "test_windows": int(len(seqs_test)),
        },
        "per_seed": results,
        "aggregate": aggregate,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {summary_path}")
    if EXPORT_CURVE_ARTIFACTS:
        print(f"Curve artifacts saved under {output_dir}")


if __name__ == "__main__":
    main()
