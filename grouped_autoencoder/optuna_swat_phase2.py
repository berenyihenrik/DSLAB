# -*- coding: utf-8 -*-
"""Phase 2: Re-evaluate top Optuna configs with more data and multiple seeds.

Takes a Phase 1 trials JSONL file, extracts the top-N configs by F1,
and re-evaluates each with:
  - Higher subsampling ratios (moderate → full data)
  - More training epochs
  - Multiple random seeds for statistical robustness

Reports mean ± std F1 and AUCPR per config, and writes a summary JSON.

Usage (example):
    nohup python3 -u optuna_swat_phase2.py \
        --phase1-trials optuna_results/swat_phase1_v2_20260322_165548_trials.jsonl \
        --top-n 3 --seeds 42,123,7 \
        --train-ratio 0.40 --val-ratio 0.50 --test-ratio 0.50 --fs-ratio 0.15 \
        --epochs 30 \
        --study-name swat_phase2_eval \
        > /tmp/phase2.log 2>&1 &
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from functools import partial

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import (
    DEVICE,
    CUDNN_BENCHMARK,
    DATALOADER_WORKERS,
    PIN_MEMORY,
    SEQUENCE_LENGTH,
)
from data_loader import GroupedSequenceDataset
from evaluation import (
    compute_anomaly_scores_grouped,
    compute_threshold_from_baseline,
    fit_group_ecdf,
)
from feature_selection import split_features_by_groups
from models import LSTMVAE_Grouped
from optuna_tuning import train_model_for_optuna
from training import loss_function_grouped

from optuna_swat import (
    _normalize_binary_with_train_stats,
    _prepare_swat_arrays,
    _subsample_timesteps_for_fs,
    _subsample_windows,
    _validate_ratio,
    set_seed,
    MAX_ENCODER_GROUPS_DEFAULT,
)
from feature_selection import perform_feature_selection


def load_top_configs(trials_path, top_n):
    """Load top-N configs from a Phase 1 trials JSONL file, ranked by F1."""
    trials = []
    with open(trials_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t.get("state", "COMPLETE") == "COMPLETE" and t.get("value") is not None:
                trials.append(t)
    trials.sort(key=lambda t: t["value"], reverse=True)
    top = trials[:top_n]
    print(f"Loaded {len(trials)} completed trials from {trials_path}")
    print(f"Selected top {len(top)} configs:")
    for i, t in enumerate(top):
        print(f"  #{i+1}: trial {t['trial']} F1={t['value']:.4f} params={t['params']}")
    return [t["params"] for t in top], [t["trial"] for t in top]


def evaluate_single_config(
    params,
    swat_data,
    device,
    sequence_length,
    num_epochs,
    train_ratio,
    val_ratio,
    test_ratio,
    fs_ratio,
    loader_kwargs,
    fs_device,
    fs_ae_hidden_dim,
    fs_ae_num_epochs,
    fs_ae_batch_size,
    fs_ae_n_repeats,
    test_subsample_mode,
    seed,
):
    """Run a single train+evaluate pass for a fixed hyperparameter config."""
    set_seed(seed)

    train_series = swat_data["train_series"]
    val_series = swat_data["val_series"]
    test_series = swat_data["test_series"]
    true_anomalies = swat_data["true_anomalies"]
    binary_feature_indices = swat_data["binary_feature_indices"]
    continuous_indices = swat_data["continuous_indices"]

    hidden_dim = params["hidden_dim"]
    latent_dim = params["latent_dim"]
    num_layers = params["num_layers"]
    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    kl_weight = params["kl_weight"]
    percentile_threshold = params["percentile_threshold"]
    corr_threshold = params["corr_threshold"]
    importance_percentile = params["importance_percentile"]

    # --- Feature selection (deterministic given seed) ---
    encoder_groups = []
    if continuous_indices:
        continuous_train = train_series[:, continuous_indices]
        fs_train, _, _ = _subsample_timesteps_for_fs(
            continuous_train, fs_ratio, sequence_length
        )
        cont_groups_local, _ = perform_feature_selection(
            fs_train,
            fs_train.shape[1],
            sequence_length,
            fs_device,
            corr_threshold=corr_threshold,
            importance_percentile=importance_percentile,
            fs_ae_hidden_dim=fs_ae_hidden_dim,
            fs_ae_num_epochs=fs_ae_num_epochs,
            fs_ae_batch_size=fs_ae_batch_size,
            fs_ae_n_repeats=fs_ae_n_repeats,
        )
        encoder_groups.extend(
            sorted(continuous_indices[idx] for idx in group)
            for group in cont_groups_local
        )

    binary_group = sorted(binary_feature_indices)
    if binary_group:
        encoder_groups.append(binary_group)

    if not encoder_groups:
        return {"f1": 0.0, "aucpr": float("nan"), "n_groups": 0}

    # Cap encoder groups (same logic as Phase 1)
    MAX_ENCODER_GROUPS = MAX_ENCODER_GROUPS_DEFAULT
    if len(encoder_groups) > MAX_ENCODER_GROUPS:
        has_binary = binary_group and encoder_groups[-1] == binary_group
        if has_binary:
            core = encoder_groups[: MAX_ENCODER_GROUPS - 2]
            overflow = encoder_groups[MAX_ENCODER_GROUPS - 2 : -1]
            catch_all = sorted(idx for g in overflow for idx in g)
            encoder_groups = core + [catch_all] + [encoder_groups[-1]]
        else:
            core = encoder_groups[: MAX_ENCODER_GROUPS - 1]
            overflow = encoder_groups[MAX_ENCODER_GROUPS - 1 :]
            catch_all = sorted(idx for g in overflow for idx in g)
            encoder_groups = core + [catch_all]

    n_groups = len(encoder_groups)
    binary_feature_set = set(binary_feature_indices)
    binary_group_flags = [
        all(idx in binary_feature_set for idx in group) for group in encoder_groups
    ]

    # --- Build datasets ---
    data_groups_train = split_features_by_groups(train_series, encoder_groups)
    data_groups_val = split_features_by_groups(val_series, encoder_groups)
    data_groups_test = split_features_by_groups(test_series, encoder_groups)

    train_dataset = GroupedSequenceDataset(data_groups_train, sequence_length)
    val_dataset = GroupedSequenceDataset(data_groups_val, sequence_length)
    test_dataset = GroupedSequenceDataset(data_groups_test, sequence_length)
    adjusted_true_full = np.asarray(true_anomalies[sequence_length - 1 :], dtype=np.int64)

    train_data, train_used, _ = _subsample_windows(train_dataset, train_ratio, mode="stride")
    val_data, val_used, _ = _subsample_windows(val_dataset, val_ratio, mode="stride")
    test_data, test_used, _, test_indices = _subsample_windows(
        test_dataset,
        test_ratio,
        mode=test_subsample_mode,
        labels=adjusted_true_full,
        return_indices=True,
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    # --- Build & train model ---
    model = LSTMVAE_Grouped(
        encoder_groups=encoder_groups,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        sequence_length=sequence_length,
        num_layers=num_layers,
        device=device,
        binary_group_flags=binary_group_flags,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = partial(loss_function_grouped, kl_weight=kl_weight)

    try:
        train_model_for_optuna(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            num_epochs=num_epochs,
            device=device,
            trial=None,
            scheduler=None,
        )
    except Exception as exc:
        print(f"  Training failed: {exc}")
        return {"f1": 0.0, "aucpr": float("nan"), "n_groups": n_groups, "error": str(exc)}

    # --- Evaluate ---
    baseline_ecdfs = fit_group_ecdf(model, train_loader, device)
    threshold, _ = compute_threshold_from_baseline(
        model, val_loader, device, percentile_threshold, baseline_ecdfs=baseline_ecdfs,
    )
    test_scores = compute_anomaly_scores_grouped(
        model, test_loader, device, baseline_ecdfs=baseline_ecdfs,
    )

    adjusted_true = adjusted_true_full[test_indices]
    scores_arr = np.asarray(test_scores[: len(adjusted_true)], dtype=np.float64)
    preds = (scores_arr > threshold).astype(int)

    f1 = f1_score(adjusted_true, preds, zero_division=0)
    aucpr = (
        average_precision_score(adjusted_true, scores_arr)
        if len(np.unique(adjusted_true)) > 1
        else float("nan")
    )

    del model
    torch.cuda.empty_cache()

    return {
        "f1": float(f1),
        "aucpr": float(aucpr),
        "threshold": float(threshold),
        "n_groups": n_groups,
        "train_windows": int(train_used),
        "val_windows": int(val_used),
        "test_windows": int(test_used),
    }


def run_phase2(
    configs,
    phase1_trial_ids,
    swat_data,
    device,
    sequence_length,
    num_epochs,
    train_ratio,
    val_ratio,
    test_ratio,
    fs_ratio,
    loader_kwargs,
    fs_device,
    fs_ae_hidden_dim,
    fs_ae_num_epochs,
    fs_ae_batch_size,
    fs_ae_n_repeats,
    test_subsample_mode,
    seeds,
    output_dir,
    study_name,
):
    """Run Phase 2 multi-seed evaluation for each config."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"{study_name}_{timestamp}_results.json")
    log_path = os.path.join(output_dir, f"{study_name}_{timestamp}_detail.jsonl")

    all_results = []
    total_runs = len(configs) * len(seeds)
    run_idx = 0

    for cfg_idx, (params, p1_trial) in enumerate(zip(configs, phase1_trial_ids)):
        print("=" * 72)
        print(f"Config {cfg_idx+1}/{len(configs)} (Phase 1 trial {p1_trial})")
        print(f"  Params: {params}")
        print("-" * 72)

        seed_results = []
        for seed in seeds:
            run_idx += 1
            t0 = time.time()
            print(f"  Seed {seed} [{run_idx}/{total_runs}] ...", end=" ", flush=True)

            result = evaluate_single_config(
                params=params,
                swat_data=swat_data,
                device=device,
                sequence_length=sequence_length,
                num_epochs=num_epochs,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                fs_ratio=fs_ratio,
                loader_kwargs=loader_kwargs,
                fs_device=fs_device,
                fs_ae_hidden_dim=fs_ae_hidden_dim,
                fs_ae_num_epochs=fs_ae_num_epochs,
                fs_ae_batch_size=fs_ae_batch_size,
                fs_ae_n_repeats=fs_ae_n_repeats,
                test_subsample_mode=test_subsample_mode,
                seed=seed,
            )
            elapsed = time.time() - t0
            result["seed"] = seed
            result["elapsed_s"] = round(elapsed, 1)
            seed_results.append(result)

            print(f"F1={result['f1']:.4f}  AUCPR={result['aucpr']:.4f}  ({elapsed:.0f}s)")

            # Append to detail log
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "config_idx": cfg_idx,
                    "phase1_trial": p1_trial,
                    "params": params,
                    **result,
                }, sort_keys=True))
                f.write("\n")

        f1_vals = [r["f1"] for r in seed_results]
        aucpr_vals = [r["aucpr"] for r in seed_results if not np.isnan(r["aucpr"])]

        summary = {
            "config_idx": cfg_idx,
            "phase1_trial": p1_trial,
            "params": params,
            "f1_mean": float(np.mean(f1_vals)),
            "f1_std": float(np.std(f1_vals)),
            "f1_min": float(np.min(f1_vals)),
            "f1_max": float(np.max(f1_vals)),
            "aucpr_mean": float(np.mean(aucpr_vals)) if aucpr_vals else float("nan"),
            "aucpr_std": float(np.std(aucpr_vals)) if aucpr_vals else float("nan"),
            "seeds": seeds,
            "seed_results": seed_results,
        }
        all_results.append(summary)

        print(
            f"  → F1: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f} "
            f"[{summary['f1_min']:.4f}, {summary['f1_max']:.4f}]"
        )
        print(
            f"  → AUCPR: {summary['aucpr_mean']:.4f} ± {summary['aucpr_std']:.4f}"
        )

    # Final ranking
    all_results.sort(key=lambda r: r["f1_mean"], reverse=True)

    print("\n" + "=" * 72)
    print("Phase 2 — Final Ranking (by mean F1)")
    print("=" * 72)
    print(f"{'Rank':>4} {'P1 Trial':>9} {'F1 mean':>8} {'F1 std':>8} {'AUCPR':>8} {'hdim':>5} {'ldim':>5}")
    for i, r in enumerate(all_results):
        p = r["params"]
        print(
            f"{i+1:4d} {r['phase1_trial']:9d} "
            f"{r['f1_mean']:8.4f} {r['f1_std']:8.4f} "
            f"{r['aucpr_mean']:8.4f} {p['hidden_dim']:5d} {p['latent_dim']:5d}"
        )

    # Best config
    best = all_results[0]
    print(f"\nBest config: Phase 1 trial {best['phase1_trial']}")
    print(f"  F1 = {best['f1_mean']:.4f} ± {best['f1_std']:.4f}")
    print(f"  AUCPR = {best['aucpr_mean']:.4f} ± {best['aucpr_std']:.4f}")
    print(f"\nDEFAULT_PARAMS_SWAT candidate:")
    print("{")
    for key, value in best["params"].items():
        print(f"    '{key}': {value!r},")
    print("}")

    # Save
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, sort_keys=True)
    print(f"\nResults saved to: {results_path}")
    print(f"Detail log: {log_path}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: Re-evaluate top Optuna configs with more data & multiple seeds"
    )
    parser.add_argument(
        "--phase1-trials", type=str, required=True,
        help="Path to Phase 1 trials JSONL file.",
    )
    parser.add_argument("--top-n", type=int, default=3, help="Number of top configs to re-evaluate.")
    parser.add_argument(
        "--seeds", type=str, default="42,123,7",
        help="Comma-separated random seeds for multi-seed evaluation.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-ratio", type=float, default=0.40)
    parser.add_argument("--val-ratio", type=float, default=0.50)
    parser.add_argument("--test-ratio", type=float, default=0.50)
    parser.add_argument("--fs-ratio", type=float, default=0.15)
    parser.add_argument(
        "--test-subsample-mode", type=str, choices=["stride", "stratified"], default="stratified",
    )
    parser.add_argument("--fs-ae-hidden-dim", type=int, default=64)
    parser.add_argument("--fs-ae-epochs", type=int, default=15)
    parser.add_argument("--fs-ae-batch-size", type=int, default=64)
    parser.add_argument("--fs-ae-n-repeats", type=int, default=3)
    parser.add_argument(
        "--fs-device", type=str, choices=["cpu", "cuda"], default="cpu",
    )
    parser.add_argument("--study-name", type=str, default="swat_phase2_eval")
    parser.add_argument("--output-dir", type=str, default="optuna_results")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    device = DEVICE
    if torch.cuda.is_available() and CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True

    fs_device = torch.device(
        "cuda" if args.fs_device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    print(f"Phase 2 Re-evaluation")
    print(f"  Device: {device}, FS device: {fs_device}")
    print(f"  Epochs: {args.epochs}, Seeds: {seeds}")
    print(f"  Ratios: train={args.train_ratio}, val={args.val_ratio}, "
          f"test={args.test_ratio}, fs={args.fs_ratio}")
    print(f"  Test subsample mode: {args.test_subsample_mode}")
    print()

    _validate_ratio("train ratio", args.train_ratio)
    _validate_ratio("val ratio", args.val_ratio)
    _validate_ratio("test ratio", args.test_ratio)
    _validate_ratio("feature-selection ratio", args.fs_ratio)

    configs, trial_ids = load_top_configs(args.phase1_trials, args.top_n)

    swat_data = _prepare_swat_arrays()

    num_workers = min(DATALOADER_WORKERS, max(0, (os.cpu_count() or 2) // 2))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() and PIN_MEMORY,
        "persistent_workers": num_workers > 0,
    }

    run_phase2(
        configs=configs,
        phase1_trial_ids=trial_ids,
        swat_data=swat_data,
        device=device,
        sequence_length=SEQUENCE_LENGTH,
        num_epochs=args.epochs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        fs_ratio=args.fs_ratio,
        loader_kwargs=loader_kwargs,
        fs_device=fs_device,
        fs_ae_hidden_dim=args.fs_ae_hidden_dim,
        fs_ae_num_epochs=args.fs_ae_epochs,
        fs_ae_batch_size=args.fs_ae_batch_size,
        fs_ae_n_repeats=args.fs_ae_n_repeats,
        test_subsample_mode=args.test_subsample_mode,
        seeds=seeds,
        output_dir=args.output_dir,
        study_name=args.study_name,
    )


if __name__ == "__main__":
    main()
