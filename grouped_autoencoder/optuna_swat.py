# -*- coding: utf-8 -*-
"""SWaT-specific Optuna tuning for grouped LSTM-VAE.

This runner follows the same SWaT protocol as ``main_swat.py``:
1. Contiguous split on normal series.
2. Train-only normalization for two-valued features.
3. Train-only z-score standardization for continuous features.
4. Feature selection on continuous features + separate binary group.
5. ECDF-calibrated scoring with threshold derived from validation-normal data.

To keep trial runtime practical, subsampling is supported for
feature-selection rows and train/val/test window datasets.
Test-window subsampling can use label-aware stratification to preserve
anomaly prevalence under aggressive downsampling.
"""

import argparse
import json
import os
from datetime import datetime
from functools import partial
import random

import joblib
import numpy as np
import optuna
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from config import (
    SWAT_NORMAL_DATASET,
    SWAT_ATTACK_DATASET,
    SWAT_VAL_RATIO,
    SEQUENCE_LENGTH,
    DEVICE,
    CUDNN_BENCHMARK,
    DATALOADER_WORKERS,
    PIN_MEMORY,
)
from data_loader import (
    GroupedSequenceDataset,
    detect_binary_features,
    load_swat_data,
    standardize_continuous_features,
)
from evaluation import (
    compute_anomaly_scores_grouped,
    compute_threshold_from_baseline,
    fit_group_ecdf,
)
from feature_selection import perform_feature_selection, split_features_by_groups
from models import LSTMVAE_Grouped
from optuna_tuning import train_model_for_optuna
from training import loss_function_grouped


TEST_SUBSAMPLE_MODES = ("stride", "stratified")
MAX_ENCODER_GROUPS_DEFAULT = 12



def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _normalize_binary_with_train_stats(train_data, other_arrays, binary_indices):
    """Normalize binary/two-valued features to [0, 1] using train-only stats."""
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


def _parse_int_choices(name, raw_csv):
    """Parse comma-separated positive integers for categorical sweeps."""
    values = []
    for token in raw_csv.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError(f"{name} must include at least one integer value")
    values = sorted(set(values))
    if any(v <= 0 for v in values):
        raise ValueError(f"{name} must contain positive integers, got {values}")
    return values


def _uniform_index_pick(indices, n_target):
    """Pick approximately uniform index locations from a sorted index array."""
    indices = np.asarray(indices, dtype=np.int64)
    if n_target >= len(indices):
        return indices
    pick_positions = np.linspace(0, len(indices) - 1, num=n_target, dtype=np.int64)
    return indices[pick_positions]


def _stratified_window_indices(labels, n_target):
    """Select window indices while preserving the anomaly class ratio."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1-D array")

    n_total = len(labels)
    if n_target >= n_total:
        return np.arange(n_total, dtype=np.int64)

    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return _uniform_index_pick(np.arange(n_total, dtype=np.int64), n_target)

    pos_target = int(round(n_target * (len(pos_idx) / n_total)))
    if pos_target == 0 and n_target > 1:
        pos_target = 1
    pos_target = min(pos_target, len(pos_idx))
    neg_target = n_target - pos_target

    if neg_target > len(neg_idx):
        neg_target = len(neg_idx)
        pos_target = min(len(pos_idx), n_target - neg_target)

    remaining = n_target - (pos_target + neg_target)
    if remaining > 0:
        extra_pos = min(remaining, len(pos_idx) - pos_target)
        pos_target += extra_pos
        remaining -= extra_pos
    if remaining > 0:
        extra_neg = min(remaining, len(neg_idx) - neg_target)
        neg_target += extra_neg

    sampled_pos = _uniform_index_pick(pos_idx, pos_target)
    sampled_neg = _uniform_index_pick(neg_idx, neg_target)
    return np.sort(np.concatenate([sampled_pos, sampled_neg])).astype(np.int64)


def _subsample_windows(dataset, ratio, mode="stride", labels=None, return_indices=False):
    """Subsample overlap-heavy windows with deterministic stride/stratified modes."""
    _validate_ratio("subsample ratio", ratio)
    if mode not in TEST_SUBSAMPLE_MODES:
        raise ValueError(f"subsample mode must be one of {TEST_SUBSAMPLE_MODES}, got {mode!r}")

    n_total = len(dataset)
    if ratio >= 1.0:
        indices = np.arange(n_total, dtype=np.int64)
        if return_indices:
            return dataset, n_total, n_total, indices
        return dataset, n_total, n_total

    n_target = max(1, int(round(n_total * ratio)))
    if mode == "stride":
        stride = max(1, n_total // n_target)
        indices = np.arange(0, n_total, stride, dtype=np.int64)[:n_target]
    else:
        if labels is None:
            raise ValueError("labels are required when using stratified subsampling")
        labels = np.asarray(labels)
        if len(labels) != n_total:
            raise ValueError(
                f"labels length ({len(labels)}) must match dataset length ({n_total})"
            )
        indices = _stratified_window_indices(labels, n_target)

    subset = Subset(dataset, indices.tolist())
    if return_indices:
        return subset, len(indices), n_total, indices
    return subset, len(indices), n_total


def _subsample_timesteps_for_fs(data, ratio, seq_len):
    """Subsample timesteps before feature selection for faster Stage-2 AE."""
    _validate_ratio("feature-selection ratio", ratio)
    n_total = data.shape[0]
    if ratio >= 1.0:
        return data, n_total, n_total

    min_rows = seq_len + 1
    n_target = max(min_rows, int(round(n_total * ratio)))
    stride = max(1, n_total // n_target)
    sampled = data[::stride]
    if sampled.shape[0] > n_target:
        sampled = sampled[:n_target]
    if sampled.shape[0] < min_rows:
        sampled = data[:min_rows]
    return sampled, sampled.shape[0], n_total


def _prepare_swat_arrays():
    """Load SWaT and apply canonical train-only preprocessing."""
    metric_train, metric_test, true_anomalies = load_swat_data(
        SWAT_NORMAL_DATASET, SWAT_ATTACK_DATASET
    )

    split_idx = int(len(metric_train) * (1.0 - SWAT_VAL_RATIO))
    train_series = metric_train[:split_idx].astype(np.float32, copy=True)
    val_series = metric_train[split_idx:].astype(np.float32, copy=True)
    test_series = metric_test.astype(np.float32, copy=True)

    binary_feature_indices = detect_binary_features(train_series)
    all_indices = set(range(train_series.shape[1]))
    continuous_indices = sorted(all_indices - binary_feature_indices)

    if binary_feature_indices:
        train_series, [val_series, test_series] = _normalize_binary_with_train_stats(
            train_series, [val_series, test_series], binary_feature_indices
        )

    train_series, [val_series, test_series] = standardize_continuous_features(
        train_series,
        [val_series, test_series],
        continuous_indices,
    )

    print(
        f"SWaT contiguous split: train={len(train_series)}, "
        f"val={len(val_series)}, test={len(test_series)}"
    )
    print(
        f"SWaT feature split: {len(binary_feature_indices)} binary, "
        f"{len(continuous_indices)} continuous"
    )

    return {
        "train_series": train_series,
        "val_series": val_series,
        "test_series": test_series,
        "true_anomalies": true_anomalies,
        "binary_feature_indices": binary_feature_indices,
        "continuous_indices": continuous_indices,
    }


def create_swat_objective(
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
    batch_size_choices,
    fs_ae_hidden_dim,
    fs_ae_num_epochs,
    fs_ae_batch_size,
    fs_ae_n_repeats,
    test_subsample_mode,
):
    """Create Optuna objective following SWaT contiguous/evaluation protocol."""
    train_series = swat_data["train_series"]
    val_series = swat_data["val_series"]
    test_series = swat_data["test_series"]
    true_anomalies = swat_data["true_anomalies"]
    binary_feature_indices = swat_data["binary_feature_indices"]
    continuous_indices = swat_data["continuous_indices"]

    def objective(trial):
        set_seed(42)

        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 192, 256])
        latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 24, 32])
        num_layers = trial.suggest_int("num_layers", 1, 2)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", batch_size_choices)
        kl_weight = trial.suggest_float("kl_weight", 1e-3, 2e-2, log=True)
        percentile_threshold = trial.suggest_int("percentile_threshold", 85, 99)

        corr_threshold = trial.suggest_float("corr_threshold", 0.75, 0.97)
        importance_percentile = trial.suggest_int("importance_percentile", 50, 90, step=10)

        encoder_groups = []
        if continuous_indices:
            continuous_train = train_series[:, continuous_indices]
            fs_train, fs_rows_used, fs_rows_total = _subsample_timesteps_for_fs(
                continuous_train, fs_ratio, sequence_length
            )
            trial.set_user_attr("fs_rows_used", int(fs_rows_used))
            trial.set_user_attr("fs_rows_total", int(fs_rows_total))

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
                sorted(continuous_indices[idx] for idx in group) for group in cont_groups_local
            )

        binary_group = sorted(binary_feature_indices)
        if binary_group:
            encoder_groups.append(binary_group)

        if not encoder_groups:
            return 0.0

        # Cap encoder groups to prevent pathological runtime explosion.
        # Merge excess groups (least important, i.e. last continuous ones
        # before the binary group) into a single catch-all.
        MAX_ENCODER_GROUPS = MAX_ENCODER_GROUPS_DEFAULT
        if len(encoder_groups) > MAX_ENCODER_GROUPS:
            has_binary = binary_group and encoder_groups[-1] == binary_group
            if has_binary:
                core = encoder_groups[:MAX_ENCODER_GROUPS - 2]
                overflow = encoder_groups[MAX_ENCODER_GROUPS - 2:-1]
                catch_all = sorted(idx for g in overflow for idx in g)
                encoder_groups = core + [catch_all] + [encoder_groups[-1]]
            else:
                core = encoder_groups[:MAX_ENCODER_GROUPS - 1]
                overflow = encoder_groups[MAX_ENCODER_GROUPS - 1:]
                catch_all = sorted(idx for g in overflow for idx in g)
                encoder_groups = core + [catch_all]

        binary_feature_set = set(binary_feature_indices)
        binary_group_flags = [all(idx in binary_feature_set for idx in group) for group in encoder_groups]

        data_groups_train = split_features_by_groups(train_series, encoder_groups)
        data_groups_val = split_features_by_groups(val_series, encoder_groups)
        data_groups_test = split_features_by_groups(test_series, encoder_groups)

        train_dataset = GroupedSequenceDataset(data_groups_train, sequence_length)
        val_dataset = GroupedSequenceDataset(data_groups_val, sequence_length)
        test_dataset = GroupedSequenceDataset(data_groups_test, sequence_length)
        adjusted_true_full = np.asarray(true_anomalies[sequence_length - 1:], dtype=np.int64)

        train_data, train_used, train_total = _subsample_windows(
            train_dataset, train_ratio, mode="stride"
        )
        val_data, val_used, val_total = _subsample_windows(
            val_dataset, val_ratio, mode="stride"
        )
        test_data, test_used, test_total, test_indices = _subsample_windows(
            test_dataset,
            test_ratio,
            mode=test_subsample_mode,
            labels=adjusted_true_full,
            return_indices=True,
        )
        trial.set_user_attr("train_windows", int(train_used))
        trial.set_user_attr("val_windows", int(val_used))
        trial.set_user_attr("test_windows", int(test_used))
        trial.set_user_attr("test_subsample_mode", test_subsample_mode)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, **loader_kwargs)

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
                trial=trial,
                scheduler=None,
            )
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            print(f"Trial failed with error: {exc}")
            return 0.0

        baseline_ecdfs = fit_group_ecdf(model, train_loader, device)
        threshold, _ = compute_threshold_from_baseline(
            model,
            val_loader,
            device,
            percentile_threshold,
            baseline_ecdfs=baseline_ecdfs,
        )
        test_scores = compute_anomaly_scores_grouped(
            model,
            test_loader,
            device,
            baseline_ecdfs=baseline_ecdfs,
        )

        adjusted_true = adjusted_true_full[test_indices]
        scores_arr = np.asarray(test_scores[:len(adjusted_true)], dtype=np.float64)
        preds = (scores_arr > threshold).astype(int)

        f1 = f1_score(adjusted_true, preds, zero_division=0)
        aucpr = (
            average_precision_score(adjusted_true, scores_arr)
            if len(np.unique(adjusted_true)) > 1
            else float("nan")
        )
        trial.set_user_attr("aucpr", float(aucpr))
        trial.set_user_attr("threshold", float(threshold))
        trial.set_user_attr("test_anomaly_ratio_full", float(adjusted_true_full.mean()))
        trial.set_user_attr("test_anomaly_ratio_sampled", float(adjusted_true.mean()))

        del model
        torch.cuda.empty_cache()

        return f1

    return objective


def run_study(objective_fn, n_trials, study_name, output_dir):
    """Run SWaT Optuna study and persist artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(output_dir, f"{study_name}_{timestamp}_checkpoint.pkl")
    trial_log_path = os.path.join(output_dir, f"{study_name}_{timestamp}_trials.jsonl")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=8),
    )

    def _on_trial_complete(study_obj, trial):
        best_value = None
        try:
            best_value = float(study_obj.best_value)
            best_str = f"{best_value:.4f}"
        except ValueError:
            best_str = "n/a"

        trial_value = trial.value
        trial_str = "n/a" if trial_value is None else f"{trial_value:.4f}"
        print(
            f"[Trial {trial.number + 1:03d}] state={trial.state.name} "
            f"value={trial_str} best={best_str}"
        )

        joblib.dump(study_obj, checkpoint_path)
        with open(trial_log_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "trial": int(trial.number),
                        "state": trial.state.name,
                        "value": None if trial.value is None else float(trial.value),
                        "best_value": best_value,
                        "params": trial.params,
                    },
                    sort_keys=True,
                )
            )
            f.write("\n")

    print(f"Starting SWaT Optuna study '{study_name}' with {n_trials} trials...")
    print(f"Per-trial checkpoint: {checkpoint_path}")
    print(f"Per-trial log (jsonl): {trial_log_path}")
    print("=" * 72)
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
        callbacks=[_on_trial_complete],
    )
    print("=" * 72)
    print("Study complete.")

    study_path = os.path.join(output_dir, f"{study_name}_{timestamp}.pkl")
    params_path = os.path.join(output_dir, f"{study_name}_{timestamp}_best_params.json")

    joblib.dump(study, study_path)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2, sort_keys=True)

    print(f"Best trial F1: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Study saved to: {study_path}")
    print(f"Best params saved to: {params_path}")

    print("\nDEFAULT_PARAMS_SWAT candidate:")
    print("{")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value!r},")
    print("}")

    return study, study_path, params_path


def parse_args():
    """Parse CLI arguments for SWaT Optuna execution."""
    parser = argparse.ArgumentParser(description="SWaT-specific Optuna tuning")
    parser.add_argument("--trials", type=int, default=int(os.getenv("SWAT_OPTUNA_TRIALS", "40")))
    parser.add_argument(
        "--epochs-per-trial",
        type=int,
        default=int(os.getenv("SWAT_OPTUNA_EPOCHS_PER_TRIAL", "50")),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=float(os.getenv("SWAT_OPTUNA_TRAIN_RATIO", "0.15")),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=float(os.getenv("SWAT_OPTUNA_VAL_RATIO", "0.30")),
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=float(os.getenv("SWAT_OPTUNA_TEST_RATIO", "1.0")),
    )
    parser.add_argument(
        "--fs-ratio",
        type=float,
        default=float(os.getenv("SWAT_OPTUNA_FS_RATIO", "0.20")),
    )
    parser.add_argument(
        "--batch-size-choices",
        type=str,
        default=os.getenv("SWAT_OPTUNA_BATCH_SIZE_CHOICES", "256,512,768,1024"),
        help="Comma-separated batch size candidates for Optuna categorical sweep.",
    )
    parser.add_argument(
        "--fs-ae-hidden-dim",
        type=int,
        default=int(os.getenv("SWAT_OPTUNA_FS_AE_HIDDEN_DIM", "64")),
        help="Hidden dimension for Stage-2 feature-selection AE.",
    )
    parser.add_argument(
        "--fs-ae-epochs",
        type=int,
        default=int(os.getenv("SWAT_OPTUNA_FS_AE_EPOCHS", "15")),
        help="Epochs for Stage-2 feature-selection AE.",
    )
    parser.add_argument(
        "--fs-ae-batch-size",
        type=int,
        default=int(os.getenv("SWAT_OPTUNA_FS_AE_BATCH_SIZE", "64")),
        help="Batch size for Stage-2 feature-selection AE.",
    )
    parser.add_argument(
        "--fs-ae-n-repeats",
        type=int,
        default=int(os.getenv("SWAT_OPTUNA_FS_AE_N_REPEATS", "3")),
        help="Permutation repeats for Stage-2 feature importance.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=os.getenv("SWAT_OPTUNA_STUDY_NAME", "lstm_vae_grouped_swat_optuna"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getenv("SWAT_OPTUNA_OUTPUT_DIR", "grouped_autoencoder/optuna_results"),
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("SWAT_OPTUNA_SEED", "42")))
    parser.add_argument(
        "--fs-device",
        type=str,
        choices=["cpu", "cuda"],
        default=os.getenv("SWAT_OPTUNA_FS_DEVICE", "cpu"),
        help="Device for feature selection Stage-2 AE. Final model training still uses config.DEVICE.",
    )
    parser.add_argument(
        "--test-subsample-mode",
        type=str,
        choices=list(TEST_SUBSAMPLE_MODES),
        default=os.getenv("SWAT_OPTUNA_TEST_SUBSAMPLE_MODE", "stratified"),
        help="Window subsampling mode for test/evaluation set.",
    )
    return parser.parse_args()


def main():
    """Run SWaT Optuna study."""
    args = parse_args()
    set_seed(args.seed)

    device = DEVICE
    if torch.cuda.is_available() and CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True

    fs_device = torch.device("cuda" if args.fs_device == "cuda" and torch.cuda.is_available() else "cpu")
    batch_size_choices = _parse_int_choices("batch-size choices", args.batch_size_choices)

    print(f"Using device: {device}")
    print(f"Feature-selection device: {fs_device}")
    print(f"CUDA available: {torch.cuda.is_available()} (count={torch.cuda.device_count()})")
    print(
        "Trial settings: "
        f"trials={args.trials}, epochs={args.epochs_per_trial}, "
        f"train_ratio={args.train_ratio:.3f}, val_ratio={args.val_ratio:.3f}, "
        f"test_ratio={args.test_ratio:.3f}, fs_ratio={args.fs_ratio:.3f}, "
        f"test_subsample_mode={args.test_subsample_mode}, "
        f"batch_sizes={batch_size_choices}"
    )
    print(
        "Feature-selection AE settings: "
        f"hidden_dim={args.fs_ae_hidden_dim}, epochs={args.fs_ae_epochs}, "
        f"batch_size={args.fs_ae_batch_size}, n_repeats={args.fs_ae_n_repeats}"
    )

    _validate_ratio("train ratio", args.train_ratio)
    _validate_ratio("val ratio", args.val_ratio)
    _validate_ratio("test ratio", args.test_ratio)
    _validate_ratio("feature-selection ratio", args.fs_ratio)
    if args.trials <= 0:
        raise ValueError(f"trials must be > 0, got {args.trials}")
    if args.epochs_per_trial <= 0:
        raise ValueError(f"epochs-per-trial must be > 0, got {args.epochs_per_trial}")
    if args.fs_ae_hidden_dim <= 0:
        raise ValueError(f"fs-ae-hidden-dim must be > 0, got {args.fs_ae_hidden_dim}")
    if args.fs_ae_epochs <= 0:
        raise ValueError(f"fs-ae-epochs must be > 0, got {args.fs_ae_epochs}")
    if args.fs_ae_batch_size <= 0:
        raise ValueError(f"fs-ae-batch-size must be > 0, got {args.fs_ae_batch_size}")
    if args.fs_ae_n_repeats <= 0:
        raise ValueError(f"fs-ae-n-repeats must be > 0, got {args.fs_ae_n_repeats}")

    swat_data = _prepare_swat_arrays()

    num_workers = min(DATALOADER_WORKERS, max(0, (os.cpu_count() or 2) // 2))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() and PIN_MEMORY,
        "persistent_workers": num_workers > 0,
    }

    objective_fn = create_swat_objective(
        swat_data=swat_data,
        device=device,
        sequence_length=SEQUENCE_LENGTH,
        num_epochs=args.epochs_per_trial,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        fs_ratio=args.fs_ratio,
        loader_kwargs=loader_kwargs,
        fs_device=fs_device,
        batch_size_choices=batch_size_choices,
        fs_ae_hidden_dim=args.fs_ae_hidden_dim,
        fs_ae_num_epochs=args.fs_ae_epochs,
        fs_ae_batch_size=args.fs_ae_batch_size,
        fs_ae_n_repeats=args.fs_ae_n_repeats,
        test_subsample_mode=args.test_subsample_mode,
    )

    run_study(
        objective_fn=objective_fn,
        n_trials=args.trials,
        study_name=args.study_name,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
