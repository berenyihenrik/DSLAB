# -*- coding: utf-8 -*-
"""LSTM VAE grouped for SWaT (Secure Water Treatment) dataset."""

import os
from functools import partial
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import (
    SWAT_NORMAL_DATASET, SWAT_ATTACK_DATASET, SWAT_VAL_RATIO,
    SEQUENCE_LENGTH, INPUT_DIM, NUM_EPOCHS,
    USE_OPTUNA, DEFAULT_PARAMS_SWAT, DEVICE,
    CUDNN_BENCHMARK, USE_AMP, USE_TORCH_COMPILE, DATALOADER_WORKERS, PIN_MEMORY
)
from data_loader import (
    load_swat_data, detect_binary_features, GroupedSequenceDataset,
    standardize_continuous_features
)
from models import LSTMVAE_Grouped
from training import loss_function_grouped, train_model_grouped, save_model
from evaluation import (
    fit_group_ecdf, compute_threshold_from_baseline, compute_anomaly_scores_grouped,
    print_evaluation_results, extract_anomaly_sequences_from_labels
)
from visualization import print_final_summary
from feature_selection import perform_feature_selection, split_features_by_groups


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
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


def main(seed=42, num_epochs_override=None):
    """Main execution function for SWaT dataset."""
    set_seed(seed)
    if torch.cuda.is_available() and CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True

    device = DEVICE
    print(f"Using device: {device}")

    # 1) Load SWaT data
    metric_tensor, metric_test_tensor, true_anomalies = load_swat_data(
        SWAT_NORMAL_DATASET, SWAT_ATTACK_DATASET
    )

    # 2) Contiguous 80/20 split on raw normal series
    split_idx = int(len(metric_tensor) * (1.0 - SWAT_VAL_RATIO))
    train_series = metric_tensor[:split_idx].astype(np.float32, copy=True)
    val_series = metric_tensor[split_idx:].astype(np.float32, copy=True)
    test_series = metric_test_tensor.astype(np.float32, copy=True)
    print(f"SWaT contiguous split: train={len(train_series)}, val={len(val_series)}, test={len(test_series)}")

    # 3) Binary normalization + continuous standardization (train-only stats)
    binary_feature_indices = detect_binary_features(train_series)
    all_indices = set(range(train_series.shape[1]))
    continuous_indices = sorted(all_indices - binary_feature_indices)
    print(f"SWaT feature split: {len(binary_feature_indices)} binary, {len(continuous_indices)} continuous")

    if binary_feature_indices:
        train_series, [val_series, test_series] = _normalize_binary_with_train_stats(
            train_series, [val_series, test_series], binary_feature_indices
        )

    train_series, [val_series, test_series] = standardize_continuous_features(
        train_series,
        [val_series, test_series],
        continuous_indices,
    )

    # 4-5) Feature selection on continuous features only + one binary actuator group
    if USE_OPTUNA:
        print("USE_OPTUNA=True is currently deferred for SWaT. Using DEFAULT_PARAMS_SWAT.")
    best_params = DEFAULT_PARAMS_SWAT.copy()

    encoder_groups = []
    dropped_feature_indices = []
    if continuous_indices:
        continuous_train = train_series[:, continuous_indices]
        # SWaT is large enough that Stage-2 importance can OOM on GPU.
        # Run feature selection on CPU while keeping final model training on DEVICE.
        fs_device = torch.device("cpu")
        cont_groups_local, dropped_local = perform_feature_selection(
            continuous_train,
            continuous_train.shape[1],
            SEQUENCE_LENGTH,
            fs_device,
            corr_threshold=best_params.get('corr_threshold', 0.9),
            importance_percentile=best_params.get('importance_percentile', 50),
        )
        encoder_groups.extend([
            sorted(continuous_indices[idx] for idx in group)
            for group in cont_groups_local
        ])
        dropped_feature_indices = [continuous_indices[idx] for idx in dropped_local]

    binary_group = sorted(binary_feature_indices)
    if binary_group:
        encoder_groups.append(binary_group)

    if not encoder_groups:
        raise RuntimeError("No encoder groups were produced for SWaT")

    binary_feature_set = set(binary_feature_indices)
    binary_group_flags = [all(idx in binary_feature_set for idx in group) for group in encoder_groups]

    print(f"Encoder groups: {len(encoder_groups)}")
    for i, group in enumerate(encoder_groups):
        kind = "binary" if binary_group_flags[i] else "continuous"
        print(f"  Group {i}: {len(group)} features ({kind})")
    if dropped_feature_indices:
        print(f"Dropped {len(dropped_feature_indices)} continuous static features: {dropped_feature_indices}")

    # 6) Lazy grouped sequence datasets
    data_groups_train = split_features_by_groups(train_series, encoder_groups)
    data_groups_val = split_features_by_groups(val_series, encoder_groups)
    data_groups_test = split_features_by_groups(test_series, encoder_groups)

    train_dataset = GroupedSequenceDataset(data_groups_train, SEQUENCE_LENGTH)
    val_dataset = GroupedSequenceDataset(data_groups_val, SEQUENCE_LENGTH)
    test_dataset = GroupedSequenceDataset(data_groups_test, SEQUENCE_LENGTH)

    num_workers = min(DATALOADER_WORKERS, max(0, (os.cpu_count() or 2) // 2))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() and PIN_MEMORY,
        "persistent_workers": num_workers > 0,
    }

    print("\nFinal training parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    final_hidden_dim = best_params['hidden_dim']
    final_latent_dim = best_params['latent_dim']
    final_num_layers = best_params['num_layers']
    final_learning_rate = best_params['learning_rate']
    final_batch_size = best_params['batch_size']
    final_percentile_threshold = best_params['percentile_threshold']

    train_loader_final = DataLoader(train_dataset, batch_size=final_batch_size, shuffle=True, **loader_kwargs)
    val_loader_final = DataLoader(val_dataset, batch_size=final_batch_size, shuffle=False, **loader_kwargs)
    test_loader_final = DataLoader(test_dataset, batch_size=final_batch_size, shuffle=False, **loader_kwargs)

    # 7) Train grouped LSTM-VAE
    set_seed(seed)
    model = LSTMVAE_Grouped(
        encoder_groups=encoder_groups,
        hidden_dim=final_hidden_dim,
        latent_dim=final_latent_dim,
        sequence_length=SEQUENCE_LENGTH,
        num_layers=final_num_layers,
        device=device,
        binary_group_flags=binary_group_flags,
    ).to(device)

    if USE_TORCH_COMPILE and hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model)

    optimizer = Adam(model.parameters(), lr=final_learning_rate)
    if best_params.get('use_scheduler', False):
        scheduler = ReduceLROnPlateau(
            optimizer,
            'min',
            patience=best_params.get('scheduler_patience', 5),
            factor=best_params.get('scheduler_factor', 0.1),
        )
    else:
        scheduler = None

    kl_weight = best_params.get('kl_weight', 0.1)
    loss_fn = partial(loss_function_grouped, kl_weight=kl_weight)
    num_epochs = NUM_EPOCHS if num_epochs_override is None else num_epochs_override

    print(f"\nTraining SWaT model with {len(encoder_groups)} encoder groups...")
    train_model_grouped(
        model,
        train_loader_final,
        val_loader_final,
        optimizer,
        loss_fn,
        scheduler,
        num_epochs=num_epochs,
        device=device,
        use_amp=USE_AMP,
    )

    model_name = "vae_grouped_SWAT"
    save_model(model, model_name, INPUT_DIM, final_latent_dim, final_hidden_dim)
    print(f"\nModel saved as: {model_name}.pth")

    # 8) Calibrated thresholding: ECDF(train-normal) + threshold(val-normal)
    print("\n--- Evaluating Final Model (validation-calibrated) ---")
    baseline_ecdfs = fit_group_ecdf(model, train_loader_final, device)
    threshold_value, val_scores = compute_threshold_from_baseline(
        model,
        val_loader_final,
        device,
        final_percentile_threshold,
        baseline_ecdfs=baseline_ecdfs,
    )
    anomaly_scores = compute_anomaly_scores_grouped(
        model,
        test_loader_final,
        device,
        baseline_ecdfs=baseline_ecdfs,
    )
    anomalies = [i for i, score in enumerate(anomaly_scores) if score > threshold_value]

    # 9) Point-wise + point-adjust metrics
    anomaly_sequences = extract_anomaly_sequences_from_labels(true_anomalies)
    f1, pa_results, predicted_anomalies, adjusted_true_anomalies = print_evaluation_results(
        anomaly_scores,
        anomalies,
        true_anomalies,
        SEQUENCE_LENGTH,
        final_percentile_threshold,
        anomaly_sequences,
        threshold_value=threshold_value,
    )

    print("\n--- Threshold Sweep (validation-calibrated) ---")
    best_sweep_f1, best_sweep_pct = 0, 0
    for pct in range(85, 100):
        thr = np.percentile(val_scores, pct)
        preds = np.array([1 if s > thr else 0 for s in anomaly_scores[:len(adjusted_true_anomalies)]])
        from sklearn.metrics import f1_score as f1_fn
        sweep_f1 = f1_fn(adjusted_true_anomalies, preds, zero_division=0)
        n_det = int(preds.sum())
        if sweep_f1 > best_sweep_f1:
            best_sweep_f1 = sweep_f1
            best_sweep_pct = pct
        print(f"  Percentile {pct}: threshold={thr:.4f}, detections={n_det}, F1={sweep_f1:.4f}")
    print(f"  Best: percentile={best_sweep_pct}, F1={best_sweep_f1:.4f}")

    # 10) Final summary (Optuna deferred)
    print_final_summary("SWAT", "SWaT", best_params, f1, pa_results)

    return model, best_params, f1, pa_results


if __name__ == "__main__":
    model, best_params, f1, pa_results = main()
