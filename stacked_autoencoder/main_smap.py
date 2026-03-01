# -*- coding: utf-8 -*-
"""LSTM VAE grouped for NASA SMAP/MSL datasets.

Main entry point that orchestrates data loading, feature selection,
model training, hyperparameter optimization, and evaluation.

Refactored to work with NASA SMAP (Soil Moisture Active Passive satellite) and
MSL (Mars Science Laboratory rover) anomaly detection datasets.
"""

import ast
import os
from functools import partial
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Import from modular components
from config import (
    DATASET_TYPE, DRIVE, CHANNEL, LABELS_FILE,
    SEQUENCE_LENGTH, INPUT_DIM, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS,
    BATCH_SIZE, NUM_EPOCHS, USE_OPTUNA, N_OPTUNA_TRIALS, DEFAULT_PARAMS, DEVICE,
    CUDNN_BENCHMARK, USE_AMP, USE_TORCH_COMPILE, DATALOADER_WORKERS, PIN_MEMORY
)
from data_loader import (
    load_smap_msl_data, get_available_channels, preprocess_data,
    create_grouped_sequences, detect_binary_features, normalize_binary_features
)
from models import LSTMVAE_Grouped
from training import loss_function_grouped, train_model_grouped, save_model
from optuna_tuning import (
    create_optuna_objective, run_optuna_study, evaluate_for_optuna
)
from evaluation import (
    evaluate_lstm_weighted, calculate_f1_score_smap_msl,
    point_adjust_f1_score, print_evaluation_results, fit_group_ecdf,
    compute_threshold_from_baseline
)
from visualization import (
    visualize_optuna_study, print_optuna_summary, print_final_summary
)
from feature_selection import perform_feature_selection, split_features_by_groups


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    """Main execution function."""
    set_seed(42)
    if torch.cuda.is_available() and CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True

    device = DEVICE
    print(f"Using device: {device}")

    # Print available channels
    print(f"Available {DATASET_TYPE} channels:")
    available_channels = get_available_channels(DRIVE, DATASET_TYPE)
    print(available_channels[:10], "..." if len(available_channels) > 10 else "")
    
    # Load the dataset
    metric_tensor, metric_test_tensor, true_anomalies = load_smap_msl_data(CHANNEL, DRIVE, DATASET_TYPE)
    
    # Convert to float32 for PyTorch compatibility
    metric_tensor = metric_tensor.astype(np.float32)
    metric_test_tensor = metric_test_tensor.astype(np.float32)
    
    # Preprocess data (handle NaN values)
    metric_tensor = preprocess_data(metric_tensor)
    metric_test_tensor = preprocess_data(metric_test_tensor)
    
    sequence_length = SEQUENCE_LENGTH
    print(f"Number of features: {metric_tensor.shape[1]}")
    
    # Detect binary features for appropriate loss function selection
    binary_feature_indices = detect_binary_features(metric_tensor)
    n_binary = len(binary_feature_indices)
    n_total = metric_tensor.shape[1]
    print(f"Binary features: {n_binary}/{n_total}")
    
    # Use higher corr_threshold and importance_percentile when most features
    # are binary, to reduce the number of encoder groups
    if n_binary > n_total * 0.5:
        corr_thresh = 0.95
        imp_pct = 90
        print(f"Majority-binary data: using corr_threshold={corr_thresh}, "
              f"importance_percentile={imp_pct}")
    else:
        corr_thresh = 0.9
        imp_pct = 50
    
    # Perform feature selection (unsupervised, on training data only)
    # Set GROUPING_MODE to experiment with different strategies:
    #   "auto"    — full 4-stage pipeline (default)
    #   "single"  — all non-static features in one group (baseline)
    #   "clusters"— each Stage-1 cluster becomes its own encoder group
    GROUPING_MODE = "auto"
    
    if GROUPING_MODE == "single":
        from feature_selection import _drop_static_features
        kept_indices, dropped_indices, static_binary_indices = _drop_static_features(metric_tensor)
        encoder_groups = [kept_indices.tolist()]
        if len(static_binary_indices) > 0:
            encoder_groups.append(sorted(static_binary_indices.tolist()))
        dropped_feature_indices = dropped_indices.tolist()
        print(f"\nGrouping mode: SINGLE — {len(kept_indices)} dynamic features in 1 group"
              f", {len(static_binary_indices)} static binary in sentinel group")
    elif GROUPING_MODE == "clusters":
        from feature_selection import _drop_static_features, _compute_redundancy_clusters
        kept_indices, dropped_indices, static_binary_indices = _drop_static_features(metric_tensor)
        dropped_feature_indices = dropped_indices.tolist()
        kept_data = metric_tensor[:, kept_indices]
        _, _, cluster_members_dict, _ = _compute_redundancy_clusters(
            kept_data, corr_thresh, max_lag=sequence_length)
        encoder_groups = [
            sorted(kept_indices[m] for m in members)
            for members in cluster_members_dict.values()
        ]
        if len(static_binary_indices) > 0:
            encoder_groups.append(sorted(static_binary_indices.tolist()))
        print(f"\nGrouping mode: CLUSTERS — {len(encoder_groups)} groups (incl. sentinel)")
    else:
        encoder_groups, dropped_feature_indices = perform_feature_selection(
            metric_tensor, metric_tensor.shape[1], sequence_length, device,
            corr_threshold=corr_thresh, importance_percentile=imp_pct
        )
    
    # Determine which encoder groups contain only binary features
    binary_group_flags = [
        all(idx in binary_feature_indices for idx in group)
        for group in encoder_groups
    ]
    
    # Re-seed after feature selection so main model training is deterministic
    # regardless of how many epochs the importance AE used
    set_seed(42)
    
    print(f"Encoder groups: {len(encoder_groups)}")
    for i, group in enumerate(encoder_groups):
        label = "binary" if binary_group_flags[i] else "continuous"
        print(f"  Group {i}: {len(group)} features ({label})")
    if dropped_feature_indices:
        print(f"Dropped {len(dropped_feature_indices)} static features: {dropped_feature_indices}")
    
    # Normalize two-valued features to [0, 1] so BCE loss targets are valid
    if binary_feature_indices:
        metric_tensor, metric_test_tensor = normalize_binary_features(
            metric_tensor, metric_test_tensor, binary_feature_indices
        )
    
    # Split data by groups
    data_groups_train = split_features_by_groups(metric_tensor, encoder_groups)
    data_groups_test = split_features_by_groups(metric_test_tensor, encoder_groups)
    
    # Create grouped sequences
    sequences_grouped = create_grouped_sequences(data_groups_train, sequence_length)
    test_sequences_grouped = create_grouped_sequences(data_groups_test, sequence_length)
    
    num_workers = min(DATALOADER_WORKERS, max(0, (os.cpu_count() or 2) // 2))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() and PIN_MEMORY,
        "persistent_workers": num_workers > 0,
    }
    
    # Optuna hyperparameter tuning or use defaults
    if USE_OPTUNA:
        objective_fn = create_optuna_objective(
            encoder_groups, data_groups_train, data_groups_test,
            true_anomalies, device, binary_group_flags=binary_group_flags,
            use_ecdf=True
        )
        
        study = run_optuna_study(
            objective_fn, 
            n_trials=N_OPTUNA_TRIALS,
            dataset_type=DATASET_TYPE,
            channel=CHANNEL
        )
        best_params = study.best_params
        if 'kl_weight' not in best_params:
            best_params['kl_weight'] = 0.1
        print("\nUsing optimized hyperparameters for final training...")
    else:
        best_params = DEFAULT_PARAMS.copy()
        print("Using default hyperparameters...")
    
    print("\nFinal training parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Extract final hyperparameters
    final_hidden_dim = best_params['hidden_dim']
    final_latent_dim = best_params['latent_dim']
    final_num_layers = best_params['num_layers']
    final_learning_rate = best_params['learning_rate']
    final_batch_size = best_params['batch_size']
    final_percentile_threshold = best_params['percentile_threshold']
    
    # Create final data loaders with optimized batch size
    train_data_final, val_data_final = train_test_split(
        sequences_grouped, test_size=0.3, random_state=42
    )

    train_loader_final = DataLoader(dataset=train_data_final, batch_size=final_batch_size, shuffle=True, **loader_kwargs)
    val_loader_final = DataLoader(dataset=val_data_final, batch_size=final_batch_size, shuffle=False, **loader_kwargs)
    test_loader_final = DataLoader(dataset=test_sequences_grouped, batch_size=final_batch_size, shuffle=False, **loader_kwargs)
    
    # Reset seed before final model for reproducibility
    set_seed(42)

    # Create and train final model
    model = LSTMVAE_Grouped(
        encoder_groups=encoder_groups,
        hidden_dim=final_hidden_dim,
        latent_dim=final_latent_dim,
        sequence_length=sequence_length,
        num_layers=final_num_layers,
        device=device,
        binary_group_flags=binary_group_flags,
    ).to(device)

    if USE_TORCH_COMPILE and hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model)

    optimizer = Adam(model.parameters(), lr=final_learning_rate)
    
    if best_params.get('use_scheduler', False):
        scheduler = ReduceLROnPlateau(
            optimizer, 'min',
            patience=best_params.get('scheduler_patience', 5),
            factor=best_params.get('scheduler_factor', 0.1))
    else:
        scheduler = None
    
    kl_weight = best_params.get('kl_weight', 0.1)
    loss_fn = partial(loss_function_grouped, kl_weight=kl_weight)

    print(f"\nTraining final model with {len(encoder_groups)} encoder groups")
    
    train_losses, val_losses = train_model_grouped(
        model, train_loader_final, val_loader_final,
        optimizer, loss_fn, scheduler,
        num_epochs=NUM_EPOCHS, device=device, use_amp=USE_AMP
    )
    
    # Save the model
    model_name = f'vae_grouped_{DATASET_TYPE}_{CHANNEL}_optuna'
    save_model(model, model_name, INPUT_DIM, final_latent_dim, final_hidden_dim)
    print(f"\nOptimized model saved as: {model_name}.pth")
    
    # Evaluate the model (threshold derived from validation scores)
    print("\n--- Evaluating Final Model ---")
    final_f1, anomaly_scores = evaluate_for_optuna(
        model, test_loader_final, device,
        final_percentile_threshold, true_anomalies, sequence_length,
        kl_weight=kl_weight,
        baseline_loader=val_loader_final
    )
    
    threshold_value, val_scores = compute_threshold_from_baseline(
        model, val_loader_final, device, final_percentile_threshold)
    anomalies = [i for i, score in enumerate(anomaly_scores) if score > threshold_value]
    
    # Get anomaly sequences for point-adjust evaluation
    labels_df = pd.read_csv(LABELS_FILE)
    channel_labels = labels_df[labels_df['chan_id'] == CHANNEL]
    anomaly_sequences = ast.literal_eval(channel_labels['anomaly_sequences'].iloc[0])
    
    # Print evaluation results
    f1, pa_results, predicted_anomalies, adjusted_true_anomalies = print_evaluation_results(
        anomaly_scores, anomalies, true_anomalies, sequence_length,
        final_percentile_threshold, anomaly_sequences
    )
    
    # Threshold sweep — thresholds derived from validation scores
    adjusted_true = true_anomalies[sequence_length-1:]
    print("\n--- Threshold Sweep (validation-calibrated) ---")
    best_sweep_f1, best_sweep_pct = 0, 0
    for pct in range(85, 100):
        thr = np.percentile(val_scores, pct)
        preds = np.array([1 if s > thr else 0 for s in anomaly_scores[:len(adjusted_true)]])
        from sklearn.metrics import f1_score as f1_fn
        sweep_f1 = f1_fn(adjusted_true, preds, zero_division=0)
        n_det = int(preds.sum())
        if sweep_f1 > best_sweep_f1:
            best_sweep_f1 = sweep_f1
            best_sweep_pct = pct
        print(f"  Percentile {pct}: threshold={thr:.4f}, detections={n_det}, F1={sweep_f1:.4f}")
    print(f"  Best: percentile={best_sweep_pct}, F1={best_sweep_f1:.4f}")

    # Visualize Optuna results if optimization was run
    if USE_OPTUNA and 'study' in dir():
        print_optuna_summary(study, DATASET_TYPE, CHANNEL)
        visualize_optuna_study(study, save_path=f'optuna_{DATASET_TYPE}_{CHANNEL}',
                              dataset_name=DATASET_TYPE, identifier=CHANNEL)
    
    # Print final summary
    print_final_summary(DATASET_TYPE, CHANNEL, best_params, f1, pa_results)
    
    return model, best_params, f1, pa_results


if __name__ == "__main__":
    model, best_params, f1, pa_results = main()
