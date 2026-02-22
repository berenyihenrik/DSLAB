# -*- coding: utf-8 -*-
"""LSTM VAE grouped for SMD (Server Machine Dataset).

Main entry point that orchestrates data loading, feature selection,
model training, hyperparameter optimization, and evaluation.

Refactored to work with SMD (Server Machine Dataset) for anomaly detection.
"""

import os
import functools
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler

# Import from modular components
from config import (
    SMD_DRIVE, MACHINE,
    SEQUENCE_LENGTH, INPUT_DIM, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS,
    BATCH_SIZE, NUM_EPOCHS, USE_OPTUNA, N_OPTUNA_TRIALS, DEFAULT_PARAMS, DEVICE,
    CUDNN_BENCHMARK, USE_AMP, USE_TORCH_COMPILE, DATALOADER_WORKERS, PIN_MEMORY
)
from data_loader import (
    load_smd_data, get_available_machines, preprocess_data,
    create_sequences, create_grouped_sequences
)
from models import LSTMVAE_Grouped
from training import loss_function_grouped, train_model_grouped, save_model
from optuna_tuning import (
    create_optuna_objective, create_optuna_objective_with_fs,
    run_optuna_study, evaluate_for_optuna
)
from evaluation import (
    evaluate_lstm_grouped, calculate_f1_score,
    print_evaluation_results_simple
)
from visualization import (
    visualize_optuna_study, print_optuna_summary, print_final_summary
)
from feature_selection import (
    perform_feature_selection, split_features_by_groups,
    precompute_feature_selection_cache, derive_encoder_groups
)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(seed=0):
    """Main execution function for SMD dataset."""
    set_seed(seed)
    if torch.cuda.is_available() and CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True
    
    # Print available machines
    print("Available SMD machines:")
    available_machines = get_available_machines(SMD_DRIVE)
    print(available_machines[:10], "..." if len(available_machines) > 10 else "")
    
    # Load the dataset
    metric_tensor, metric_test_tensor, true_anomalies = load_smd_data(MACHINE, SMD_DRIVE)
    
    # Convert to float32 for PyTorch compatibility
    metric_tensor = metric_tensor.astype(np.float32)
    metric_test_tensor = metric_test_tensor.astype(np.float32)
    
    # Preprocess data (handle NaN values)
    metric_tensor = preprocess_data(metric_tensor)
    metric_test_tensor = preprocess_data(metric_test_tensor)
    
    # Normalize: fit on train only, apply to both
    scaler = RobustScaler(quantile_range=(25, 75))
    scaler.fit(metric_tensor)
    metric_tensor = scaler.transform(metric_tensor).astype(np.float32)
    metric_test_tensor = scaler.transform(metric_test_tensor).astype(np.float32)
    print("Applied RobustScaler normalization (train-fitted)")
    
    sequence_length = SEQUENCE_LENGTH
    device = DEVICE
    print(f"Using device: {device}")
    print(f"Number of features: {metric_tensor.shape[1]}")
    
    # DataLoader kwargs
    num_workers = min(DATALOADER_WORKERS, max(0, (os.cpu_count() or 2) // 2))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() and PIN_MEMORY,
        "persistent_workers": num_workers > 0,
    }
    
    # Optuna hyperparameter tuning (jointly tunes feature selection + model)
    # or use defaults with fixed feature selection
    if USE_OPTUNA:
        # Precompute feature selection cache (Stage 0, 1, 2) once
        lambda_grid = (None, 5, 10, 15)
        cache = precompute_feature_selection_cache(
            metric_tensor, sequence_length, device, lambda_grid=lambda_grid
        )
        
        # Create joint objective (feature selection + model params)
        objective_fn = create_optuna_objective_with_fs(
            cache, metric_tensor, metric_test_tensor,
            true_anomalies, device,
            lambda_grid=lambda_grid, num_epochs=15
        )
        
        machine_name = MACHINE.replace('.txt', '')
        
        study = run_optuna_study(
            objective_fn, 
            n_trials=N_OPTUNA_TRIALS,
            dataset_type="SMD",
            channel=machine_name
        )
        best_params = study.best_params
        if 'kl_weight' not in best_params:
            best_params['kl_weight'] = 0.1
        
        # Derive final encoder groups from best feature selection params
        lambda_map = {repr(l): l for l in lambda_grid}
        best_lambda = lambda_map[best_params['lag_penalty_lambda']]
        encoder_groups, dropped_feature_indices = derive_encoder_groups(
            cache,
            corr_threshold=best_params['corr_threshold'],
            importance_percentile=best_params['importance_percentile'],
            lag_penalty_lambda=best_lambda
        )
        print("\nUsing optimized hyperparameters for final training...")
    else:
        # Use same cache+derive path as Optuna for consistent feature selection
        best_params = DEFAULT_PARAMS.copy()
        best_lambda = best_params.get('lag_penalty_lambda', None)
        cache = precompute_feature_selection_cache(
            metric_tensor, sequence_length, device,
            lambda_grid=(best_lambda,)
        )
        encoder_groups, dropped_feature_indices = derive_encoder_groups(
            cache,
            corr_threshold=best_params.get('corr_threshold', 0.9),
            importance_percentile=best_params.get('importance_percentile', 50),
            lag_penalty_lambda=best_lambda,
        )
        print("Using default hyperparameters...")
    
    print(f"\nEncoder groups: {len(encoder_groups)}")
    for i, group in enumerate(encoder_groups):
        print(f"  Group {i}: {len(group)} features")
    if dropped_feature_indices:
        print(f"  Dropped features: {len(dropped_feature_indices)}")
    
    # Split features by groups
    data_groups_train = split_features_by_groups(metric_tensor, encoder_groups)
    data_groups_test = split_features_by_groups(metric_test_tensor, encoder_groups)
    
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
    
    # Temporal train/val split: split raw time series before windowing
    T = metric_tensor.shape[0]
    split_idx = int(T * 0.7)
    train_raw = metric_tensor[:split_idx]
    val_raw = metric_tensor[max(0, split_idx - (sequence_length - 1)):]  # overlap for first val window
    
    train_groups_final = split_features_by_groups(train_raw, encoder_groups)
    val_groups_final = split_features_by_groups(val_raw, encoder_groups)
    
    train_data_final = create_grouped_sequences(train_groups_final, sequence_length)
    val_data_final = create_grouped_sequences(val_groups_final, sequence_length)
    test_sequences_grouped_final = create_grouped_sequences(data_groups_test, sequence_length)
    
    print(f"Temporal split: {len(train_data_final)} train / {len(val_data_final)} val sequences")
    
    train_loader_final = DataLoader(dataset=train_data_final, batch_size=final_batch_size, shuffle=True, **loader_kwargs)
    val_loader_final = DataLoader(dataset=val_data_final, batch_size=final_batch_size, shuffle=False, **loader_kwargs)
    test_loader_final = DataLoader(dataset=test_sequences_grouped_final, batch_size=final_batch_size, shuffle=False, **loader_kwargs)
    
    # Create and train final model
    model = LSTMVAE_Grouped(
        encoder_groups=encoder_groups,
        hidden_dim=final_hidden_dim,
        latent_dim=final_latent_dim,
        sequence_length=sequence_length,
        num_layers=final_num_layers,
        device=device,
    ).to(device)

    if USE_TORCH_COMPILE and hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model)
    
    optimizer = Adam(model.parameters(), lr=final_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-5)
    
    kl_weight = best_params.get('kl_weight', 0.1)
    loss_fn = functools.partial(loss_function_grouped, kl_weight=kl_weight)
    print(f"\nTraining final model with {len(encoder_groups)} encoder groups (kl_weight={kl_weight:.6f})...")
    
    train_losses, val_losses = train_model_grouped(
        model, train_loader_final, val_loader_final,
        optimizer, loss_fn, scheduler,
        num_epochs=NUM_EPOCHS, device=device, use_amp=USE_AMP
    )
    
    # Save the model
    machine_name = MACHINE.replace('.txt', '')
    model_name = f'vae_grouped_SMD_{machine_name}_optuna'
    save_model(model, model_name, INPUT_DIM, final_latent_dim, final_hidden_dim)
    print(f"\nOptimized model saved as: {model_name}.pth")
    
    # Evaluate the model using group-weighted recon scoring with smoothing
    print("\n--- Evaluating Final Model (group-weighted recon scoring) ---")
    anomalies, anomaly_scores = evaluate_lstm_grouped(
        model, test_loader_final, device,
        final_percentile_threshold, scoring="group_weighted", smooth_window=7
    )
    
    # Print evaluation results (without point-adjust for SMD)
    f1, predicted_anomalies, adjusted_true_anomalies = print_evaluation_results_simple(
        anomaly_scores, anomalies, true_anomalies, sequence_length,
        final_percentile_threshold
    )
    
    # Visualize Optuna results if optimization was run
    if USE_OPTUNA and 'study' in dir():
        print_optuna_summary(study, "SMD", machine_name)
        visualize_optuna_study(study, save_path=f'optuna_SMD_{machine_name}',
                               dataset_name="SMD", identifier=machine_name)
    
    # Print final summary (without point-adjust results)
    print_final_summary("SMD", machine_name, best_params, f1, pa_results=None)
    
    return model, best_params, f1


if __name__ == "__main__":
    model, best_params, f1 = main()
