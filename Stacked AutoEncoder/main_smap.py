# -*- coding: utf-8 -*-
"""LSTM VAE grouped for NASA SMAP/MSL datasets.

Main entry point that orchestrates data loading, feature selection,
model training, hyperparameter optimization, and evaluation.

Refactored to work with NASA SMAP (Soil Moisture Active Passive satellite) and
MSL (Mars Science Laboratory rover) anomaly detection datasets.
"""

import ast
import os
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
    create_grouped_sequences
)
from models import LSTMVAE_Grouped
from training import loss_function_grouped, train_model_grouped, save_model
from optuna_tuning import (
    create_optuna_objective, run_optuna_study, evaluate_for_optuna
)
from evaluation import (
    evaluate_lstm_weighted, calculate_f1_score_smap_msl,
    point_adjust_f1_score, print_evaluation_results
)
from visualization import (
    visualize_optuna_study, print_optuna_summary, print_final_summary
)
from feature_selection import perform_feature_selection, split_features_by_groups


def main():
    """Main execution function."""
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
    
    # Perform feature selection (unsupervised, on training data only)
    encoder_groups, dropped_feature_indices = perform_feature_selection(
        metric_tensor, metric_tensor.shape[1], sequence_length, device
    )
    
    print(f"Encoder groups: {len(encoder_groups)}")
    for i, group in enumerate(encoder_groups):
        print(f"  Group {i}: {len(group)} features")
    if dropped_feature_indices:
        print(f"Dropped {len(dropped_feature_indices)} static features: {dropped_feature_indices}")
    
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
            true_anomalies, device
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
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    
    print(f"\nTraining final model with {len(encoder_groups)} encoder groups")
    
    train_losses, val_losses = train_model_grouped(
        model, train_loader_final, val_loader_final,
        optimizer, loss_function_grouped, scheduler,
        num_epochs=NUM_EPOCHS, device=device, use_amp=USE_AMP
    )
    
    # Save the model
    model_name = f'vae_grouped_{DATASET_TYPE}_{CHANNEL}_optuna'
    save_model(model, model_name, INPUT_DIM, final_latent_dim, final_hidden_dim)
    print(f"\nOptimized model saved as: {model_name}.pth")
    
    # Evaluate the model
    print("\n--- Evaluating Final Model ---")
    final_f1, anomaly_scores = evaluate_for_optuna(
        model, test_loader_final, device,
        final_percentile_threshold, true_anomalies, sequence_length
    )
    
    threshold_value = np.percentile(anomaly_scores, final_percentile_threshold)
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
