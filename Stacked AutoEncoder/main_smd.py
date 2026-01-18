# -*- coding: utf-8 -*-
"""LSTM VAE stacked for SMD (Server Machine Dataset).

Main entry point that orchestrates data loading, feature selection,
model training, hyperparameter optimization, and evaluation.

Refactored to work with SMD (Server Machine Dataset) for anomaly detection.
"""

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Import from modular components
from config import (
    SMD_DRIVE, MACHINE,
    SEQUENCE_LENGTH, INPUT_DIM, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS,
    BATCH_SIZE, NUM_EPOCHS, USE_OPTUNA, N_OPTUNA_TRIALS, DEFAULT_PARAMS, DEVICE
)
from data_loader import (
    load_smd_data, get_available_machines, preprocess_data,
    create_sequences, create_combined_sequences
)
from models import LSTMVAE_Stacked_Weighted
from training import loss_function_weighted, train_model_weighted, save_model
from optuna_tuning import (
    create_optuna_objective, run_optuna_study, evaluate_for_optuna
)
from evaluation import (
    evaluate_lstm_weighted, calculate_f1_score,
    print_evaluation_results_simple
)
from visualization import (
    visualize_optuna_study, print_optuna_summary, print_final_summary
)
from feature_selection import perform_feature_selection, split_features_by_indices


def main():
    """Main execution function for SMD dataset."""
    
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
    
    # Create initial sequences for feature selection
    sequence_length = SEQUENCE_LENGTH
    sequences = create_sequences(metric_tensor, sequence_length)
    test_sequences = create_sequences(metric_test_tensor, sequence_length)
    
    # Initial data loaders (for reference)
    train_data, val_data = train_test_split(sequences, test_size=0.3, random_state=42)
    
    batch_size = BATCH_SIZE
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_sequences, batch_size=batch_size, shuffle=False)
    device = DEVICE
    print(f"Using device: {device}")
    
    print(f"Sequence shape: {sequences[0].shape}")
    print(f"Number of features: {metric_tensor.shape[1]}")
    
    # Perform feature selection
    selected_feature_indices, remaining_feature_indices = perform_feature_selection(
        test_sequences, true_anomalies, sequence_length, metric_tensor.shape[1]
    )
    
    # Split features into top and remaining groups
    metric_tensor_top, metric_tensor_remaining = split_features_by_indices(
        metric_tensor, selected_feature_indices, remaining_feature_indices
    )
    metric_test_tensor_top, metric_test_tensor_remaining = split_features_by_indices(
        metric_test_tensor, selected_feature_indices, remaining_feature_indices
    )
    
    # Create combined sequences
    sequences_combined = create_combined_sequences(
        metric_tensor_top, metric_tensor_remaining, sequence_length
    )
    test_sequences_combined = create_combined_sequences(
        metric_test_tensor_top, metric_test_tensor_remaining, sequence_length
    )
    
    train_data_combined, val_data_combined = train_test_split(
        sequences_combined, test_size=0.3, random_state=42
    )
    
    batch_size = 32
    train_loader_combined = DataLoader(dataset=train_data_combined, batch_size=batch_size, shuffle=True)
    val_loader_combined = DataLoader(dataset=val_data_combined, batch_size=batch_size, shuffle=False)
    test_loader_combined = DataLoader(dataset=test_sequences_combined, batch_size=batch_size, shuffle=False)
    
    print(f"Top features dimension: {len(selected_feature_indices)}")
    print(f"Remaining features dimension: {len(remaining_feature_indices)}")
    
    # Model parameters
    num_top_sensors = len(selected_feature_indices)
    num_remaining_sensors = len(remaining_feature_indices)
    input_dim = INPUT_DIM
    hidden_dim = HIDDEN_DIM
    latent_dim = LATENT_DIM
    num_layers = NUM_LAYERS
    
    # Optuna hyperparameter tuning or use defaults
    if USE_OPTUNA:
        # Create objective function with data context
        objective_fn = create_optuna_objective(
            selected_feature_indices, remaining_feature_indices,
            metric_tensor_top, metric_tensor_remaining,
            metric_test_tensor_top, metric_test_tensor_remaining,
            true_anomalies, device
        )
        
        # Extract machine name without extension for study naming
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
        print("\nUsing optimized hyperparameters for final training...")
    else:
        best_params = DEFAULT_PARAMS.copy()
        print("Using default hyperparameters...")
    
    print("\nFinal training parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Extract final hyperparameters
    final_top_weight = best_params['top_weight']
    final_remaining_weight = 1.0 - final_top_weight
    final_hidden_dim = best_params['hidden_dim']
    final_latent_dim = best_params['latent_dim']
    final_num_layers = best_params['num_layers']
    final_learning_rate = best_params['learning_rate']
    final_batch_size = best_params['batch_size']
    final_percentile_threshold = best_params['percentile_threshold']
    
    # Create final data loaders with optimized batch size
    sequences_combined_final = create_combined_sequences(
        metric_tensor_top, metric_tensor_remaining, sequence_length
    )
    test_sequences_combined_final = create_combined_sequences(
        metric_test_tensor_top, metric_test_tensor_remaining, sequence_length
    )
    
    train_data_final, val_data_final = train_test_split(
        sequences_combined_final, test_size=0.3, random_state=42
    )
    
    train_loader_final = DataLoader(dataset=train_data_final, batch_size=final_batch_size, shuffle=True)
    val_loader_final = DataLoader(dataset=val_data_final, batch_size=final_batch_size, shuffle=False)
    test_loader_final = DataLoader(dataset=test_sequences_combined_final, batch_size=final_batch_size, shuffle=False)
    
    # Create and train final model
    model = LSTMVAE_Stacked_Weighted(
        num_top_sensors=num_top_sensors,
        num_remaining_sensors=num_remaining_sensors,
        input_dim=input_dim,
        hidden_dim=final_hidden_dim,
        latent_dim=final_latent_dim,
        sequence_length=sequence_length,
        num_layers=final_num_layers,
        device=device,
        top_weight=final_top_weight,
        remaining_weight=final_remaining_weight
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=final_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    
    print(f"\nTraining final model with optimized encoder weights:")
    print(f"  top_weight: {final_top_weight:.4f}")
    print(f"  remaining_weight: {final_remaining_weight:.4f}")
    
    train_losses, val_losses = train_model_weighted(
        model, train_loader_final, val_loader_final,
        optimizer, loss_function_weighted, scheduler,
        num_epochs=NUM_EPOCHS, device=device
    )
    
    # Save the model
    machine_name = MACHINE.replace('.txt', '')
    model_name = f'vae_stacked_weighted_SMD_{machine_name}_optuna'
    save_model(model, model_name, input_dim, final_latent_dim, final_hidden_dim)
    print(f"\nOptimized model saved as: {model_name}.pth")
    
    # Evaluate the model
    print("\n--- Evaluating Final Model ---")
    final_f1, anomaly_scores = evaluate_for_optuna(
        model, test_loader_final, device,
        final_percentile_threshold, true_anomalies, sequence_length
    )
    
    threshold_value = np.percentile(anomaly_scores, final_percentile_threshold)
    anomalies = [i for i, score in enumerate(anomaly_scores) if score > threshold_value]
    
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
