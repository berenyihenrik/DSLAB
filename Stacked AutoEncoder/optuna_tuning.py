# -*- coding: utf-8 -*-
"""Optuna hyperparameter optimization for LSTM VAE with stacked weighted encoders."""

import copy
import torch
import numpy as np
import joblib
import optuna
from optuna.trial import TrialState
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from models import LSTMVAE_Stacked_Weighted
from training import loss_function_weighted


def train_model_for_optuna(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=20, device='cpu', trial=None):
    """
    Training function optimized for Optuna trials with early stopping and pruning.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        loss_fn: Loss function
        num_epochs: Maximum number of epochs
        device: Device to train on
        trial: Optuna trial for pruning
    
    Returns:
        train_losses, val_losses, best_loss
    """
    torch.cuda.empty_cache()
    train_losses = []
    val_losses = []

    early_stop_tolerant_count = 0
    early_stop_tolerant = 5  # Shorter patience for Optuna trials
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        
        for batch_top, batch_remaining in train_loader:
            batch_top = torch.tensor(batch_top, dtype=torch.float32).to(device)
            batch_remaining = torch.tensor(batch_remaining, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            
            recon_top, recon_remaining, mean, logvar = model(batch_top, batch_remaining)
            loss = loss_fn(batch_top, batch_remaining, recon_top, recon_remaining, mean, logvar, 
                          model.top_weight, model.remaining_weight)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_top, batch_remaining in val_loader:
                batch_top = torch.tensor(batch_top, dtype=torch.float32).to(device)
                batch_remaining = torch.tensor(batch_remaining, dtype=torch.float32).to(device)
                
                recon_top, recon_remaining, mean, logvar = model(batch_top, batch_remaining)
                loss = loss_fn(batch_top, batch_remaining, recon_top, recon_remaining, mean, logvar,
                              model.top_weight, model.remaining_weight)
                valid_loss += loss.item()

        valid_loss /= len(val_loader)
        val_losses.append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_tolerant_count = 0
        else:
            early_stop_tolerant_count += 1

        # Report to Optuna for pruning
        if trial is not None:
            trial.report(valid_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if early_stop_tolerant_count >= early_stop_tolerant:
            break

    model.load_state_dict(best_model_wts)
    return train_losses, val_losses, best_loss


def evaluate_for_optuna(model, test_loader, device, percentile_threshold, true_anomalies, sequence_length):
    """
    Evaluation function that returns F1 score for Optuna optimization.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
        percentile_threshold: Percentile threshold for anomaly detection
        true_anomalies: Ground truth anomaly labels
        sequence_length: Sequence length used
    
    Returns:
        f1: F1 score at the given threshold
        anomaly_scores: List of anomaly scores for each sequence
    """
    model.eval()
    anomaly_scores = []

    with torch.no_grad():
        for batch_top, batch_remaining in test_loader:
            batch_top = torch.tensor(batch_top, dtype=torch.float32).to(device)
            batch_remaining = torch.tensor(batch_remaining, dtype=torch.float32).to(device)

            batch_scores = []
            for i in range(batch_top.shape[0]):
                sequence_top = batch_top[i, :, :].unsqueeze(0)
                sequence_remaining = batch_remaining[i, :, :].unsqueeze(0)
                
                recon_top, recon_remaining, mean, logvar = model(sequence_top, sequence_remaining)
                loss = loss_function_weighted(sequence_top, sequence_remaining, recon_top, recon_remaining, 
                                             mean, logvar, model.top_weight, model.remaining_weight)
                batch_scores.append(loss.item())
            anomaly_scores.extend(batch_scores)

    # Use the given percentile threshold
    adjusted_true_anomalies = true_anomalies[sequence_length-1:]
    
    threshold = np.percentile(anomaly_scores, percentile_threshold)
    anomaly_indices = [i for i, score in enumerate(anomaly_scores) if score > threshold]
    
    predicted_anomalies = np.zeros(len(adjusted_true_anomalies), dtype=int)
    for index in anomaly_indices:
        if index < len(predicted_anomalies):
            predicted_anomalies[index] = 1
    
    f1 = f1_score(adjusted_true_anomalies, predicted_anomalies, zero_division=0)
    
    return f1, anomaly_scores


def create_optuna_objective(selected_feature_indices, remaining_feature_indices, 
                            metric_tensor_top, metric_tensor_remaining,
                            metric_test_tensor_top, metric_test_tensor_remaining,
                            true_anomalies, device):
    """
    Create an Optuna objective function with the given data context.
    
    Args:
        selected_feature_indices: Indices of top features
        remaining_feature_indices: Indices of remaining features
        metric_tensor_top: Training data for top features
        metric_tensor_remaining: Training data for remaining features
        metric_test_tensor_top: Test data for top features
        metric_test_tensor_remaining: Test data for remaining features
        true_anomalies: Ground truth anomaly labels
        device: Device to use
    
    Returns:
        Optuna objective function
    """
    def optuna_objective(trial):
        """
        Optuna objective function for hyperparameter optimization.
        """
        # Hyperparameters to tune
        top_weight = trial.suggest_float("top_weight", 0.3, 0.9)
        remaining_weight = 1.0 - top_weight  # Ensure weights sum to 1
        
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        latent_dim = trial.suggest_categorical("latent_dim", [16, 32, 64])
        num_layers = trial.suggest_int("num_layers", 1, 2)
        
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        
        kl_weight = trial.suggest_float("kl_weight", 0.01, 1.0, log=True)
        
        percentile_threshold = trial.suggest_int("percentile_threshold", 50, 99)
        
        seq_length = 30
        sequences_top = []
        sequences_remaining = []
        for i in range(metric_tensor_top.shape[0] - seq_length + 1):
            sequences_top.append(metric_tensor_top[i:i + seq_length])
            sequences_remaining.append(metric_tensor_remaining[i:i + seq_length])
        
        sequences_combined = list(zip(sequences_top, sequences_remaining))
        train_data_opt, val_data_opt = train_test_split(sequences_combined, test_size=0.3, random_state=42)
        
        test_sequences_top = []
        test_sequences_remaining = []
        for i in range(metric_test_tensor_top.shape[0] - seq_length + 1):
            test_sequences_top.append(metric_test_tensor_top[i:i + seq_length])
            test_sequences_remaining.append(metric_test_tensor_remaining[i:i + seq_length])
        
        test_sequences_opt = list(zip(test_sequences_top, test_sequences_remaining))
        
        train_loader_opt = DataLoader(dataset=train_data_opt, batch_size=batch_size, shuffle=True)
        val_loader_opt = DataLoader(dataset=val_data_opt, batch_size=batch_size, shuffle=False)
        test_loader_opt = DataLoader(dataset=test_sequences_opt, batch_size=batch_size, shuffle=False)
        
        num_top = len(selected_feature_indices)
        num_remaining = len(remaining_feature_indices)
        
        model_opt = LSTMVAE_Stacked_Weighted(
            num_top_sensors=num_top,
            num_remaining_sensors=num_remaining,
            input_dim=1,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            sequence_length=seq_length,
            num_layers=num_layers,
            device=device,
            top_weight=top_weight,
            remaining_weight=remaining_weight
        ).to(device)
        
        optimizer_opt = Adam(model_opt.parameters(), lr=learning_rate)
        
        # Train with fewer epochs for hyperparameter search
        try:
            train_losses, val_losses, best_val_loss = train_model_for_optuna(
                model_opt, train_loader_opt, val_loader_opt, 
                optimizer_opt, loss_function_weighted, 
                num_epochs=30, device=device, trial=trial
            )
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return 0.0
        
        # Evaluate on test set with the suggested threshold
        f1, _ = evaluate_for_optuna(
            model_opt, test_loader_opt, device, 
            percentile_threshold, true_anomalies, seq_length
        )
        
        del model_opt
        torch.cuda.empty_cache()
        
        return f1
    
    return optuna_objective


def run_optuna_study(objective_fn, n_trials=50, study_name="lstm_vae_stacked_weighted", 
                     dataset_type="MSL", channel="M-1"):
    """
    Run Optuna hyperparameter optimization study.
    
    Args:
        objective_fn: Optuna objective function
        n_trials: Number of trials to run
        study_name: Name for the study (used for saving)
        dataset_type: Dataset type for filename
        channel: Channel for filename
    
    Returns:
        study: Completed Optuna study object
    """
    # Create study with TPE sampler and median pruner
    study = optuna.create_study(
        direction="maximize",  # Maximize F1 score
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    print(f"Starting Optuna study with {n_trials} trials...")
    print("=" * 60)
    
    study.optimize(
        objective_fn, 
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    print("\n" + "=" * 60)
    print("Optuna Study Complete!")
    print("=" * 60)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print(f"\nStudy statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")
    
    print(f"\nBest trial:")
    trial = study.best_trial
    print(f"  Value (F1 Score): {trial.value:.4f}")
    print(f"\n  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    study_filename = f"{study_name}_{dataset_type}_{channel}_study.pkl"
    joblib.dump(study, study_filename)
    print(f"\nStudy saved to: {study_filename}")
    
    return study
