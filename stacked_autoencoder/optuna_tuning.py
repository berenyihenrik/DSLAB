# -*- coding: utf-8 -*-
"""Optuna hyperparameter optimization for LSTM VAE with grouped encoders."""

import copy
import functools
import torch
import numpy as np
import joblib
import optuna
from optuna.trial import TrialState
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from models import LSTMVAE_Grouped
from training import loss_function_grouped
from evaluation import window_recon_score


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
        
        for batch in train_loader:
            x_groups = [torch.as_tensor(g, dtype=torch.float32).to(device) for g in batch]
            
            optimizer.zero_grad()
            
            x_recon, mean, logvar = model(x_groups)
            loss = loss_fn(x_groups, x_recon, mean, logvar, model.group_weights, model.group_positions)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_groups = [torch.as_tensor(g, dtype=torch.float32).to(device) for g in batch]
                
                x_recon, mean, logvar = model(x_groups)
                loss = loss_fn(x_groups, x_recon, mean, logvar, model.group_weights, model.group_positions)
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


def evaluate_for_optuna(model, test_loader, device, percentile_threshold,
                        true_anomalies, sequence_length, smooth_window=7):
    """
    Evaluation function that returns F1 score for Optuna optimization.

    Uses group-weighted recon-only scoring with smoothing to match the
    full-pipeline evaluation in evaluate_lstm_grouped.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
        percentile_threshold: Percentile threshold for anomaly detection
        true_anomalies: Ground truth anomaly labels
        sequence_length: Sequence length used
        smooth_window: Rolling average window for score smoothing
    
    Returns:
        f1: F1 score at the given threshold
        anomaly_scores: Anomaly scores (numpy array)
    """
    model.eval()
    anomaly_scores = []

    with torch.no_grad():
        for batch in test_loader:
            x_groups = [torch.as_tensor(g, dtype=torch.float32).to(device) for g in batch]
            scores = window_recon_score(model, x_groups, mode="group_weighted")
            anomaly_scores.extend(scores.detach().cpu().numpy().tolist())

    # Apply rolling average smoothing (matches full pipeline)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        anomaly_scores = np.convolve(anomaly_scores, kernel, mode='same').tolist()

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


def create_optuna_objective(encoder_groups, data_groups_train, data_groups_test,
                            true_anomalies, device):
    """
    Create an Optuna objective function with the given data context.
    
    Args:
        encoder_groups: list[list[int]] — feature indices per encoder group
        data_groups_train: list of per-group training arrays
        data_groups_test: list of per-group test arrays
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
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        latent_dim = trial.suggest_categorical("latent_dim", [16, 32, 64])
        num_layers = trial.suggest_int("num_layers", 1, 2)
        
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [512, 640, 768, 896, 1024])
        
        kl_weight = trial.suggest_float("kl_weight", 0.01, 1.0, log=True)
        
        percentile_threshold = trial.suggest_int("percentile_threshold", 50, 99)
        
        seq_length = 30

        # Temporal train/val split (matches full pipeline)
        T = data_groups_train[0].shape[0]
        split_idx = int(T * 0.7)
        train_groups = [dg[:split_idx] for dg in data_groups_train]
        val_groups = [dg[max(0, split_idx - (seq_length - 1)):] for dg in data_groups_train]

        n_train = train_groups[0].shape[0] - seq_length + 1
        train_data_opt = [tuple(dg[i:i+seq_length] for dg in train_groups) for i in range(n_train)]
        n_val = val_groups[0].shape[0] - seq_length + 1
        val_data_opt = [tuple(dg[i:i+seq_length] for dg in val_groups) for i in range(n_val)]

        n_test_samples = data_groups_test[0].shape[0] - seq_length + 1
        test_sequences = [tuple(dg[i:i+seq_length] for dg in data_groups_test) for i in range(n_test_samples)]

        train_loader_opt = DataLoader(dataset=train_data_opt, batch_size=batch_size, shuffle=True)
        val_loader_opt = DataLoader(dataset=val_data_opt, batch_size=batch_size, shuffle=False)
        test_loader_opt = DataLoader(dataset=test_sequences, batch_size=batch_size, shuffle=False)
        
        model_opt = LSTMVAE_Grouped(
            encoder_groups=encoder_groups,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            sequence_length=seq_length,
            num_layers=num_layers,
            device=device,
        ).to(device)
        
        optimizer_opt = Adam(model_opt.parameters(), lr=learning_rate)
        
        # Train with fewer epochs for hyperparameter search
        try:
            train_losses, val_losses, best_val_loss = train_model_for_optuna(
                model_opt, train_loader_opt, val_loader_opt, 
                optimizer_opt, loss_function_grouped, 
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


def create_optuna_objective_with_fs(cache, train_data, test_data,
                                    true_anomalies, device,
                                    lambda_grid=(None, 5, 10, 15),
                                    num_epochs=15):
    """Create Optuna objective that jointly tunes feature selection + model params.

    Feature selection params (corr_threshold, importance_percentile,
    lag_penalty_lambda) are tuned alongside model params in the same study.
    Encoder groups are derived instantly from the precomputed cache.

    Args:
        cache: precomputed feature selection cache
        train_data: scaled training array (timesteps, n_features)
        test_data: scaled test array (timesteps, n_features)
        true_anomalies: ground truth anomaly labels
        device: torch device
        lambda_grid: precomputed lambda values (must match cache)
        num_epochs: training epochs per trial (low for fast exploration)
    """
    from feature_selection import derive_encoder_groups, split_features_by_groups

    seq_length = 30
    lambda_choices = [repr(l) for l in lambda_grid]
    lambda_map = {repr(l): l for l in lambda_grid}

    def objective(trial):
        # --- Feature selection params ---
        corr_threshold = trial.suggest_float("corr_threshold", 0.80, 0.95)
        importance_percentile = trial.suggest_int("importance_percentile", 25, 75)
        lambda_str = trial.suggest_categorical("lag_penalty_lambda", lambda_choices)
        lag_penalty_lambda = lambda_map[lambda_str]

        encoder_groups, _ = derive_encoder_groups(
            cache, corr_threshold, importance_percentile, lag_penalty_lambda
        )

        if not encoder_groups:
            return 0.0

        n_groups = len(encoder_groups)
        n_features_used = sum(len(g) for g in encoder_groups)
        trial.set_user_attr("n_encoder_groups", n_groups)
        trial.set_user_attr("n_features_used", n_features_used)

        # --- Model params ---
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        latent_dim = trial.suggest_categorical("latent_dim", [16, 32, 64])
        num_layers = trial.suggest_int("num_layers", 1, 2)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [512, 640, 768, 896, 1024])
        kl_weight = trial.suggest_float("kl_weight", 0.01, 1.0, log=True)
        percentile_threshold = trial.suggest_int("percentile_threshold", 50, 99)

        # --- Split data by derived encoder groups ---
        data_groups_train = split_features_by_groups(train_data, encoder_groups)
        data_groups_test = split_features_by_groups(test_data, encoder_groups)

        # --- Temporal train/val split (matches full pipeline) ---
        T = data_groups_train[0].shape[0]
        split_idx = int(T * 0.7)
        train_groups = [dg[:split_idx] for dg in data_groups_train]
        val_groups = [dg[max(0, split_idx - (seq_length - 1)):] for dg in data_groups_train]

        n_train = train_groups[0].shape[0] - seq_length + 1
        train_data_opt = [tuple(dg[i:i+seq_length] for dg in train_groups) for i in range(n_train)]
        n_val = val_groups[0].shape[0] - seq_length + 1
        val_data_opt = [tuple(dg[i:i+seq_length] for dg in val_groups) for i in range(n_val)]

        n_test = data_groups_test[0].shape[0] - seq_length + 1
        test_sequences = [tuple(dg[i:i+seq_length] for dg in data_groups_test) for i in range(n_test)]

        train_loader = DataLoader(train_data_opt, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data_opt, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_sequences, batch_size=batch_size, shuffle=False)

        # --- Build and train model ---
        model = LSTMVAE_Grouped(
            encoder_groups=encoder_groups,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            sequence_length=seq_length,
            num_layers=num_layers,
            device=device,
        ).to(device)

        optimizer = Adam(model.parameters(), lr=learning_rate)
        loss_fn = functools.partial(loss_function_grouped, kl_weight=kl_weight)

        try:
            _, _, best_val_loss = train_model_for_optuna(
                model, train_loader, val_loader,
                optimizer, loss_fn,
                num_epochs=num_epochs, device=device, trial=trial
            )
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0

        # --- Evaluate ---
        f1, _ = evaluate_for_optuna(
            model, test_loader, device,
            percentile_threshold, true_anomalies, seq_length
        )

        del model
        torch.cuda.empty_cache()

        return f1

    return objective


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
