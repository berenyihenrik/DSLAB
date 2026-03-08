# -*- coding: utf-8 -*-
"""Optuna hyperparameter optimization for LSTM VAE with grouped encoders."""

import copy
from functools import partial
import torch
import numpy as np
import joblib
import optuna
from optuna.trial import TrialState
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from models import LSTMVAE_Grouped
from training import loss_function_grouped
from evaluation import compute_anomaly_scores_grouped, fit_group_ecdf, compute_threshold_from_baseline
from feature_selection import perform_feature_selection, split_features_by_groups


def train_model_for_optuna(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=20, device='cpu', trial=None, scheduler=None):
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
        scheduler: Optional learning rate scheduler
    
    Returns:
        train_losses, val_losses, best_loss
    """
    torch.cuda.empty_cache()
    train_losses = []
    val_losses = []

    early_stop_tolerant_count = 0
    early_stop_tolerant = 10  # Match final training patience
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        
        for batch in train_loader:
            x_groups = [torch.as_tensor(g, dtype=torch.float32).to(device) for g in batch]
            
            optimizer.zero_grad()
            
            x_recon, mean, logvar = model(x_groups)
            loss = loss_fn(x_groups, x_recon, mean, logvar, model.group_weights, model.group_positions,
                           binary_group_flags=model.binary_group_flags)
            
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
                loss = loss_fn(x_groups, x_recon, mean, logvar, model.group_weights, model.group_positions,
                               binary_group_flags=model.binary_group_flags)
                valid_loss += loss.item()

        valid_loss /= len(val_loader)
        val_losses.append(valid_loss)

        if scheduler is not None:
            scheduler.step(valid_loss)

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


def evaluate_for_optuna(model, test_loader, device, percentile_threshold, true_anomalies, sequence_length,
                        kl_weight=0.1, baseline_loader=None):
    """
    Evaluation function that returns F1 score for Optuna optimization.
    
    When ``baseline_loader`` is provided, the anomaly-score threshold is
    derived from the **validation** (normal-data) score distribution at
    ``percentile_threshold``, eliminating test-data leakage.  ECDF-
    calibrated two-sided scoring is also enabled automatically.

    When ``baseline_loader`` is ``None``, falls back to legacy behaviour
    (threshold computed from test scores — NOT recommended).
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
        percentile_threshold: Percentile threshold for anomaly detection
        true_anomalies: Ground truth anomaly labels
        sequence_length: Sequence length used
        kl_weight: Unused, kept for backward compatibility
        baseline_loader: DataLoader over training/validation data used for
                         both ECDF calibration and threshold derivation.
    
    Returns:
        f1: F1 score at the given threshold
        anomaly_scores: List of anomaly scores for each sequence
    """
    baseline_ecdfs = None
    if baseline_loader is not None:
        baseline_ecdfs = fit_group_ecdf(model, baseline_loader, device)

    anomaly_scores = compute_anomaly_scores_grouped(
        model, test_loader, device, baseline_ecdfs=baseline_ecdfs)

    adjusted_true_anomalies = true_anomalies[sequence_length-1:]

    if baseline_loader is not None:
        threshold, _ = compute_threshold_from_baseline(
            model, baseline_loader, device, percentile_threshold,
            baseline_ecdfs=baseline_ecdfs)
    else:
        threshold = np.percentile(anomaly_scores, percentile_threshold)

    anomaly_indices = [i for i, score in enumerate(anomaly_scores) if score > threshold]
    
    predicted_anomalies = np.zeros(len(adjusted_true_anomalies), dtype=int)
    for index in anomaly_indices:
        if index < len(predicted_anomalies):
            predicted_anomalies[index] = 1
    
    f1 = f1_score(adjusted_true_anomalies, predicted_anomalies, zero_division=0)
    
    return f1, anomaly_scores


def create_optuna_objective(encoder_groups, data_groups_train, data_groups_test,
                            true_anomalies, device, binary_group_flags=None,
                            use_ecdf=False,
                            raw_train_data=None, raw_test_data=None,
                            sequence_length=30):
    """
    Create an Optuna objective function with the given data context.
    
    When ``raw_train_data`` and ``raw_test_data`` are provided, feature
    selection parameters (corr_threshold, importance_percentile,
    lag_penalty_lambda) are tuned inside each trial and ``encoder_groups``,
    ``data_groups_train``, ``data_groups_test`` are ignored.
    
    Args:
        encoder_groups: list[list[int]] — feature indices per encoder group
        data_groups_train: list of per-group training arrays
        data_groups_test: list of per-group test arrays
        true_anomalies: Ground truth anomaly labels
        device: Device to use
        binary_group_flags: optional list[bool] — True for groups with binary features
        use_ecdf: If True, use ECDF-calibrated scoring (good for binary/inverted signals).
                  If False, use legacy weighted-mean scoring (default for continuous data).
        raw_train_data: Raw training array (timesteps, features). When provided,
                        feature selection is run inside each trial.
        raw_test_data: Raw test array (timesteps, features).
        sequence_length: Sequence/window length.
    
    Returns:
        Optuna objective function
    """
    def optuna_objective(trial):
        """
        Optuna objective function for hyperparameter optimization.
        """
        # Reset seed for reproducibility within each trial
        torch.manual_seed(42)
        np.random.seed(42)
        seq_length = sequence_length

        # --- Feature selection hyperparameters (when raw data provided) ---
        if raw_train_data is not None:
            corr_threshold = trial.suggest_float("corr_threshold", 0.80, 0.97)
            importance_percentile = trial.suggest_int(
                "importance_percentile", 30, 90, step=10)
            lag_penalty_lambda = trial.suggest_float("lag_penalty_lambda", 3.0, 30.0)

            trial_groups, _ = perform_feature_selection(
                raw_train_data, raw_train_data.shape[1], seq_length, device,
                corr_threshold=corr_threshold,
                importance_percentile=importance_percentile,
                lag_penalty_lambda=lag_penalty_lambda,
            )
            if len(trial_groups) == 0:
                return 0.0

            eff_encoder_groups = trial_groups
            eff_data_train = split_features_by_groups(raw_train_data, trial_groups)
            eff_data_test = split_features_by_groups(raw_test_data, trial_groups)
            eff_binary_flags = None
        else:
            eff_encoder_groups = encoder_groups
            eff_data_train = data_groups_train
            eff_data_test = data_groups_test
            eff_binary_flags = binary_group_flags

        # Hyperparameters to tune (ranges informed by manual experiments)
        hidden_dim = trial.suggest_categorical("hidden_dim", [96, 128, 192, 256])
        latent_dim = trial.suggest_int("latent_dim", 4, 16)
        num_layers = trial.suggest_int("num_layers", 1, 2)
        
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [512, 640, 768, 896, 1024])
        
        # Sharp phase transition at ~0.005; search narrow range
        kl_weight = trial.suggest_float("kl_weight", 0.003, 0.015, log=True)
        
        percentile_threshold = trial.suggest_int("percentile_threshold", 93, 99)
        
        n_samples = eff_data_train[0].shape[0] - seq_length + 1
        sequences = []
        for i in range(n_samples):
            sample = tuple(dg[i:i+seq_length] for dg in eff_data_train)
            sequences.append(sample)
        
        train_data_opt, val_data_opt = train_test_split(sequences, test_size=0.3, random_state=42)
        
        n_test_samples = eff_data_test[0].shape[0] - seq_length + 1
        test_sequences = []
        for i in range(n_test_samples):
            sample = tuple(dg[i:i+seq_length] for dg in eff_data_test)
            test_sequences.append(sample)
        
        train_loader_opt = DataLoader(dataset=train_data_opt, batch_size=batch_size, shuffle=True)
        val_loader_opt = DataLoader(dataset=val_data_opt, batch_size=batch_size, shuffle=False)
        test_loader_opt = DataLoader(dataset=test_sequences, batch_size=batch_size, shuffle=False)
        
        model_opt = LSTMVAE_Grouped(
            encoder_groups=eff_encoder_groups,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            sequence_length=seq_length,
            num_layers=num_layers,
            device=device,
            binary_group_flags=eff_binary_flags,
        ).to(device)
        
        optimizer_opt = Adam(model_opt.parameters(), lr=learning_rate)
        
        # Scheduler: let Optuna decide whether to use one
        use_scheduler = trial.suggest_categorical("use_scheduler", [True, False])
        scheduler_opt = None
        if use_scheduler:
            scheduler_patience = trial.suggest_int("scheduler_patience", 3, 10)
            scheduler_factor = trial.suggest_float("scheduler_factor", 0.1, 0.5)
            scheduler_opt = ReduceLROnPlateau(optimizer_opt, 'min',
                                              patience=scheduler_patience,
                                              factor=scheduler_factor)

        # Bind kl_weight so training and evaluation use the trial's value
        loss_fn = partial(loss_function_grouped, kl_weight=kl_weight)

        # Train with enough epochs to match final training dynamics
        try:
            train_losses, val_losses, best_val_loss = train_model_for_optuna(
                model_opt, train_loader_opt, val_loader_opt, 
                optimizer_opt, loss_fn, 
                num_epochs=50, device=device, trial=trial,
                scheduler=scheduler_opt
            )
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return 0.0
        
        # Evaluate on test set; legacy weighted-mean scoring, threshold from test scores
        f1, _ = evaluate_for_optuna(
            model_opt, test_loader_opt, device, 
            percentile_threshold, true_anomalies, seq_length,
            kl_weight=kl_weight,
            baseline_loader=None
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
