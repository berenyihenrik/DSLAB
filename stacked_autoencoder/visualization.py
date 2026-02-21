# -*- coding: utf-8 -*-
"""Visualization functions for Optuna study results."""

import numpy as np
import matplotlib.pyplot as plt
from optuna.trial import TrialState


def visualize_optuna_study(study, save_path=None, dataset_name="MSL", identifier="M-1"):
    """
    Generate visualizations for Optuna study results.
    
    Args:
        study: Optuna study object
        save_path: Optional path prefix to save figures
        dataset_name: Dataset name for titles (e.g., "MSL", "SMAP", "SMD")
        identifier: Channel or machine identifier for titles
    """
    # Create figure for optimization history
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    trial_values = [t.value for t in study.trials if t.value is not None]
    trial_numbers = [t.number for t in study.trials if t.value is not None]
    
    ax1.plot(trial_numbers, trial_values, 'bo-', alpha=0.6, label='Trial F1 Score')
    
    # Add best value line
    best_values = []
    current_best = 0
    for val in trial_values:
        if val > current_best:
            current_best = val
        best_values.append(current_best)
    ax1.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best F1 Score')
    
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('F1 Score')
    ax1.set_title(f'Optuna Optimization History - {dataset_name} {identifier}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(f'{save_path}_optimization_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create figure for KL weight analysis
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Extract parameter values from completed trials
    completed_trials = [t for t in study.trials if t.value is not None]
    
    if len(completed_trials) > 0:
        kl_weights = [t.params.get('kl_weight', None) for t in completed_trials]
        f1_scores = [t.value for t in completed_trials]
        
        # Filter out None values
        valid_data = [(kw, f1) for kw, f1 in zip(kl_weights, f1_scores) if kw is not None]
        
        if valid_data:
            kl_weights, f1_scores = zip(*valid_data)
            
            scatter = ax2.scatter(kl_weights, f1_scores, c=range(len(kl_weights)), 
                                  cmap='viridis', alpha=0.7, s=50)
            
            # Highlight best trial
            best_idx = np.argmax(f1_scores)
            ax2.scatter([kl_weights[best_idx]], [f1_scores[best_idx]], 
                       c='red', s=200, marker='*', edgecolors='black', 
                       linewidths=2, label=f'Best (kl_weight={kl_weights[best_idx]:.3f})')
            
            ax2.set_xlabel('KL Weight')
            ax2.set_ylabel('F1 Score')
            ax2.set_title(f'KL Weight vs Model Performance - {dataset_name} {identifier}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.colorbar(scatter, ax=ax2, label='Trial Number')
    
    if save_path:
        plt.savefig(f'{save_path}_weight_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Parameter distribution plot
    fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    param_names = ['kl_weight', 'hidden_dim', 'latent_dim', 'learning_rate', 
                   'batch_size', 'percentile_threshold']
    
    for idx, param_name in enumerate(param_names):
        if idx < len(axes):
            param_values = [t.params.get(param_name, None) for t in completed_trials]
            trial_f1s = [t.value for t in completed_trials]
            
            valid_data = [(p, f1) for p, f1 in zip(param_values, trial_f1s) if p is not None]
            
            if valid_data:
                param_vals, f1_vals = zip(*valid_data)
                axes[idx].scatter(param_vals, f1_vals, alpha=0.6)
                axes[idx].set_xlabel(param_name)
                axes[idx].set_ylabel('F1 Score')
                axes[idx].set_title(f'{param_name} vs F1 Score')
                axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(f'Parameter Analysis - {dataset_name} {identifier}')
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}_parameter_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_optuna_summary(study, dataset_name="MSL", identifier="M-1"):
    """
    Print a detailed summary of the Optuna study results.
    
    Args:
        study: Optuna study object
        dataset_name: Dataset name for display (e.g., "MSL", "SMAP", "SMD")
        identifier: Channel or machine identifier for display
    """
    print("\n" + "=" * 70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    print(f"\nDataset: {dataset_name}, Identifier: {identifier}")
    print(f"Total trials: {len(study.trials)}")
    
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    
    print(f"Completed trials: {len(completed)}")
    print(f"Pruned trials: {len(pruned)}")
    
    if completed:
        f1_scores = [t.value for t in completed]
        print(f"\nF1 Score Statistics:")
        print(f"  Best:   {max(f1_scores):.4f}")
        print(f"  Mean:   {np.mean(f1_scores):.4f}")
        print(f"  Std:    {np.std(f1_scores):.4f}")
        print(f"  Median: {np.median(f1_scores):.4f}")
        
        print(f"\nBest Hyperparameters:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        # Analyze KL weight importance
        kl_weights = [t.params.get('kl_weight', 0.1) for t in completed]
        weight_corr = np.corrcoef(kl_weights, f1_scores)[0, 1]
        print(f"\nKL Weight Analysis:")
        print(f"  Correlation with F1 Score: {weight_corr:.4f}")
        print(f"  Optimal kl_weight: {study.best_params.get('kl_weight', 'N/A')}")
    
    print("=" * 70)


def print_final_summary(dataset_name, identifier, best_params, f1, pa_results=None):
    """
    Print final model performance summary.
    
    Args:
        dataset_name: Dataset name (e.g., "MSL", "SMAP", "SMD")
        identifier: Channel or machine identifier
        best_params: Best hyperparameters dictionary
        f1: Point-wise F1 score
        pa_results: Point-adjust evaluation results (optional, for SMAP/MSL)
    """
    print("\n" + "=" * 70)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"\nDataset: {dataset_name}, Identifier: {identifier}")
    print(f"\nOptimized Hyperparameters:")
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print(f"\nPerformance Metrics:")
    print(f"  Point-wise F1 Score: {f1:.4f}")
    if pa_results is not None:
        print(f"  Point-Adjust F1 Score: {pa_results['f1']:.4f}")
        print(f"  Detected Anomaly Segments: {pa_results['true_positives']} / {pa_results['total_segments']}")
    print("=" * 70)
