# -*- coding: utf-8 -*-
"""Feature selection functions using RandomForest for importance ranking."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def perform_feature_selection(test_sequences, true_anomalies, sequence_length, n_features):
    """
    Perform feature selection using RandomForestClassifier.
    
    Args:
        test_sequences: List of test sequences
        true_anomalies: Ground truth anomaly labels
        sequence_length: Sequence length
        n_features: Number of features
    
    Returns:
        selected_feature_indices: Indices of top features
        remaining_feature_indices: Indices of remaining features
    """
    # Adjust true_anomalies to match test_sequences length (due to windowing)
    adjusted_true_anomalies = true_anomalies[sequence_length-1:]
    
    # Flatten the test sequences for the RandomForestClassifier
    n_samples_test, n_timesteps, n_features_actual = np.array(test_sequences).shape
    X_test_flat = np.array(test_sequences).reshape(n_samples_test, n_timesteps * n_features_actual)
    y_test_true = adjusted_true_anomalies[:n_samples_test]
    
    print(f"Test sequences shape: {n_samples_test}, {n_timesteps}, {n_features_actual}")
    print(f"Adjusted labels length: {len(y_test_true)}")
    
    # Check if we have both classes for training RF
    unique_classes = np.unique(y_test_true)
    print(f"Unique classes in labels: {unique_classes}")
    
    if len(unique_classes) < 2:
        print("Warning: Only one class present in labels. Using all features without RF selection.")
        n_features_total = n_features
        num_selected_features = min(25, n_features_total)
        selected_feature_indices = np.arange(num_selected_features)
        remaining_feature_indices = np.arange(num_selected_features, n_features_total)
    else:
        # Initialize and train the RandomForestClassifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_test_flat, y_test_true)
        
        # Get feature importances
        feature_importances = rf_model.feature_importances_
        
        # Map back to original features by averaging across timesteps
        original_feature_importances = feature_importances.reshape(n_timesteps, n_features_actual).mean(axis=0)
        
        # Rank features by importance
        ranked_features_indices = np.argsort(original_feature_importances)[::-1]
        ranked_features_importance = original_feature_importances[ranked_features_indices]
        
        print("Feature Importances (averaged across timesteps):")
        for i, index in enumerate(ranked_features_indices):
            print(f"Feature {index}: {ranked_features_importance[i]:.4f}")
        
        # Determine the number of top features to select
        n_features_total = n_features_actual
        num_selected_features = min(25, max(1, n_features_total // 2))
        
        # Get the indices of the top N features
        selected_feature_indices = ranked_features_indices[:num_selected_features]
        remaining_feature_indices = ranked_features_indices[num_selected_features:]
    
    print(f"\nSelected Top {len(selected_feature_indices)} Feature Indices: {selected_feature_indices}")
    print(f"Remaining {len(remaining_feature_indices)} Feature Indices: {remaining_feature_indices}")
    
    # Handle case where we have very few features
    if len(remaining_feature_indices) == 0:
        print("Warning: No remaining features. Using half of selected features as remaining.")
        mid = len(selected_feature_indices) // 2
        remaining_feature_indices = selected_feature_indices[mid:]
        selected_feature_indices = selected_feature_indices[:mid]
    
    return selected_feature_indices, remaining_feature_indices


def split_features_by_indices(data, selected_indices, remaining_indices):
    """
    Split data into top and remaining features based on indices.
    
    Args:
        data: numpy array of shape (timesteps, features)
        selected_indices: Indices of selected/top features
        remaining_indices: Indices of remaining features
    
    Returns:
        data_top: Data with only top features
        data_remaining: Data with only remaining features
    """
    data_top = data[:, selected_indices]
    data_remaining = data[:, remaining_indices]
    return data_top, data_remaining
