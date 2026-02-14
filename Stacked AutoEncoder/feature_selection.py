# -*- coding: utf-8 -*-
"""Feature selection functions using unsupervised methods on training data only."""

import numpy as np


def perform_feature_selection(train_data, n_features):
    """
    Perform feature selection using unsupervised variance-based ranking
    on training data only (no data leakage).
    
    Features are ranked by their variance on the training set. High-variance
    features are more likely to carry useful signal for anomaly detection.
    
    Args:
        train_data: Training data array of shape (timesteps, n_features)
        n_features: Number of features
    
    Returns:
        selected_feature_indices: Indices of top features
        remaining_feature_indices: Indices of remaining features
    """
    # Compute per-feature variance on training data
    feature_variances = np.var(train_data, axis=0)
    
    # Rank features by variance (highest first)
    ranked_features_indices = np.argsort(feature_variances)[::-1]
    ranked_features_variance = feature_variances[ranked_features_indices]
    
    print("Feature Variances (training data):")
    for i, index in enumerate(ranked_features_indices):
        print(f"Feature {index}: {ranked_features_variance[i]:.6f}")
    
    # Determine the number of top features to select
    num_selected_features = min(25, max(1, n_features // 2))
    
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
