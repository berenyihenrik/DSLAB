# -*- coding: utf-8 -*-
"""Evaluation functions for LSTM VAE anomaly detection."""

import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score

from training import loss_function_grouped, loss_function_grouped_decomposed


def compute_anomaly_scores_grouped(model, loader, device, kl_weight=0.1):
    """
    Compute per-sample anomaly scores efficiently using decomposed loss.

    Args:
        model: Trained LSTMVAE_Grouped model
        loader: Data loader (yields tuples of N tensors, one per group)
        device: Device
        kl_weight: KL weight used during training

    Returns:
        scores: List of anomaly scores (one per sequence)
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            x_groups = [g.to(device) for g in batch]
            x_recon, mean, logvar = model(x_groups)
            total_per_sample, _ = loss_function_grouped_decomposed(
                x_groups, x_recon, mean, logvar,
                model.group_weights, model.group_positions,
                latent_dim=model.latent_dim,
                kl_weight=kl_weight,
                return_timestep_errors=False,
            )
            scores.extend(total_per_sample.detach().cpu().numpy().tolist())
    return scores


def extract_anomaly_attributions(model, loader, device, indices, kl_weight=0.1):
    """
    Extract detailed per-group and per-feature attribution for specific sample indices.

    Only computes the heavy (B, T, F) tensors for the requested indices to avoid
    storing attributions for the entire test set.

    Args:
        model: Trained LSTMVAE_Grouped model
        loader: Data loader (yields tuples of N tensors, one per group)
        device: Device
        indices: Iterable of sample indices to extract attributions for
        kl_weight: KL weight used during training

    Returns:
        List of dicts, one per requested index, each containing:
            index, window_start, window_end, score_total,
            recon_contrib_by_group (G,), recon_contrib_by_feature (F,),
            recon_sqerr_t_f (T, F), kld_total, kld_contrib_by_group (G,),
            feature_order, encoder_groups
    """
    model.eval()
    indices_set = set(indices)
    results = []
    global_i = 0

    with torch.no_grad():
        for batch in loader:
            x_groups = [g.to(device) for g in batch]
            B = x_groups[0].shape[0]

            # Check if any requested index falls within this batch
            batch_indices = set(range(global_i, global_i + B))
            if not batch_indices & indices_set:
                global_i += B
                continue

            x_recon, mean, logvar = model(x_groups)
            total_per_sample, comp = loss_function_grouped_decomposed(
                x_groups, x_recon, mean, logvar,
                model.group_weights, model.group_positions,
                latent_dim=model.latent_dim,
                kl_weight=kl_weight,
                return_timestep_errors=True,
            )

            for bi in range(B):
                if global_i in indices_set:
                    results.append({
                        "index": global_i,
                        "window_start": global_i,
                        "window_end": global_i + model.sequence_length - 1,
                        "score_total": float(total_per_sample[bi].cpu()),
                        "recon_contrib_by_group": comp["recon_contrib_per_group"][bi].cpu().numpy(),
                        "recon_contrib_by_feature": comp["recon_contrib_per_feature"][bi].cpu().numpy(),
                        "recon_sqerr_t_f": comp["recon_sqerr_t_f"][bi].cpu().numpy(),
                        "kld_total": float(comp["kld_total_per_sample"][bi].cpu()),
                        "kld_contrib_by_group": comp["kld_contrib_per_group"][bi].cpu().numpy(),
                        "feature_order": model.feature_order,
                        "encoder_groups": model.encoder_groups,
                    })
                global_i += 1

    return results


def evaluate_lstm_grouped(model, test_loader, device, percentile_threshold=90,
                          kl_weight=0.1, return_attributions=False):
    """
    Evaluate the grouped LSTM VAE model and return anomaly indices.
    
    Args:
        model: Trained LSTMVAE_Grouped model
        test_loader: Test data loader (yields tuples of N tensors, one per group)
        device: Device
        percentile_threshold: Percentile threshold for anomaly detection
        kl_weight: KL weight used during training
        return_attributions: If True, perform a second pass to extract per-feature
                            attribution dicts for each detected anomaly
    
    Returns:
        anomaly_indices: List of indices classified as anomalies
        anomaly_scores: List of anomaly scores for each sequence
        attributions: (only if return_attributions=True) List of attribution dicts
    """
    anomaly_scores = compute_anomaly_scores_grouped(model, test_loader, device, kl_weight)

    threshold = np.percentile(anomaly_scores, percentile_threshold)
    anomaly_indices = [i for i, score in enumerate(anomaly_scores) if score > threshold]

    if return_attributions:
        attributions = extract_anomaly_attributions(
            model, test_loader, device, anomaly_indices, kl_weight
        )
        return anomaly_indices, anomaly_scores, attributions

    return anomaly_indices, anomaly_scores


# Backward compatibility alias
evaluate_lstm_weighted = evaluate_lstm_grouped


def calculate_f1_score_smap_msl(anomaly_indices, true_anomalies, sequence_length):
    """
    Calculate F1 score for SMAP/MSL datasets.
    
    For windowed sequences, we need to align predictions with the original labels.
    Each sequence at index i corresponds to the time window [i, i+sequence_length-1].
    We assign the prediction to the last timestep of each window.
    
    Args:
        anomaly_indices: List of detected anomaly indices
        true_anomalies: Ground truth anomaly labels
        sequence_length: Sequence length used
    
    Returns:
        f1: F1 score
        predicted_anomalies: Binary prediction array
        adjusted_true_anomalies: Adjusted ground truth labels
    """
    # Adjust true_anomalies to match the sequence indices
    adjusted_true_anomalies = true_anomalies[sequence_length-1:]
    
    # Create a binary array representing predicted anomalies
    predicted_anomalies = np.zeros(len(adjusted_true_anomalies), dtype=int)
    for index in anomaly_indices:
        if index < len(predicted_anomalies):  # Check index bounds
            predicted_anomalies[index] = 1

    # Calculate the F1 score
    f1 = f1_score(adjusted_true_anomalies, predicted_anomalies)
    return f1, predicted_anomalies, adjusted_true_anomalies


# Alias for backward compatibility and generic usage
calculate_f1_score = calculate_f1_score_smap_msl


def point_adjust_f1_score(anomaly_indices, true_anomalies, sequence_length, anomaly_sequences):
    """
    Calculate Point-Adjust F1 score used in SMAP/MSL literature.
    
    If any point within an anomaly segment is detected, the entire segment 
    is considered as correctly detected (True Positive).
    
    Args:
        anomaly_indices: List of detected anomaly indices
        true_anomalies: Ground truth anomaly labels
        sequence_length: Sequence length used
        anomaly_sequences: List of [start, end] pairs for each anomaly segment
    
    Returns:
        Dictionary with precision, recall, f1, and detection statistics
    """
    # Map sequence indices back to original time indices
    detected_positions = set()
    for idx in anomaly_indices:
        # The sequence at index idx covers positions [idx, idx + sequence_length - 1]
        for pos in range(idx, idx + sequence_length):
            if pos < len(true_anomalies):
                detected_positions.add(pos)
    
    # Check each anomaly segment
    true_positives = 0
    false_negatives = 0
    
    for seq in anomaly_sequences:
        start, end = seq
        # Check if any position in this segment was detected
        segment_detected = any(pos in detected_positions for pos in range(start, end + 1))
        if segment_detected:
            true_positives += 1
        else:
            false_negatives += 1
    
    # Count false positives (detected positions that are not in any anomaly)
    false_positives = 0
    for pos in detected_positions:
        if true_anomalies[pos] == 0:
            false_positives += 1
    
    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_segments': len(anomaly_sequences)
    }


def print_evaluation_results(anomaly_scores, anomalies, true_anomalies, sequence_length, 
                             percentile_threshold, anomaly_sequences):
    """
    Print comprehensive evaluation results.
    
    Args:
        anomaly_scores: List of anomaly scores
        anomalies: List of detected anomaly indices
        true_anomalies: Ground truth anomaly labels
        sequence_length: Sequence length used
        percentile_threshold: Percentile threshold used
        anomaly_sequences: List of anomaly segments
    """
    threshold_value = np.percentile(anomaly_scores, percentile_threshold)
    
    print("\n--- Anomaly Score Diagnostics ---")
    print(f"Anomaly scores - Min: {np.min(anomaly_scores):.4f}, Max: {np.max(anomaly_scores):.4f}")
    print(f"Anomaly scores - Mean: {np.mean(anomaly_scores):.4f}, Std: {np.std(anomaly_scores):.4f}")
    print(f"Threshold percentile: {percentile_threshold}")
    print(f"Threshold value: {threshold_value:.4f}")
    print(f"Number of detected anomalies: {len(anomalies)}")
    print(f"True anomaly rate: {np.sum(true_anomalies) / len(true_anomalies) * 100:.2f}%")
    
    # Calculate F1 score
    f1, predicted_anomalies, adjusted_true_anomalies = calculate_f1_score_smap_msl(
        anomalies, true_anomalies, sequence_length
    )
    print(f"\nF1 Score (with threshold percentile {percentile_threshold}): {f1:.4f}")
    
    # Calculate AUC-ROC if we have both classes
    unique_true = np.unique(adjusted_true_anomalies)
    if len(unique_true) > 1:
        auc_roc = roc_auc_score(adjusted_true_anomalies, predicted_anomalies)
        print(f"AUC-ROC Score: {auc_roc:.4f}")
        
        auc_pr = average_precision_score(adjusted_true_anomalies, predicted_anomalies)
        print(f"AUCPR Score: {auc_pr:.4f}")
    else:
        print("Warning: Only one class in true labels, cannot compute AUC-ROC or AUCPR")
    
    print("\nClassification Report:")
    print(classification_report(adjusted_true_anomalies, predicted_anomalies, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(adjusted_true_anomalies, predicted_anomalies))
    
    # Point-Adjust F1
    pa_results = point_adjust_f1_score(anomalies, true_anomalies, sequence_length, anomaly_sequences)
    print(f"\nPoint-Adjust Evaluation (common in SMAP/MSL papers):")
    print(f"  Precision: {pa_results['precision']:.4f}")
    print(f"  Recall: {pa_results['recall']:.4f}")
    print(f"  F1 Score: {pa_results['f1']:.4f}")
    print(f"  Detected Segments: {pa_results['true_positives']} / {pa_results['total_segments']}")
    
    return f1, pa_results, predicted_anomalies, adjusted_true_anomalies


def print_evaluation_results_simple(anomaly_scores, anomalies, true_anomalies, sequence_length, 
                                    percentile_threshold):
    """
    Print evaluation results without point-adjust metrics (for SMD dataset).
    
    Args:
        anomaly_scores: List of anomaly scores
        anomalies: List of detected anomaly indices
        true_anomalies: Ground truth anomaly labels
        sequence_length: Sequence length used
        percentile_threshold: Percentile threshold used
    
    Returns:
        f1: F1 score
        predicted_anomalies: Binary prediction array
        adjusted_true_anomalies: Adjusted ground truth labels
    """
    threshold_value = np.percentile(anomaly_scores, percentile_threshold)
    
    print("\n--- Anomaly Score Diagnostics ---")
    print(f"Anomaly scores - Min: {np.min(anomaly_scores):.4f}, Max: {np.max(anomaly_scores):.4f}")
    print(f"Anomaly scores - Mean: {np.mean(anomaly_scores):.4f}, Std: {np.std(anomaly_scores):.4f}")
    print(f"Threshold percentile: {percentile_threshold}")
    print(f"Threshold value: {threshold_value:.4f}")
    print(f"Number of detected anomalies: {len(anomalies)}")
    print(f"True anomaly rate: {np.sum(true_anomalies) / len(true_anomalies) * 100:.2f}%")
    
    # Calculate F1 score
    f1, predicted_anomalies, adjusted_true_anomalies = calculate_f1_score(
        anomalies, true_anomalies, sequence_length
    )
    print(f"\nF1 Score (with threshold percentile {percentile_threshold}): {f1:.4f}")
    
    # Calculate AUC-ROC if we have both classes
    unique_true = np.unique(adjusted_true_anomalies)
    if len(unique_true) > 1:
        auc_roc = roc_auc_score(adjusted_true_anomalies, predicted_anomalies)
        print(f"AUC-ROC Score: {auc_roc:.4f}")
        
        auc_pr = average_precision_score(adjusted_true_anomalies, predicted_anomalies)
        print(f"AUCPR Score: {auc_pr:.4f}")
    else:
        print("Warning: Only one class in true labels, cannot compute AUC-ROC or AUCPR")
    
    print("\nClassification Report:")
    print(classification_report(adjusted_true_anomalies, predicted_anomalies, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(adjusted_true_anomalies, predicted_anomalies))
    
    return f1, predicted_anomalies, adjusted_true_anomalies
