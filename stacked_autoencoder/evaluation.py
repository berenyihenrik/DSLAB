# -*- coding: utf-8 -*-
"""Evaluation functions for LSTM VAE anomaly detection."""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score

from training import loss_function_grouped


def _compute_per_group_losses(model, loader, device):
    """Compute per-sample reconstruction loss for each encoder group.

    Returns:
        list of 1-D numpy arrays, one per group.  Each array has length
        equal to the number of samples in ``loader``.
    """
    model.eval()
    all_group_losses = [[] for _ in model.group_positions]

    with torch.no_grad():
        for batch in loader:
            x_groups = [torch.as_tensor(g, dtype=torch.float32).to(device) for g in batch]
            x_recon, _mean, _logvar = model(x_groups)

            for gi, (x_g, positions) in enumerate(zip(x_groups, model.group_positions)):
                x_recon_g = x_recon[:, :, positions]
                if model.binary_group_flags is not None and model.binary_group_flags[gi]:
                    losses = nn.functional.binary_cross_entropy_with_logits(
                        x_recon_g, x_g, reduction='none').mean(dim=(1, 2))
                else:
                    losses = (x_recon_g - x_g).pow(2).mean(dim=(1, 2))
                all_group_losses[gi].append(losses.cpu().numpy())

    return [np.concatenate(chunks) for chunks in all_group_losses]


def fit_group_ecdf(model, baseline_loader, device, trim_q=(0.005, 0.995)):
    """Build sorted per-group loss arrays from a baseline (train/val) loader.

    These sorted arrays serve as empirical CDFs so that test-time losses
    can be converted into two-sided tail probabilities.

    Args:
        model: Trained LSTMVAE_Grouped model.
        baseline_loader: DataLoader over normal (training/validation) data.
        device: torch device.
        trim_q: (lo, hi) quantile pair for trimming extremes before storing
                 the ECDF.  Set to None to disable trimming.

    Returns:
        baseline_ecdfs: list of sorted 1-D numpy arrays, one per group.
    """
    group_losses = _compute_per_group_losses(model, baseline_loader, device)
    baseline_ecdfs = []
    for arr in group_losses:
        if trim_q is not None:
            lo, hi = np.quantile(arr, trim_q)
            arr = arr[(arr >= lo) & (arr <= hi)]
        arr = np.sort(arr)
        baseline_ecdfs.append(arr)
    return baseline_ecdfs


def compute_anomaly_scores_grouped(model, test_loader, device, baseline_ecdfs=None,
                                   combine="mean", kl_score_weight=0.0):
    """
    Compute per-sample anomaly scores.

    When ``baseline_ecdfs`` is provided (from :func:`fit_group_ecdf`), each
    group's reconstruction loss is converted into a two-sided ECDF tail
    score (``-log(p)``).  This makes the score direction-agnostic: both
    unusually *high* and unusually *low* reconstruction errors relative to
    the training baseline are flagged.  Group scores are then combined
    with ``combine`` (default ``"mean"``).

    When ``baseline_ecdfs`` is ``None``, the original weighted-mean
    reconstruction scoring is used (backward-compatible).

    Args:
        model: Trained LSTMVAE_Grouped model
        test_loader: Test data loader
        device: Device
        baseline_ecdfs: Optional list of sorted baseline loss arrays from
                        :func:`fit_group_ecdf`.  Enables calibrated scoring.
        combine: Aggregation mode when using calibrated scoring.
                 ``"mean"`` (default) or ``"max"``.
        kl_score_weight: Weight for per-sample KL term (only used in
                         uncalibrated mode).

    Returns:
        anomaly_scores: list[float] — one score per sample
    """
    model.eval()
    anomaly_scores = []
    eps = 1e-12

    with torch.no_grad():
        for batch in test_loader:
            x_groups = [torch.as_tensor(g, dtype=torch.float32).to(device) for g in batch]
            x_recon, mean, logvar = model(x_groups)
            batch_size = x_groups[0].shape[0]

            if baseline_ecdfs is not None:
                # --- Calibrated two-sided ECDF scoring ---
                per_group = np.zeros((batch_size, len(model.group_positions)))

                for gi, (x_g, positions) in enumerate(zip(x_groups, model.group_positions)):
                    x_recon_g = x_recon[:, :, positions]
                    if model.binary_group_flags is not None and model.binary_group_flags[gi]:
                        losses = nn.functional.binary_cross_entropy_with_logits(
                            x_recon_g, x_g, reduction='none').mean(dim=(1, 2))
                    else:
                        losses = (x_recon_g - x_g).pow(2).mean(dim=(1, 2))

                    losses_np = losses.cpu().numpy()
                    sorted_base = baseline_ecdfs[gi]
                    # Empirical CDF: fraction of baseline values <= l
                    u = np.searchsorted(sorted_base, losses_np, side='right') / max(1, len(sorted_base))
                    # Two-sided tail probability → anomaly score
                    p = 2.0 * np.minimum(u, 1.0 - u)
                    per_group[:, gi] = -np.log(p + eps)

                if combine == "max":
                    batch_scores = per_group.max(axis=1)
                else:
                    batch_scores = per_group.mean(axis=1)

                anomaly_scores.extend(batch_scores.tolist())
            else:
                # --- Legacy weighted-mean scoring ---
                batch_scores = torch.zeros(batch_size, device=device)

                for gi, (x_g, positions) in enumerate(zip(x_groups, model.group_positions)):
                    x_recon_g = x_recon[:, :, positions]
                    if model.binary_group_flags is not None and model.binary_group_flags[gi]:
                        group_scores = nn.functional.binary_cross_entropy_with_logits(
                            x_recon_g, x_g, reduction='none').mean(dim=(1, 2))
                    else:
                        group_scores = (x_recon_g - x_g).pow(2).mean(dim=(1, 2))
                    batch_scores += model.group_weights[gi] * group_scores

                if kl_score_weight > 0:
                    kl_per_sample = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).mean(dim=1)
                    batch_scores += kl_score_weight * kl_per_sample

                anomaly_scores.extend(batch_scores.cpu().tolist())

    return anomaly_scores


def compute_threshold_from_baseline(model, baseline_loader, device, percentile_threshold,
                                    baseline_ecdfs=None):
    """Derive an anomaly-score threshold from a **normal** (validation) loader.

    The threshold is set at the *percentile_threshold*-th percentile of the
    anomaly-score distribution computed over normal data.  This avoids any
    dependence on the test set.

    Args:
        model: Trained LSTMVAE_Grouped model.
        baseline_loader: DataLoader over normal (training/validation) data.
        device: torch device.
        percentile_threshold: Percentile (0–100) of the validation score
                              distribution above which a sample is flagged.
        baseline_ecdfs: Optional ECDF arrays (from :func:`fit_group_ecdf`)
                        for calibrated scoring.

    Returns:
        threshold: float — absolute anomaly-score cutoff.
        val_scores: list[float] — anomaly scores on the baseline data.
    """
    val_scores = compute_anomaly_scores_grouped(
        model, baseline_loader, device, baseline_ecdfs=baseline_ecdfs)
    threshold = np.percentile(val_scores, percentile_threshold)
    return threshold, val_scores


def evaluate_lstm_grouped(model, test_loader, device, percentile_threshold=90,
                          baseline_ecdfs=None, baseline_loader=None):
    """
    Evaluate the grouped LSTM VAE model and return anomaly indices.
    
    When ``baseline_loader`` is provided the threshold is derived from
    the **validation** score distribution (no test-data leakage).
    Otherwise falls back to thresholding on the test scores themselves.

    Args:
        model: Trained LSTMVAE_Grouped model
        test_loader: Test data loader (yields tuples of N tensors, one per group)
        device: Device
        percentile_threshold: Percentile threshold for anomaly detection
        baseline_ecdfs: Optional calibration from :func:`fit_group_ecdf`.
        baseline_loader: Optional DataLoader over normal data for threshold
                         calibration.  Recommended to avoid test leakage.
    
    Returns:
        anomaly_indices: List of indices classified as anomalies
        anomaly_scores: List of anomaly scores for each sequence
    """
    anomaly_scores = compute_anomaly_scores_grouped(
        model, test_loader, device, baseline_ecdfs=baseline_ecdfs)

    if baseline_loader is not None:
        threshold, _ = compute_threshold_from_baseline(
            model, baseline_loader, device, percentile_threshold,
            baseline_ecdfs=baseline_ecdfs)
    else:
        threshold = np.percentile(anomaly_scores, percentile_threshold)

    anomaly_indices = [i for i, score in enumerate(anomaly_scores) if score > threshold]
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
    
    # Calculate AUC-ROC and AUCPR using continuous anomaly scores
    unique_true = np.unique(adjusted_true_anomalies)
    if len(unique_true) > 1:
        scores_array = np.array(anomaly_scores[:len(adjusted_true_anomalies)])
        auc_roc = roc_auc_score(adjusted_true_anomalies, scores_array)
        print(f"AUC-ROC Score: {auc_roc:.4f}")
        
        auc_pr = average_precision_score(adjusted_true_anomalies, scores_array)
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
