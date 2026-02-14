# -*- coding: utf-8 -*-
"""Two-stage unsupervised feature selection on training data only.

Stage 1: Redundancy filtering via Spearman correlation + agglomerative clustering.
Stage 2: AE masking importance — train a small LSTM-AE, then measure per-feature
         reconstruction loss increase under block permutation on a held-out val split.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Internal lightweight LSTM autoencoder (used only for feature importance)
# ---------------------------------------------------------------------------

class _SmallLSTMAE(nn.Module):
    """Lightweight LSTM autoencoder for feature importance scoring."""

    def __init__(self, n_features, hidden_dim, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.latent_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.encoder(x)
        h = self.latent_to_hidden(h_n[-1])  # (batch, hidden_dim)
        h_repeated = h.unsqueeze(1).repeat(1, self.sequence_length, 1)
        dec_out, _ = self.decoder(h_repeated)
        return self.output_layer(dec_out)


# ---------------------------------------------------------------------------
# Stage 1: Redundancy filtering
# ---------------------------------------------------------------------------

def _compute_redundancy_clusters(train_data, corr_threshold=0.9):
    """Cluster features by Spearman correlation and pick one representative
    per cluster (highest IQR variance).

    Args:
        train_data: array of shape (timesteps, n_features)
        corr_threshold: |corr| above which features are merged

    Returns:
        representative_indices: 1-D array of selected feature indices
        cluster_labels: cluster id for every original feature (for logging)
    """
    n_features = train_data.shape[1]

    # Spearman correlation matrix
    corr_matrix, _ = spearmanr(train_data)
    # spearmanr returns a scalar when n_features==2; ensure matrix
    corr_matrix = np.atleast_2d(corr_matrix)

    # Replace NaN correlations (from constant features) with 0 (max distance)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Distance = 1 - |corr|
    dist_matrix = 1.0 - np.abs(corr_matrix)
    np.fill_diagonal(dist_matrix, 0.0)
    # Ensure numerical symmetry and finite values
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    dist_matrix = np.clip(dist_matrix, 0.0, 1.0)
    # Condensed form for scipy
    condensed = dist_matrix[np.triu_indices(n_features, k=1)]

    # Agglomerative clustering
    Z = linkage(condensed, method='average')
    cluster_labels = fcluster(Z, t=1.0 - corr_threshold, criterion='distance')

    # For each cluster, pick the feature with the highest IQR
    iqr_scores = np.subtract(*np.percentile(train_data, [75, 25], axis=0))
    representative_indices = []
    unique_clusters = np.unique(cluster_labels)
    for cid in unique_clusters:
        members = np.where(cluster_labels == cid)[0]
        best = members[np.argmax(iqr_scores[members])]
        representative_indices.append(best)

    representative_indices = np.sort(representative_indices)

    print(f"Stage 1 — Redundancy filtering:")
    print(f"  {n_features} features → {len(unique_clusters)} clusters (threshold |corr| > {corr_threshold})")
    print(f"  Representative indices: {representative_indices}")

    return representative_indices, cluster_labels


# ---------------------------------------------------------------------------
# Stage 2: AE masking importance
# ---------------------------------------------------------------------------

def _block_permute(arr, block_size):
    """Permute an array in contiguous blocks to preserve local temporal structure."""
    n = len(arr)
    n_blocks = max(1, n // block_size)
    blocks = np.array_split(arr, n_blocks)
    perm = np.random.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in perm])


def _compute_masking_importance(
    train_data,
    representative_indices,
    sequence_length,
    device,
    hidden_dim=64,
    num_epochs=30,
    batch_size=64,
    n_repeats=3,
):
    """Train a small LSTM-AE on the representative features and score each
    feature by reconstruction-loss increase under block permutation.

    Args:
        train_data: array (timesteps, n_all_features)
        representative_indices: feature indices to evaluate
        sequence_length: window length for sequences
        device: torch device
        hidden_dim: hidden size for the small AE
        num_epochs: training epochs for the small AE
        batch_size: batch size for AE training
        n_repeats: number of permutation repeats to average importance

    Returns:
        importance_scores: array of shape (len(representative_indices),)
    """
    # Subset to representative features
    data = train_data[:, representative_indices].astype(np.float32)
    n_features = data.shape[1]

    # Create sequences
    sequences = []
    for i in range(data.shape[0] - sequence_length + 1):
        sequences.append(data[i: i + sequence_length])
    sequences = np.array(sequences)  # (N, seq_len, n_features)

    # Train / val split
    train_seq, val_seq = train_test_split(sequences, test_size=0.3, random_state=42)
    train_tensor = torch.tensor(train_seq, dtype=torch.float32)
    val_tensor = torch.tensor(val_seq, dtype=torch.float32)

    train_ds = TensorDataset(train_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Build and train small AE
    model = _SmallLSTMAE(n_features, hidden_dim, sequence_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"\nStage 2 — Training small LSTM-AE ({n_features} features, {num_epochs} epochs)...")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}  loss={epoch_loss / len(train_loader):.6f}")

    # Compute baseline MSE on val set
    model.eval()
    val_device = val_tensor.to(device)
    with torch.no_grad():
        baseline_recon = model(val_device)
        baseline_mse = torch.mean((baseline_recon - val_device) ** 2).item()
    print(f"  Baseline val MSE: {baseline_mse:.6f}")

    # Per-feature importance via block permutation
    importance_scores = np.zeros(n_features)
    rng = np.random.RandomState(42)

    for fi in range(n_features):
        delta_sum = 0.0
        for _ in range(n_repeats):
            corrupted = val_seq.copy()  # (N, seq_len, n_features)
            # Block-permute feature fi across the time axis within each sample
            for si in range(corrupted.shape[0]):
                corrupted[si, :, fi] = _block_permute(corrupted[si, :, fi], block_size=max(1, sequence_length // 4))
            corrupted_tensor = torch.tensor(corrupted, dtype=torch.float32).to(device)
            with torch.no_grad():
                recon_corrupted = model(corrupted_tensor)
                mse_corrupted = torch.mean((recon_corrupted - val_device) ** 2).item()
            delta_sum += mse_corrupted - baseline_mse
        importance_scores[fi] = delta_sum / n_repeats

    print("\n  Feature masking importance (ΔMSE):")
    ranked = np.argsort(importance_scores)[::-1]
    for rank, fi in enumerate(ranked):
        orig_idx = representative_indices[fi]
        print(f"    Feature {orig_idx}: {importance_scores[fi]:.6f}")

    return importance_scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def perform_feature_selection(train_data, n_features, sequence_length, device, corr_threshold=0.9):
    """Two-stage unsupervised feature selection on training data only.

    Stage 1: Spearman correlation clustering → remove redundant features.
    Stage 2: LSTM-AE masking importance → rank remaining features.

    Args:
        train_data: Training array of shape (timesteps, n_features)
        n_features: Total number of features
        sequence_length: Window length used for sequences
        device: torch device (for AE training)
        corr_threshold: |correlation| above which features are clustered together

    Returns:
        selected_feature_indices: Indices of top features (into original feature space)
        remaining_feature_indices: Indices of remaining features
    """
    # Stage 1: redundancy filtering
    representative_indices, _ = _compute_redundancy_clusters(train_data, corr_threshold)

    # Stage 2: masking importance on representatives
    importance_scores = _compute_masking_importance(
        train_data, representative_indices, sequence_length, device
    )

    # Rank representatives by importance
    ranked_order = np.argsort(importance_scores)[::-1]
    ranked_original_indices = representative_indices[ranked_order]

    # Select top-K
    num_selected = min(25, max(1, n_features // 2))
    # Can't select more than we have representatives
    num_selected = min(num_selected, len(ranked_original_indices))

    selected_feature_indices = np.sort(ranked_original_indices[:num_selected])
    remaining_feature_indices = np.sort(ranked_original_indices[num_selected:])

    # If all representatives were selected, no remaining — handle edge case
    if len(remaining_feature_indices) == 0:
        print("Warning: No remaining features after selection. Splitting selected in half.")
        mid = len(selected_feature_indices) // 2
        remaining_feature_indices = selected_feature_indices[mid:]
        selected_feature_indices = selected_feature_indices[:mid]

    print(f"\nFinal selection:")
    print(f"  Selected ({len(selected_feature_indices)}): {selected_feature_indices}")
    print(f"  Remaining ({len(remaining_feature_indices)}): {remaining_feature_indices}")

    return selected_feature_indices, remaining_feature_indices


def split_features_by_indices(data, selected_indices, remaining_indices):
    """Split data into top and remaining features based on indices.

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
