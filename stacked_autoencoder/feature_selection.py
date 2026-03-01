# -*- coding: utf-8 -*-
"""Four-stage unsupervised feature selection on training data only.

Stage 0: Drop static features (std == 0).
Stage 1: Redundancy clustering via lagged Spearman correlation + agglomerative clustering.
Stage 2: AE masking importance — train a small LSTM-AE on cluster representatives,
         then measure per-feature reconstruction loss increase under block permutation.
Stage 3: Group-aware encoder assignment — rank clusters by importance, assign
         high-importance clusters to individual encoders and merge the rest.
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
# Stage 0: Drop static features
# ---------------------------------------------------------------------------

def _drop_static_features(train_data):
    """Handle static features (std == 0).

    Static binary features (≤2 unique values) are kept — they may only
    flip during anomalies and serve as strong detectors.  Static
    continuous features are dropped as uninformative.

    Args:
        train_data: array of shape (timesteps, n_features)

    Returns:
        kept_feature_indices: 1-D array of non-static original feature indices
        dropped_feature_indices: 1-D array of dropped (static continuous) feature indices
        static_binary_indices: 1-D array of static binary feature indices (kept separately)
    """
    n_features = train_data.shape[1]
    stds = np.std(train_data, axis=0)

    is_static = stds == 0
    static_indices = np.where(is_static)[0]

    # Classify static features as binary vs continuous.
    # A static feature is "binary-safe" only if its values are in {0, 1},
    # so BCE loss can be used safely at eval time.
    static_binary = []
    static_continuous = []
    for idx in static_indices:
        unique_vals = np.unique(train_data[:, idx])
        if len(unique_vals) <= 2 and all(v in (0.0, 1.0) for v in unique_vals):
            static_binary.append(idx)
        else:
            static_continuous.append(idx)

    static_binary = np.array(static_binary, dtype=int)
    dropped = np.array(static_continuous, dtype=int)
    kept = np.where(~is_static)[0]

    print("Stage 0 — Handle static features:")
    print(f"  {n_features} total features, {len(static_indices)} static (std==0)")
    print(f"  {len(kept)} dynamic (kept for clustering)")
    print(f"  {len(static_binary)} static binary (kept as sentinel group)")
    print(f"  {len(dropped)} static continuous (dropped)")
    if len(static_binary) > 0:
        print(f"  Static binary indices: {static_binary.tolist()}")
    if len(dropped) > 0:
        print(f"  Dropped indices: {dropped.tolist()}")

    return kept, dropped, static_binary


# ---------------------------------------------------------------------------
# Stage 1: Redundancy clustering (lagged Spearman)
# ---------------------------------------------------------------------------

def _compute_lagged_spearman_similarity(train_data, max_lag, lag_penalty_lambda=None):
    """Pairwise similarity via max lagged |Spearman rho| on first-differenced data.

    Steps:
        1. First-difference the data to remove drift / shared trends.
        2. For each lag tau in [-max_lag, max_lag], compute pairwise
           Spearman correlation between diff_i(t) and diff_j(t + tau).
        3. similarity(i, j) = max_tau  |rho_ij(tau)| * w(tau)
           where w(tau) = exp(-|tau| / lambda) if lambda is set, else 1.

    Args:
        train_data: array of shape (timesteps, n_features)
        max_lag: maximum lag L; tau ranges over [-L, L]
        lag_penalty_lambda: decay constant for the lag penalty; None disables it

    Returns:
        similarity: array of shape (n_features, n_features) in [0, 1]
    """
    diff_data = np.diff(train_data, axis=0)  # (T-1, F)
    T, F = diff_data.shape

    # Clamp max_lag so we keep at least half the samples for correlation
    max_lag = min(max_lag, T // 2)

    similarity = np.zeros((F, F))

    for tau in range(-max_lag, max_lag + 1):
        # Align arrays for this lag
        if tau > 0:
            x = diff_data[:T - tau]
            y = diff_data[tau:]
        elif tau < 0:
            x = diff_data[-tau:]
            y = diff_data[:T + tau]
        else:
            x = diff_data
            y = diff_data

        # Compute Spearman cross-correlation
        if F == 1:
            rho, _ = spearmanr(x[:, 0], y[:, 0])
            cross_corr = np.atleast_2d(rho)
        else:
            # spearmanr(a, b) with a=(N,F), b=(N,F) returns (2F, 2F);
            # the cross-block [0:F, F:2F] holds corr(a_i, b_j).
            combined_corr, _ = spearmanr(x, y)
            cross_corr = combined_corr[:F, F:]

        cross_corr = np.nan_to_num(cross_corr, nan=0.0)

        weight = np.exp(-abs(tau) / lag_penalty_lambda) if lag_penalty_lambda else 1.0
        weighted = np.abs(cross_corr) * weight
        similarity = np.maximum(similarity, weighted)

    # Self-similarity = 1; enforce exact symmetry
    np.fill_diagonal(similarity, 1.0)
    similarity = np.maximum(similarity, similarity.T)

    return similarity


def _compute_redundancy_clusters(train_data, corr_threshold=0.9, max_lag=30,
                                 lag_penalty_lambda=None):
    """Cluster features by lagged Spearman correlation and pick one representative
    per cluster (medoid — most central member in correlation-distance space).
    Preserves full cluster membership.

    Args:
        train_data: array of shape (timesteps, n_features)
        corr_threshold: |corr| above which features are merged
        max_lag: maximum lag for lagged Spearman similarity
        lag_penalty_lambda: optional decay constant for lag penalty

    Returns:
        representative_indices: 1-D array of selected feature column indices
            (positions within *train_data*, not original feature space)
        cluster_labels: cluster id for every column in train_data
        cluster_members_dict: dict mapping cluster_id → list of column indices
            (positions within *train_data*)
        cluster_rep_dict: dict mapping cluster_id → representative column index
    """
    n_features = train_data.shape[1]

    # Lagged Spearman similarity matrix
    similarity = _compute_lagged_spearman_similarity(
        train_data, max_lag, lag_penalty_lambda
    )

    # Distance = 1 - similarity
    dist_matrix = 1.0 - similarity
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0.0, 1.0)
    # Condensed form for scipy
    condensed = dist_matrix[np.triu_indices(n_features, k=1)]

    # Agglomerative clustering
    Z = linkage(condensed, method='average')
    cluster_labels = fcluster(Z, t=1.0 - corr_threshold, criterion='distance')

    # For each cluster, pick the medoid (most central member in distance space)
    representative_indices = []
    cluster_members_dict = {}
    cluster_rep_dict = {}
    unique_clusters = np.unique(cluster_labels)
    for cid in unique_clusters:
        members = np.where(cluster_labels == cid)[0]
        cluster_members_dict[cid] = members.tolist()
        if len(members) == 1:
            best = members[0]
        else:
            sub_dist = dist_matrix[np.ix_(members, members)]
            best = members[np.argmin(sub_dist.sum(axis=1))]
        cluster_rep_dict[cid] = best
        representative_indices.append(best)

    representative_indices = np.sort(representative_indices)

    lag_info = f"max_lag={max_lag}"
    if lag_penalty_lambda is not None:
        lag_info += f", λ={lag_penalty_lambda}"
    print(f"\nStage 1 — Redundancy clustering (lagged Spearman, {lag_info}):")
    print(f"  {n_features} features → {len(unique_clusters)} clusters (threshold |corr| > {corr_threshold})")
    print(f"  Representative indices (local): {representative_indices.tolist()}")
    for cid in unique_clusters:
        members = cluster_members_dict[cid]
        rep = cluster_rep_dict[cid]
        print(f"    Cluster {cid}: members {members}, representative {rep} (medoid)")

    return representative_indices, cluster_labels, cluster_members_dict, cluster_rep_dict


# ---------------------------------------------------------------------------
# Stage 2: AE masking importance
# ---------------------------------------------------------------------------

def _block_permute(arr, block_size, rng=None):
    """Permute an array in contiguous blocks to preserve local temporal structure."""
    if rng is None:
        rng = np.random.RandomState()
    n = len(arr)
    n_blocks = max(1, n // block_size)
    blocks = np.array_split(arr, n_blocks)
    perm = rng.permutation(len(blocks))
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
        train_data: array (timesteps, n_features) — only the kept features
        representative_indices: feature column indices (into train_data) to evaluate
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

    # Robust per-feature scaling (median / IQR) so ΔMSE reflects dynamics, not scale
    medians = np.median(data, axis=0)
    q75, q25 = np.percentile(data, [75, 25], axis=0)
    iqrs = q75 - q25
    iqrs[iqrs == 0] = 1.0
    data = (data - medians) / iqrs

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
                corrupted[si, :, fi] = _block_permute(corrupted[si, :, fi], block_size=max(1, sequence_length // 4), rng=rng)
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

def perform_feature_selection(train_data, n_features, sequence_length, device,
                              corr_threshold=0.9, importance_percentile=50,
                              lag_penalty_lambda=None):
    """Four-stage unsupervised feature selection on training data only.

    Stage 0: Drop static features (std == 0).
    Stage 1: Lagged Spearman correlation clustering → group redundant features.
    Stage 2: LSTM-AE masking importance → score cluster representatives.
    Stage 3: Group-aware encoder assignment → build encoder groups.

    Args:
        train_data: Training array of shape (timesteps, n_features)
        n_features: Total number of features
        sequence_length: Window length used for sequences
        device: torch device (for AE training)
        corr_threshold: |correlation| above which features are clustered together.
            Note: lagged correlation inflates similarities vs lag-0; consider
            increasing to 0.92-0.95 when using the default lagged method.
        importance_percentile: Percentile cutoff for importance scores; clusters
            with representative importance >= this percentile get their own
            encoder group.  Lower clusters are merged into a catch-all group.
        lag_penalty_lambda: optional decay constant for the lag penalty
            w(tau) = exp(-|tau| / lambda).  None disables it (all lags equal).

    Returns:
        encoder_groups: list[list[int]] — each inner list is original feature indices
                        for one encoder. Ordered by importance (most important first).
                        Last group is the catch-all "remaining" group (if non-empty).
        dropped_feature_indices: list[int] — static features that were dropped
    """
    # Stage 0: handle static features
    kept_indices, dropped_indices, static_binary_indices = _drop_static_features(train_data)
    dropped_feature_indices = dropped_indices.tolist()

    if len(kept_indices) == 0 and len(static_binary_indices) == 0:
        print("Warning: All features are static continuous. No encoder groups created.")
        return [], dropped_feature_indices

    # Stages 1-3 operate on dynamic (non-static) features only
    if len(kept_indices) > 0:
        kept_data = train_data[:, kept_indices]

        # Stage 1: redundancy clustering (on kept features only)
        representative_local, cluster_labels, cluster_members_dict, cluster_rep_dict = \
            _compute_redundancy_clusters(kept_data, corr_threshold,
                                         max_lag=sequence_length,
                                         lag_penalty_lambda=lag_penalty_lambda)

        # Stage 2: AE masking importance on representatives
        importance_scores = _compute_masking_importance(
            kept_data, representative_local, sequence_length, device
        )

        # Build mapping: representative local index → importance score
        rep_to_importance = {}
        for i, rep_local in enumerate(representative_local):
            rep_to_importance[rep_local] = importance_scores[i]

        # Map each cluster to its representative's importance score
        cluster_importance = {}
        cluster_representative = {}
        for cid in cluster_members_dict:
            rep = cluster_rep_dict[cid]
            cluster_representative[cid] = rep
            cluster_importance[cid] = rep_to_importance[rep]

        # Stage 3: group-aware encoder assignment
        scores = np.array(list(cluster_importance.values()))
        cids = list(cluster_importance.keys())
        threshold = np.percentile(scores, importance_percentile)

        high_clusters = []  # (cid, importance) pairs for clusters above threshold
        low_clusters = []
        for cid in cids:
            if cluster_importance[cid] >= threshold:
                high_clusters.append((cid, cluster_importance[cid]))
            else:
                low_clusters.append((cid, cluster_importance[cid]))

        # Sort high-importance clusters by descending importance
        high_clusters.sort(key=lambda x: x[1], reverse=True)

        encoder_groups = []
        # Each high-importance cluster becomes its own encoder group (original indices)
        for cid, imp in high_clusters:
            local_members = cluster_members_dict[cid]
            original_members = sorted(kept_indices[m] for m in local_members)
            encoder_groups.append(original_members)

        # Merge all low-importance clusters into one catch-all group
        catch_all = []
        for cid, imp in low_clusters:
            local_members = cluster_members_dict[cid]
            catch_all.extend(kept_indices[m] for m in local_members)
        if catch_all:
            encoder_groups.append(sorted(catch_all))
    else:
        encoder_groups = []
        catch_all = []
        high_clusters = []
        low_clusters = []

    # Append static binary features as a dedicated sentinel group
    if len(static_binary_indices) > 0:
        encoder_groups.append(sorted(static_binary_indices.tolist()))

    # --- Assertions ---
    all_assigned = set()
    for group in encoder_groups:
        group_set = set(group)
        assert len(group_set & all_assigned) == 0, \
            f"Encoder groups are not disjoint! Overlap: {group_set & all_assigned}"
        all_assigned |= group_set
    expected = set(kept_indices.tolist()) | set(static_binary_indices.tolist())
    assert all_assigned == expected, \
        (f"Union of encoder groups != kept + static binary features.\n"
         f"  Missing: {expected - all_assigned}\n"
         f"  Extra:   {all_assigned - expected}")

    # --- Logging ---
    n_high = len(high_clusters)
    n_low = len(low_clusters)
    has_catchall = len(catch_all) > 0
    has_sentinel = len(static_binary_indices) > 0

    print(f"\nStage 3 — Group-aware encoder assignment:")
    if len(kept_indices) > 0:
        print(f"  Importance threshold (percentile {importance_percentile}): {threshold:.6f}")
        print(f"  {n_high} high-importance cluster(s) → {n_high} individual encoder group(s)")
        print(f"  {n_low} low-importance cluster(s) → {'1 catch-all group' if has_catchall else 'no catch-all group (empty)'}")
    if has_sentinel:
        print(f"  1 static-binary sentinel group ({len(static_binary_indices)} features)")
    print(f"  Total encoder groups: {len(encoder_groups)}")
    for i, group in enumerate(encoder_groups):
        if has_sentinel and i == len(encoder_groups) - 1:
            label = "static-binary sentinel"
        elif has_catchall and i == len(encoder_groups) - (2 if has_sentinel else 1):
            label = "catch-all"
        else:
            label = f"important #{i+1}"
        print(f"    Group {i} ({label}): {len(group)} features → {group}")

    print(f"\n  Dropped static continuous features ({len(dropped_feature_indices)}): {dropped_feature_indices}")

    return encoder_groups, dropped_feature_indices


def split_features_by_groups(data, encoder_groups):
    """Split data array into per-group arrays.

    Args:
        data: numpy array of shape (timesteps, features)
        encoder_groups: list of lists of feature indices

    Returns:
        list of numpy arrays, one per group
    """
    return [data[:, group] for group in encoder_groups]
