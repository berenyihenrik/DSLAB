# -*- coding: utf-8 -*-
"""Data loading and preprocessing functions for SMAP/MSL and SMD datasets."""

import os
import ast
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# =============================================================================
# SMAP/MSL Dataset Functions
# =============================================================================

def load_smap_msl_data(channel, drive_path, dataset_type=None):
    """
    Load SMAP or MSL dataset for a given channel.
    
    Args:
        channel: Channel ID (e.g., 'P-1', 'M-1')
        drive_path: Base path to the SMAP_MSL dataset
        dataset_type: Optional filter for 'SMAP' or 'MSL'. If None, auto-detect from labels.
    
    Returns:
        train_data: numpy array of training data
        test_data: numpy array of test data
        true_anomalies: numpy array of binary anomaly labels for test data
    """
    # Load train and test data
    train_path = os.path.join(drive_path, "data/data", "train", f"{channel}.npy")
    test_path = os.path.join(drive_path, "data/data", "test", f"{channel}.npy")
    labels_path = os.path.join(drive_path, "labeled_anomalies.csv")
    
    print(f"Loading training data from: {train_path}")
    print(f"Loading test data from: {test_path}")
    
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Load labels and create binary anomaly array
    labels_df = pd.read_csv(labels_path)
    
    # Filter by dataset type if specified
    if dataset_type:
        labels_df = labels_df[labels_df['spacecraft'] == dataset_type]
    
    # Get the anomaly sequences for this channel
    channel_labels = labels_df[labels_df['chan_id'] == channel]
    
    if len(channel_labels) == 0:
        raise ValueError(f"Channel {channel} not found in labels file")
    
    # Get the number of values (test data length) and anomaly sequences
    num_values = channel_labels['num_values'].iloc[0]
    anomaly_sequences_str = channel_labels['anomaly_sequences'].iloc[0]
    
    # Parse the anomaly sequences string to list of [start, end] pairs
    anomaly_sequences = ast.literal_eval(anomaly_sequences_str)
    
    # Create binary anomaly labels
    true_anomalies = np.zeros(num_values, dtype=int)
    for seq in anomaly_sequences:
        start, end = seq
        true_anomalies[start:end+1] = 1
    
    # Ensure the labels match the test data length
    if len(true_anomalies) != len(test_data):
        print(f"Warning: Label length ({len(true_anomalies)}) differs from test data length ({len(test_data)})")
        # Adjust to match test data length
        if len(true_anomalies) > len(test_data):
            true_anomalies = true_anomalies[:len(test_data)]
        else:
            # Pad with zeros if labels are shorter
            padded = np.zeros(len(test_data), dtype=int)
            padded[:len(true_anomalies)] = true_anomalies
            true_anomalies = padded
    
    print(f"Total anomalous points: {np.sum(true_anomalies)} / {len(true_anomalies)}")
    print(f"Anomaly sequences: {anomaly_sequences}")
    
    return train_data, test_data, true_anomalies


def get_available_channels(drive_path, dataset_type=None):
    """
    Get list of available channels from the labels file.
    
    Args:
        drive_path: Base path to the SMAP_MSL dataset
        dataset_type: Optional filter for 'SMAP' or 'MSL'
    
    Returns:
        List of available channel IDs
    """
    labels_path = os.path.join(drive_path, "labeled_anomalies.csv")
    labels_df = pd.read_csv(labels_path)
    
    if dataset_type:
        labels_df = labels_df[labels_df['spacecraft'] == dataset_type]
    
    channels = labels_df['chan_id'].unique().tolist()
    return channels


# =============================================================================
# SMD (Server Machine Dataset) Functions
# =============================================================================

def load_smd_data(machine, drive_path):
    """
    Load SMD (Server Machine Dataset) for a given machine.
    
    Args:
        machine: Machine identifier (e.g., 'machine-1-1.txt')
        drive_path: Base path to the ServerMachineDataset
    
    Returns:
        train_data: numpy array of training data
        test_data: numpy array of test data
        true_anomalies: numpy array of binary anomaly labels for test data
    """
    train_path = os.path.join(drive_path, "train", machine)
    test_path = os.path.join(drive_path, "test", machine)
    label_path = os.path.join(drive_path, "test_label", machine)
    
    print(f"Loading training data from: {train_path}")
    print(f"Loading test data from: {test_path}")
    print(f"Loading labels from: {label_path}")
    
    # Load CSV files (no header)
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    true_anomalies = pd.read_csv(label_path, header=None)[0].to_numpy()
    
    train_data = train_df.values
    test_data = test_df.values
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Total anomalous points: {np.sum(true_anomalies)} / {len(true_anomalies)}")
    print(f"Anomaly rate: {np.sum(true_anomalies) / len(true_anomalies) * 100:.2f}%")
    
    return train_data, test_data, true_anomalies


def get_available_machines(drive_path):
    """
    Get list of available machines from the SMD dataset directory.
    
    Args:
        drive_path: Base path to the ServerMachineDataset
    
    Returns:
        List of available machine filenames
    """
    train_path = os.path.join(drive_path, "train")
    if os.path.exists(train_path):
        machines = [f for f in os.listdir(train_path) if f.endswith('.txt')]
        machines.sort()
        return machines
    return []


# =============================================================================
# SWaT (Secure Water Treatment) Dataset Functions
# =============================================================================

def load_swat_data(normal_path, attack_path):
    """Load SWaT normal and attack CSV files.

    SWaT quirks handled here:
        1. Strip leading/trailing whitespace from all column names.
        2. Normalize the known label typo "A ttack" -> "Attack".
        3. Drop Timestamp column and return only numeric feature columns.

    Args:
        normal_path: Path to SWaT normal CSV.
        attack_path: Path to SWaT attack CSV.

    Returns:
        train_data: numpy array of normal features (rows, 51)
        test_data: numpy array of attack features (rows, 51)
        true_anomalies: binary numpy array for attack labels (0/1)
    """
    print(f"Loading SWaT normal data from: {normal_path}")
    print(f"Loading SWaT attack data from: {attack_path}")

    normal_df = pd.read_csv(normal_path, low_memory=False)
    attack_df = pd.read_csv(attack_path, low_memory=False)

    # Header cleanup is required because attack CSV has leading spaces in
    # multiple columns (e.g. " MV101", " Timestamp").
    normal_df.columns = normal_df.columns.str.strip()
    attack_df.columns = attack_df.columns.str.strip()

    label_col = "Normal/Attack"
    timestamp_col = "Timestamp"
    if label_col not in normal_df.columns or label_col not in attack_df.columns:
        raise ValueError("SWaT label column 'Normal/Attack' not found after header cleanup")

    # Normalize typo and spacing variations in labels.
    attack_labels = (
        attack_df[label_col]
        .astype(str)
        .str.strip()
        .replace({"A ttack": "Attack"})
    )
    true_anomalies = (attack_labels == "Attack").astype(int).to_numpy()

    # Keep only feature columns (drop timestamp + label).
    feature_cols = [c for c in normal_df.columns if c not in {timestamp_col, label_col}]
    if len(feature_cols) == 0:
        raise ValueError("No SWaT feature columns found after removing Timestamp/Normal/Attack")

    normal_features = normal_df[feature_cols].apply(pd.to_numeric, errors='coerce')
    attack_features = attack_df[feature_cols].apply(pd.to_numeric, errors='coerce')

    train_data = normal_features.to_numpy(dtype=np.float32)
    test_data = attack_features.to_numpy(dtype=np.float32)

    if np.isnan(train_data).any() or np.isnan(test_data).any():
        raise ValueError("NaN values encountered while loading SWaT numeric feature columns")

    print(f"SWaT train data shape: {train_data.shape}")
    print(f"SWaT test data shape: {test_data.shape}")
    print(f"SWaT feature count: {len(feature_cols)}")
    print(f"SWaT anomalous points: {np.sum(true_anomalies)} / {len(true_anomalies)}")
    print(f"SWaT anomaly rate: {np.mean(true_anomalies) * 100:.2f}%")

    return train_data, test_data, true_anomalies


class GroupedSequenceDataset(Dataset):
    """Lazy grouped sequence dataset to avoid materializing all windows.

    Each item is a tuple with one array per encoder group. For index ``i``
    we return windows ``[i:i+sequence_length]`` from each group.
    """

    def __init__(self, data_groups, sequence_length):
        if not data_groups:
            raise ValueError("data_groups must contain at least one group")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")

        n_rows = data_groups[0].shape[0]
        for group in data_groups:
            if group.shape[0] != n_rows:
                raise ValueError("All groups in data_groups must have matching time dimension")

        self.data_groups = data_groups
        self.sequence_length = sequence_length
        self.n_samples = n_rows - sequence_length + 1
        if self.n_samples <= 0:
            raise ValueError("sequence_length is larger than number of timesteps")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n_samples:
            raise IndexError("GroupedSequenceDataset index out of range")
        return tuple(group[idx:idx + self.sequence_length] for group in self.data_groups)


def standardize_continuous_features(train_data, other_arrays, continuous_indices):
    """Z-score continuous features using train-only statistics.

    Args:
        train_data: training array (timesteps, features), transformed in place.
        other_arrays: list of arrays to transform using train stats.
        continuous_indices: iterable of column indices to standardize.

    Returns:
        train_data: standardized training array (same object, modified in place)
        transformed_others: list of transformed arrays (same objects, modified)
    """
    continuous_indices = list(continuous_indices)
    if len(continuous_indices) == 0:
        return train_data, other_arrays

    means = train_data[:, continuous_indices].mean(axis=0)
    stds = train_data[:, continuous_indices].std(axis=0)
    stds[stds == 0] = 1.0

    train_data[:, continuous_indices] = (train_data[:, continuous_indices] - means) / stds
    for arr in other_arrays:
        arr[:, continuous_indices] = (arr[:, continuous_indices] - means) / stds

    return train_data, other_arrays


# =============================================================================
# Common Preprocessing Functions
# =============================================================================

def detect_binary_features(data):
    """Detect features with ≤2 unique values (two-valued / binary features).

    This includes true {0,1} binary features as well as two-valued features on
    other intervals (e.g. {-1, 1}, {0, 100}). All of these are semantically
    discrete state switches and should be handled with BCE loss after
    normalizing to [0, 1] via normalize_binary_features().

    Args:
        data: numpy array of shape (timesteps, features)

    Returns:
        binary_indices: set of feature indices that are two-valued
    """
    binary_indices = set()
    for i in range(data.shape[1]):
        if len(np.unique(data[:, i])) <= 2:
            binary_indices.add(i)
    return binary_indices


def normalize_binary_features(train_data, test_data, binary_indices):
    """Remap binary features to [0, 1] for BCE loss compatibility.

    Uses min/max from training data to normalize both datasets.
    Features already in {0, 1} are unaffected.
    """
    for idx in binary_indices:
        col_min = train_data[:, idx].min()
        col_max = train_data[:, idx].max()
        if col_max > col_min:
            train_data[:, idx] = (train_data[:, idx] - col_min) / (col_max - col_min)
            test_data[:, idx] = (test_data[:, idx] - col_min) / (col_max - col_min)
    return train_data, test_data


def preprocess_data(data):
    """
    Handle NaN values in data by interpolation and filling.
    
    Args:
        data: numpy array of shape (timesteps, features)
    
    Returns:
        Preprocessed numpy array with NaN values handled
    """
    df = pd.DataFrame(data)
    df.interpolate(inplace=True)
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df.values.astype(np.float32)


def create_sequences(data, sequence_length):
    """
    Create sequences from time series data.
    
    Args:
        data: numpy array of shape (timesteps, features)
        sequence_length: Length of each sequence
    
    Returns:
        List of sequences
    """
    sequences = []
    for i in range(data.shape[0] - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return sequences


def create_grouped_sequences(data_groups, sequence_length):
    """Create sequences from per-group time series data.
    
    Args:
        data_groups: list of numpy arrays, one per encoder group.
                     Each array has shape (timesteps, group_features).
        sequence_length: Length of each sequence window.
    
    Returns:
        List of tuples, where each tuple contains one sequence array per group.
        Each element is shape (sequence_length, group_features).
    """
    n_samples = data_groups[0].shape[0] - sequence_length + 1
    sequences = []
    for i in range(n_samples):
        sample = tuple(dg[i:i + sequence_length] for dg in data_groups)
        sequences.append(sample)
    return sequences
