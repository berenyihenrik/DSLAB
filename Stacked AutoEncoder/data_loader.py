# -*- coding: utf-8 -*-
"""Data loading and preprocessing functions for SMAP/MSL and SMD datasets."""

import os
import ast
import numpy as np
import pandas as pd


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
# Common Preprocessing Functions
# =============================================================================

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


def create_combined_sequences(data_top, data_remaining, sequence_length):
    """
    Create combined sequences for both feature groups.
    
    Args:
        data_top: numpy array for top features
        data_remaining: numpy array for remaining features
        sequence_length: Length of each sequence
    
    Returns:
        List of tuples (sequence_top, sequence_remaining)
    """
    sequences_top = []
    sequences_remaining = []
    for i in range(data_top.shape[0] - sequence_length + 1):
        sequences_top.append(data_top[i:i + sequence_length])
        sequences_remaining.append(data_remaining[i:i + sequence_length])
    return list(zip(sequences_top, sequences_remaining))
