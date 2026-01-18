# -*- coding: utf-8 -*-
"""Configuration settings for LSTM VAE stacked SMAP/MSL anomaly detection."""

import os
import torch

# Dataset configuration for NASA SMAP/MSL
# DATASET_TYPE can be "SMAP" or "MSL"
DATASET_TYPE = "MSL"  # Change to "MSL" for Mars Science Laboratory dataset
DRIVE = "/mnt/c/Users/beren/Desktop/DSLAB/datasets/SMAP_MSL/"

# Available channels for SMAP and MSL (from labeled_anomalies.csv)
# You can change CHANNEL to any valid channel ID
CHANNEL = "M-1"  # Example: P-1, S-1, E-1, etc. for SMAP; M-1, C-1, T-4, etc. for MSL

# Paths for SMAP/MSL dataset
TRAIN_DATASET = os.path.join(DRIVE, "data/data", "train", f"{CHANNEL}.npy")
TEST_DATASET = os.path.join(DRIVE, "data/data", "test", f"{CHANNEL}.npy")
LABELS_FILE = os.path.join(DRIVE, "labeled_anomalies.csv")

# Model hyperparameters (defaults)
SEQUENCE_LENGTH = 30
INPUT_DIM = 1
HIDDEN_DIM = 128
LATENT_DIM = 32
NUM_LAYERS = 1
BATCH_SIZE = 32

# Training settings
LEARNING_RATE = 1e-3
NUM_EPOCHS = 256
EARLY_STOP_PATIENCE = 10

# Optuna settings
USE_OPTUNA = True
N_OPTUNA_TRIALS = 50

# Default hyperparameters (used when not using Optuna)
DEFAULT_PARAMS = {
    'top_weight': 0.7,
    'hidden_dim': 128,
    'latent_dim': 32,
    'num_layers': 1,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'percentile_threshold': 90,
    'kl_weight': 0.1
}

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
