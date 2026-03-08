# -*- coding: utf-8 -*-
"""Configuration settings for LSTM VAE stacked anomaly detection."""

import os
import torch

# =============================================================================
# SMAP/MSL Dataset Configuration
# =============================================================================
# DATASET_TYPE can be "SMAP" or "MSL"
DATASET_TYPE = "SMAP"  # Change to "MSL" for Mars Science Laboratory dataset
SMAP_DRIVE = "/mnt/c/Users/beren/Desktop/DSLAB/datasets/SMAP_MSL/"

# Available channels for SMAP and MSL (from labeled_anomalies.csv)
# You can change CHANNEL to any valid channel ID
CHANNEL = "E-1"  # Example: P-1, S-1, E-1, etc. for SMAP; M-1, C-1, T-4, etc. for MSL

# Paths for SMAP/MSL dataset
SMAP_TRAIN_DATASET = os.path.join(SMAP_DRIVE, "data/data", "train", f"{CHANNEL}.npy")
SMAP_TEST_DATASET = os.path.join(SMAP_DRIVE, "data/data", "test", f"{CHANNEL}.npy")
LABELS_FILE = os.path.join(SMAP_DRIVE, "labeled_anomalies.csv")

# Legacy alias for backward compatibility
DRIVE = SMAP_DRIVE

# =============================================================================
# SMD (Server Machine Dataset) Configuration
# =============================================================================
SMD_DRIVE = "/mnt/c/Users/beren/Desktop/DSLAB/datasets/ServerMachineDataset/"

# Machine identifier (e.g., "machine-1-1.txt", "machine-2-1.txt")
MACHINE = "machine-1-1.txt"

# Paths for SMD dataset
SMD_TRAIN_DATASET = os.path.join(SMD_DRIVE, "train", MACHINE)
SMD_TEST_DATASET = os.path.join(SMD_DRIVE, "test", MACHINE)
SMD_TEST_LABEL_DATASET = os.path.join(SMD_DRIVE, "test_label", MACHINE)

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

# Performance toggles

# Enables cuDNN auto-tuner to pick the fastest convolution/LSTM kernels.
CUDNN_BENCHMARK = True
# Enables automatic mixed precision during training.
USE_AMP = False
# Enables torch.compile for graph-level optimization.
USE_TORCH_COMPILE = False
# Number of DataLoader worker processes (0 = main process only).
DATALOADER_WORKERS = 0
# Pins CPU memory for faster host-to-device transfers.
PIN_MEMORY = False

# Optuna settings
USE_OPTUNA = False
N_OPTUNA_TRIALS = 20

DEFAULT_PARAMS_SMD = { # For SMD
    'hidden_dim': 256,
    'latent_dim': 13,
    'num_layers': 2,
    'learning_rate': 0.0020434554984161395,
    'batch_size': 896,
    'percentile_threshold': 90,
    'kl_weight': 0.005109860554090236,
    'use_scheduler': False,
    'corr_threshold': 0.8239571182457097,
    'importance_percentile': 80,
    'lag_penalty_lambda': 0,
}

DEFAULT_PARAMS = { # For SMAP
    'hidden_dim': 96,
    'latent_dim': 13,
    'num_layers': 2,
    'learning_rate': 0.0016434424020119224,
    'batch_size': 640,
    'percentile_threshold': 94,
    'kl_weight': 0.0067481364876198595,
    'use_scheduler': False,
    'scheduler_patience': 5,
    'scheduler_factor': 0.1
}

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
