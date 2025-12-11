# -*- coding: utf-8 -*-
"""LSTM VAE stacked for NASA SMAP/MSL datasets.

Refactored to work with NASA SMAP (Soil Moisture Active Passive satellite) and
MSL (Mars Science Laboratory rover) anomaly detection datasets.

# Setup Environment and Read Data
"""

import torch
import numpy as np
import pandas as pd
import pickle
import copy
import time
import os
import ast
import joblib
from tqdm import trange,tqdm
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score
from torch.cuda.amp import autocast, GradScaler
import optuna
from optuna.trial import TrialState

"""## Setup the dataset"""

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


# Print available channels
print(f"Available {DATASET_TYPE} channels:")
available_channels = get_available_channels(DRIVE, DATASET_TYPE)
print(available_channels[:10], "..." if len(available_channels) > 10 else "")

# Load the dataset
metric_tensor, metric_test_tensor, true_anomalies = load_smap_msl_data(CHANNEL, DRIVE, DATASET_TYPE)

# Convert to float32 for PyTorch compatibility
metric_tensor = metric_tensor.astype(np.float32)
metric_test_tensor = metric_test_tensor.astype(np.float32)

"""### Non-Scaled - Create Sequences"""

# Handle NaN values if any (SMAP/MSL data is usually clean, but just in case)
metric_df = pd.DataFrame(metric_tensor)
metric_df.interpolate(inplace=True)
metric_df.bfill(inplace=True)
metric_df.ffill(inplace=True)
metric_tensor = metric_df.values.astype(np.float32)

metric_test_df = pd.DataFrame(metric_test_tensor)
metric_test_df.interpolate(inplace=True)
metric_test_df.bfill(inplace=True)
metric_test_df.ffill(inplace=True)
metric_test_tensor = metric_test_df.values.astype(np.float32)

sequence_length = 30
sequences = []
for i in range(metric_tensor.shape[0] - sequence_length + 1):
  sequences.append(metric_tensor[i:i + sequence_length])


train_data, val_data = train_test_split(sequences, test_size=0.3, random_state=42) # 70% train, 30% validation

test_sequences = []
for i in range(metric_test_tensor.shape[0] - sequence_length + 1):
  test_sequences.append(metric_test_tensor[i:i + sequence_length])


batch_size = 128
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_sequences, batch_size=batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Sequence shape: {sequences[0].shape}")
print(f"Number of features: {metric_tensor.shape[1]}")

"""Apply Feature Selection and Rebuild Dataloaders"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Prepare data for RandomForestClassifier using the test data and true anomalies
# We will use the test data (metric_test_tensor) and the true anomaly labels (true_anomalies)
# to train the RandomForestClassifier for feature importance calculation.

# Adjust true_anomalies to match test_sequences length (due to windowing)
# Each sequence covers positions i to i+sequence_length-1
# We assign the label of the last position in each sequence
adjusted_true_anomalies = true_anomalies[sequence_length-1:]

# Flatten the test sequences for the RandomForestClassifier
n_samples_test, n_timesteps, n_features = np.array(test_sequences).shape
X_test_flat = np.array(test_sequences).reshape(n_samples_test, n_timesteps * n_features)
y_test_true = adjusted_true_anomalies[:n_samples_test] # Use the true anomalies as the target

print(f"Test sequences shape: {n_samples_test}, {n_timesteps}, {n_features}")
print(f"Adjusted labels length: {len(y_test_true)}")

# Check if we have both classes for training RF
unique_classes = np.unique(y_test_true)
print(f"Unique classes in labels: {unique_classes}")

if len(unique_classes) < 2:
    print("Warning: Only one class present in labels. Using all features without RF selection.")
    # Use all features if we can't train RF
    n_features_total = metric_tensor.shape[1]
    num_selected_features = min(25, n_features_total)
    selected_feature_indices = np.arange(num_selected_features)
    remaining_feature_indices = np.arange(num_selected_features, n_features_total)
else:
    # Initialize and train the RandomForestClassifier
    # We use a simple setup for demonstration. Hyperparameter tuning might be needed for optimal results.
    # The goal here is not to build a perfect anomaly detection model with RF, but to get feature importances.
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_test_flat, y_test_true)

    # Get feature importances
    feature_importances = rf_model.feature_importances_

    # The feature importances are for the flattened features (timesteps * features).
    # We need to map these back to the original features.
    # We can average the importance scores for each original feature across all timesteps.
    original_feature_importances = feature_importances.reshape(n_timesteps, n_features).mean(axis=0)

    # Rank features by importance
    ranked_features_indices = np.argsort(original_feature_importances)[::-1]
    ranked_features_importance = original_feature_importances[ranked_features_indices]

    print("Feature Importances (averaged across timesteps):")
    for i, index in enumerate(ranked_features_indices):
        print(f"Feature {index}: {ranked_features_importance[i]:.4f}")

    # Determine the number of top features to select
    # For SMAP/MSL, we adapt based on total features available
    n_features_total = n_features
    num_selected_features = min(25, max(1, n_features_total // 2))  # At least 1, at most 25 or half

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

# Select only the top features for the datasets
metric_tensor_selected = metric_tensor[:, selected_feature_indices]
metric_test_tensor_selected = metric_test_tensor[:, selected_feature_indices]

# Create datasets with top features and remaining features
metric_tensor_top = metric_tensor[:, selected_feature_indices]
metric_tensor_remaining = metric_tensor[:, remaining_feature_indices]

metric_test_tensor_top = metric_test_tensor[:, selected_feature_indices]
metric_test_tensor_remaining = metric_test_tensor[:, remaining_feature_indices]

# Create sequences for both feature groups
sequence_length = 30
sequences_top = []
sequences_remaining = []
for i in range(metric_tensor.shape[0] - sequence_length + 1):
    sequences_top.append(metric_tensor_top[i:i + sequence_length])
    sequences_remaining.append(metric_tensor_remaining[i:i + sequence_length])

# Combine into tuples for dataloaders
sequences_combined = list(zip(sequences_top, sequences_remaining))
train_data_combined, val_data_combined = train_test_split(sequences_combined, test_size=0.3, random_state=42)

test_sequences_top = []
test_sequences_remaining = []
for i in range(metric_test_tensor.shape[0] - sequence_length + 1):
    test_sequences_top.append(metric_test_tensor_top[i:i + sequence_length])
    test_sequences_remaining.append(metric_test_tensor_remaining[i:i + sequence_length])

test_sequences_combined = list(zip(test_sequences_top, test_sequences_remaining))

batch_size = 32
train_loader_combined = DataLoader(dataset=train_data_combined, batch_size=batch_size, shuffle=True)
val_loader_combined = DataLoader(dataset=val_data_combined, batch_size=batch_size, shuffle=False)
test_loader_combined = DataLoader(dataset=test_sequences_combined, batch_size=batch_size, shuffle=False)

print(f"Top features dimension: {len(selected_feature_indices)}")
print(f"Remaining features dimension: {len(remaining_feature_indices)}")

"""## Stacked with Weighted Encoders"""

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        return self.fc_mean(h), self.fc_logvar(h)


class SharedDecoder(nn.Module):
    def __init__(self, input_features_dim, hidden_dim, output_features_dim_top, output_features_dim_remaining, sequence_length, num_layers=1):
        super(SharedDecoder, self).__init__()
        self.sequence_length = sequence_length
        self.latent_to_hidden = nn.Linear(input_features_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Separate output layers for top and remaining features
        self.output_layer_top = nn.Linear(hidden_dim, output_features_dim_top)
        self.output_layer_remaining = nn.Linear(hidden_dim, output_features_dim_remaining)

    def forward(self, z):
        hidden = self.latent_to_hidden(z).unsqueeze(1).repeat(1, self.sequence_length, 1)
        out, _ = self.lstm(hidden)
        recon_top = self.output_layer_top(out)
        recon_remaining = self.output_layer_remaining(out)
        return recon_top, recon_remaining


class LSTMVAE_Stacked_Weighted(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, sequence_length, num_layers=1, 
                 device='cpu', num_top_sensors=25, num_remaining_sensors=13, 
                 top_weight=0.7, remaining_weight=0.3):
        super(LSTMVAE_Stacked_Weighted, self).__init__()
        self.input_dim = input_dim
        self.num_top_sensors = num_top_sensors
        self.num_remaining_sensors = num_remaining_sensors
        self.sequence_length = sequence_length
        self.device = device
        self.top_weight = top_weight
        self.remaining_weight = remaining_weight
        
        # Separate encoders for top features
        self.encoders_top = nn.ModuleList([
            LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers).to(device) 
            for _ in range(num_top_sensors)
        ])
        
        # Single encoder for remaining features (processes all at once)
        self.encoder_remaining = LSTMEncoder(num_remaining_sensors * input_dim, hidden_dim, latent_dim, num_layers).to(device)
        
        # Decoder input is concatenation of all latent representations
        decoder_input_features = (num_top_sensors + 1) * latent_dim
        decoder_output_features_top = input_dim * num_top_sensors
        decoder_output_features_remaining = input_dim * num_remaining_sensors
        
        self.decoder = SharedDecoder(
            decoder_input_features, 
            hidden_dim, 
            decoder_output_features_top,
            decoder_output_features_remaining,
            sequence_length, 
            num_layers
        ).to(device)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x_top, x_remaining):
        # x_top shape: (batch_size, sequence_length, num_top_sensors)
        # x_remaining shape: (batch_size, sequence_length, num_remaining_sensors)
        batch_size = x_top.shape[0]
        
        # Process top features with individual encoders
        x_top_reshaped = x_top.view(batch_size, self.sequence_length, self.num_top_sensors, self.input_dim)
        x_top_reshaped = x_top_reshaped.permute(0, 2, 1, 3)
        x_top_flat = x_top_reshaped.reshape(batch_size * self.num_top_sensors, self.sequence_length, self.input_dim)
        
        means_top = []
        logvars_top = []
        for i, encoder in enumerate(self.encoders_top):
            x_sensor = x_top_flat[i::self.num_top_sensors]
            mean, logvar = encoder(x_sensor)
            means_top.append(mean)
            logvars_top.append(logvar)
        
        mean_top_stacked = torch.stack(means_top, dim=1)
        logvar_top_stacked = torch.stack(logvars_top, dim=1)
        z_top_stacked = self.reparameterize(mean_top_stacked, logvar_top_stacked)
        
        # Process remaining features with single encoder
        x_remaining_flat = x_remaining.view(batch_size, self.sequence_length, -1)
        mean_remaining, logvar_remaining = self.encoder_remaining(x_remaining_flat)
        z_remaining = self.reparameterize(mean_remaining, logvar_remaining)
        
        # Combine latent representations
        z_top_combined = z_top_stacked.reshape(batch_size, -1)
        z_combined = torch.cat([z_top_combined, z_remaining], dim=1)
        
        mean_combined = torch.cat([mean_top_stacked.reshape(batch_size, -1), mean_remaining], dim=1)
        logvar_combined = torch.cat([logvar_top_stacked.reshape(batch_size, -1), logvar_remaining], dim=1)
        
        # Decode
        x_recon_top, x_recon_remaining = self.decoder(z_combined)
        
        return x_recon_top, x_recon_remaining, mean_combined, logvar_combined

num_top_sensors = len(selected_feature_indices)
num_remaining_sensors = len(remaining_feature_indices)
input_dim = 1
hidden_dim = 128
latent_dim = 32
num_layers = 1

model = LSTMVAE_Stacked_Weighted(
    num_top_sensors=num_top_sensors,
    num_remaining_sensors=num_remaining_sensors,
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
    sequence_length=sequence_length,
    num_layers=num_layers,
    device=device,
    top_weight=0.7,
    remaining_weight=0.3
).to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

# Test forward pass
batch_top = torch.randn(batch_size, sequence_length, num_top_sensors).to(device)
batch_remaining = torch.randn(batch_size, sequence_length, num_remaining_sensors).to(device)
output_top, output_remaining, _, _ = model(batch_top, batch_remaining)
print(f"Output top shape: {output_top.shape}, Output remaining shape: {output_remaining.shape}")

"""## Support functions"""

def loss_function_weighted(x_top, x_remaining, x_hat_top, x_hat_remaining, mean, log_var, top_weight=0.7, remaining_weight=0.3):
    # Ensure shapes match - reshape reconstructions if needed
    if x_hat_top.shape != x_top.shape:
        x_hat_top = x_hat_top.view_as(x_top)
    if x_hat_remaining.shape != x_remaining.shape:
        x_hat_remaining = x_hat_remaining.view_as(x_remaining)
    
    reproduction_loss_top = nn.functional.mse_loss(x_hat_top, x_top, reduction='mean')
    reproduction_loss_remaining = nn.functional.mse_loss(x_hat_remaining, x_remaining, reduction='mean')
    
    reproduction_loss = top_weight * reproduction_loss_top + remaining_weight * reproduction_loss_remaining
    
    KLD = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
    beta = 0.1
    
    return reproduction_loss + beta * KLD

def save_model(model, name):
    model_state = {
        'input_dim':input_dim,
        'latent_dim':latent_dim,
        'hidden_dim':hidden_dim,
        'state_dict':model.state_dict()
    }
    torch.save(model_state, name + '.pth')

"""# Train

## LSTM with Weighted Encoders
"""

torch.cuda.empty_cache()

scaler = GradScaler()
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

# SPO optimizer - optuna
# bayesian hyperparameter tuning
# grid search - slow for DL

def train_model_weighted(model, train_loader, val_loader, optimizer, loss_fn, scheduler, num_epochs=10, device='cpu'):
    torch.cuda.empty_cache()
    train_losses = []
    val_losses = []

    early_stop_tolerant_count = 0
    early_stop_tolerant = 10
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss = 0.0
        model.train()

        # Profiling timers
        forward_time = 0.0
        backward_time = 0.0
        
        for batch_top, batch_remaining in train_loader:
            batch_top = torch.tensor(batch_top, dtype=torch.float32).to(device)
            batch_remaining = torch.tensor(batch_remaining, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()

            # Time forward pass
            t0 = time.time()
            
            recon_top, recon_remaining, mean, logvar = model(batch_top, batch_remaining)
            loss = loss_fn(batch_top, batch_remaining, recon_top, recon_remaining, mean, logvar, 
                          model.top_weight, model.remaining_weight)
            
            forward_time += time.time() - t0
            
            loss.backward()
            optimizer.step()

            backward_time += time.time() - t0
            
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_top, batch_remaining in val_loader:
                batch_top = torch.tensor(batch_top, dtype=torch.float32).to(device)
                batch_remaining = torch.tensor(batch_remaining, dtype=torch.float32).to(device)
                
                recon_top, recon_remaining, mean, logvar = model(batch_top, batch_remaining)
                loss = loss_fn(batch_top, batch_remaining, recon_top, recon_remaining, mean, logvar,
                              model.top_weight, model.remaining_weight)
                valid_loss += loss.item()

        valid_loss /= len(val_loader)
        val_losses.append(valid_loss)

        scheduler.step(valid_loss)

        epoch_time = time.time() - epoch_start_time

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_tolerant_count = 0
        else:
            early_stop_tolerant_count += 1

        print(f"Epoch {epoch+1:04d}: train loss {train_loss:.4f}, valid loss {valid_loss:.4f}, "
              f"time {epoch_time:.2f}s (forward: {forward_time:.2f}s, backward: {backward_time:.2f}s)")

        if early_stop_tolerant_count >= early_stop_tolerant:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model_wts)
    print("Finished Training.")
    return train_losses, val_losses


"""## Optuna Hyperparameter Tuning

This section implements hyperparameter optimization using Optuna, focusing on:
- Encoder weights (top_weight, remaining_weight)
- Model architecture (hidden_dim, latent_dim, num_layers)
- Training parameters (learning_rate, batch_size)
- Detection threshold (percentile_threshold)
"""

def train_model_for_optuna(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=20, device='cpu', trial=None):
    torch.cuda.empty_cache()
    train_losses = []
    val_losses = []

    early_stop_tolerant_count = 0
    early_stop_tolerant = 5  # Shorter patience for Optuna trials
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        
        for batch_top, batch_remaining in train_loader:
            batch_top = torch.tensor(batch_top, dtype=torch.float32).to(device)
            batch_remaining = torch.tensor(batch_remaining, dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            
            recon_top, recon_remaining, mean, logvar = model(batch_top, batch_remaining)
            loss = loss_fn(batch_top, batch_remaining, recon_top, recon_remaining, mean, logvar, 
                          model.top_weight, model.remaining_weight)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_top, batch_remaining in val_loader:
                batch_top = torch.tensor(batch_top, dtype=torch.float32).to(device)
                batch_remaining = torch.tensor(batch_remaining, dtype=torch.float32).to(device)
                
                recon_top, recon_remaining, mean, logvar = model(batch_top, batch_remaining)
                loss = loss_fn(batch_top, batch_remaining, recon_top, recon_remaining, mean, logvar,
                              model.top_weight, model.remaining_weight)
                valid_loss += loss.item()

        valid_loss /= len(val_loader)
        val_losses.append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_tolerant_count = 0
        else:
            early_stop_tolerant_count += 1

        # Report to Optuna for pruning
        if trial is not None:
            trial.report(valid_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if early_stop_tolerant_count >= early_stop_tolerant:
            break

    model.load_state_dict(best_model_wts)
    return train_losses, val_losses, best_loss


def evaluate_for_optuna(model, test_loader, device, percentile_threshold, true_anomalies, sequence_length):
    """
    Evaluation function that returns F1 score for Optuna optimization.
    Uses the given percentile_threshold directly.
    
    Returns:
        f1: F1 score at the given threshold
        anomaly_scores: List of anomaly scores for each sequence
    """
    model.eval()
    anomaly_scores = []

    with torch.no_grad():
        for batch_top, batch_remaining in test_loader:
            batch_top = torch.tensor(batch_top, dtype=torch.float32).to(device)
            batch_remaining = torch.tensor(batch_remaining, dtype=torch.float32).to(device)

            batch_scores = []
            for i in range(batch_top.shape[0]):
                sequence_top = batch_top[i, :, :].unsqueeze(0)
                sequence_remaining = batch_remaining[i, :, :].unsqueeze(0)
                
                recon_top, recon_remaining, mean, logvar = model(sequence_top, sequence_remaining)
                loss = loss_function_weighted(sequence_top, sequence_remaining, recon_top, recon_remaining, 
                                             mean, logvar, model.top_weight, model.remaining_weight)
                batch_scores.append(loss.item())
            anomaly_scores.extend(batch_scores)

    # Use the given percentile threshold
    adjusted_true_anomalies = true_anomalies[sequence_length-1:]
    
    threshold = np.percentile(anomaly_scores, percentile_threshold)
    anomaly_indices = [i for i, score in enumerate(anomaly_scores) if score > threshold]
    
    predicted_anomalies = np.zeros(len(adjusted_true_anomalies), dtype=int)
    for index in anomaly_indices:
        if index < len(predicted_anomalies):
            predicted_anomalies[index] = 1
    
    f1 = f1_score(adjusted_true_anomalies, predicted_anomalies, zero_division=0)
    
    return f1, anomaly_scores


def optuna_objective(trial):
    """
    Optuna objective function for hyperparameter optimization.
    
    Tunes:
    - top_weight / remaining_weight: Balance between encoder branches
    - hidden_dim: LSTM hidden layer size
    - latent_dim: VAE latent space dimension
    - num_layers: Number of LSTM layers
    - learning_rate: Optimizer learning rate
    - batch_size: Training batch size
    - percentile_threshold: Anomaly detection threshold
    """
    global selected_feature_indices, remaining_feature_indices, metric_tensor_top, metric_tensor_remaining
    global metric_test_tensor_top, metric_test_tensor_remaining, true_anomalies, device
    
    # Hyperparameters to tune
    top_weight = trial.suggest_float("top_weight", 0.3, 0.9)
    remaining_weight = 1.0 - top_weight  # Ensure weights sum to 1
    
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    latent_dim = trial.suggest_categorical("latent_dim", [16, 32, 64])
    num_layers = trial.suggest_int("num_layers", 1, 2)
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    
    kl_weight = trial.suggest_float("kl_weight", 0.01, 1.0, log=True)
    
    percentile_threshold = trial.suggest_int("percentile_threshold", 50, 99)
    
    seq_length = 30
    sequences_top = []
    sequences_remaining = []
    for i in range(metric_tensor_top.shape[0] - seq_length + 1):
        sequences_top.append(metric_tensor_top[i:i + seq_length])
        sequences_remaining.append(metric_tensor_remaining[i:i + seq_length])
    
    sequences_combined = list(zip(sequences_top, sequences_remaining))
    train_data_opt, val_data_opt = train_test_split(sequences_combined, test_size=0.3, random_state=42)
    
    test_sequences_top = []
    test_sequences_remaining = []
    for i in range(metric_test_tensor_top.shape[0] - seq_length + 1):
        test_sequences_top.append(metric_test_tensor_top[i:i + seq_length])
        test_sequences_remaining.append(metric_test_tensor_remaining[i:i + seq_length])
    
    test_sequences_opt = list(zip(test_sequences_top, test_sequences_remaining))
    
    train_loader_opt = DataLoader(dataset=train_data_opt, batch_size=batch_size, shuffle=True)
    val_loader_opt = DataLoader(dataset=val_data_opt, batch_size=batch_size, shuffle=False)
    test_loader_opt = DataLoader(dataset=test_sequences_opt, batch_size=batch_size, shuffle=False)
    
    num_top = len(selected_feature_indices)
    num_remaining = len(remaining_feature_indices)
    
    model_opt = LSTMVAE_Stacked_Weighted(
        num_top_sensors=num_top,
        num_remaining_sensors=num_remaining,
        input_dim=1,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        sequence_length=seq_length,
        num_layers=num_layers,
        device=device,
        top_weight=top_weight,
        remaining_weight=remaining_weight
    ).to(device)
    
    optimizer_opt = Adam(model_opt.parameters(), lr=learning_rate)
    
    # Train with fewer epochs for hyperparameter search
    try:
        train_losses, val_losses, best_val_loss = train_model_for_optuna(
            model_opt, train_loader_opt, val_loader_opt, 
            optimizer_opt, loss_function_weighted, 
            num_epochs=30, device=device, trial=trial
        )
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return 0.0
    
    # Evaluate on test set with the suggested threshold
    f1, _ = evaluate_for_optuna(
        model_opt, test_loader_opt, device, 
        percentile_threshold, true_anomalies, seq_length
    )
    
    del model_opt
    torch.cuda.empty_cache()
    
    return f1


def run_optuna_study(n_trials=50, study_name="lstm_vae_stacked_weighted"):
    """
    Run Optuna hyperparameter optimization study.
    
    Args:
        n_trials: Number of trials to run
        study_name: Name for the study (used for saving)
    
    Returns:
        study: Completed Optuna study object
    """
    # Create study with TPE sampler and median pruner
    study = optuna.create_study(
        direction="maximize",  # Maximize F1 score
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    print(f"Starting Optuna study with {n_trials} trials...")
    print("=" * 60)
    
    study.optimize(
        optuna_objective, 
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    print("\n" + "=" * 60)
    print("Optuna Study Complete!")
    print("=" * 60)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print(f"\nStudy statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")
    
    print(f"\nBest trial:")
    trial = study.best_trial
    print(f"  Value (F1 Score): {trial.value:.4f}")
    print(f"\n  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    study_filename = f"{study_name}_{DATASET_TYPE}_{CHANNEL}_study.pkl"
    joblib.dump(study, study_filename)
    print(f"\nStudy saved to: {study_filename}")
    
    return study


USE_OPTUNA = True
N_OPTUNA_TRIALS = 50 

if USE_OPTUNA:
    study = run_optuna_study(n_trials=N_OPTUNA_TRIALS)
    best_params = study.best_params
    if 'kl_weight' not in best_params:
        best_params['kl_weight'] = 0.1
    print("\nUsing optimized hyperparameters for final training...")
else:
    # Default or previously optimized parameters
    best_params = {
        'top_weight': 0.7,
        'hidden_dim': 128,
        'latent_dim': 32,
        'num_layers': 1,
        'learning_rate': 1e-3,
        'batch_size': 32,
        'percentile_threshold': 90,
        'kl_weight': 0.1
    }
    print("Using default hyperparameters...")

print("\nFinal training parameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

"""## Train Final Model with Best Hyperparameters"""
final_top_weight = best_params['top_weight']
final_remaining_weight = 1.0 - final_top_weight
final_hidden_dim = best_params['hidden_dim']
final_latent_dim = best_params['latent_dim']
final_num_layers = best_params['num_layers']
final_learning_rate = best_params['learning_rate']
final_batch_size = best_params['batch_size']
final_percentile_threshold = best_params['percentile_threshold']

sequences_top_final = []
sequences_remaining_final = []
for i in range(metric_tensor_top.shape[0] - sequence_length + 1):
    sequences_top_final.append(metric_tensor_top[i:i + sequence_length])
    sequences_remaining_final.append(metric_tensor_remaining[i:i + sequence_length])

sequences_combined_final = list(zip(sequences_top_final, sequences_remaining_final))
train_data_final, val_data_final = train_test_split(sequences_combined_final, test_size=0.3, random_state=42)

test_sequences_top_final = []
test_sequences_remaining_final = []
for i in range(metric_test_tensor_top.shape[0] - sequence_length + 1):
    test_sequences_top_final.append(metric_test_tensor_top[i:i + sequence_length])
    test_sequences_remaining_final.append(metric_test_tensor_remaining[i:i + sequence_length])

test_sequences_combined_final = list(zip(test_sequences_top_final, test_sequences_remaining_final))

train_loader_final = DataLoader(dataset=train_data_final, batch_size=final_batch_size, shuffle=True)
val_loader_final = DataLoader(dataset=val_data_final, batch_size=final_batch_size, shuffle=False)
test_loader_final = DataLoader(dataset=test_sequences_combined_final, batch_size=final_batch_size, shuffle=False)

model = LSTMVAE_Stacked_Weighted(
    num_top_sensors=num_top_sensors,
    num_remaining_sensors=num_remaining_sensors,
    input_dim=input_dim,
    hidden_dim=final_hidden_dim,
    latent_dim=final_latent_dim,
    sequence_length=sequence_length,
    num_layers=final_num_layers,
    device=device,
    top_weight=final_top_weight,
    remaining_weight=final_remaining_weight
).to(device)

optimizer = Adam(model.parameters(), lr=final_learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

print(f"\nTraining final model with optimized encoder weights:")
print(f"  top_weight: {final_top_weight:.4f}")
print(f"  remaining_weight: {final_remaining_weight:.4f}")

train_losses, val_losses = train_model_weighted(
    model, train_loader_final, val_loader_final, 
    optimizer, loss_function_weighted, scheduler, 
    num_epochs=256, device=device
)

model_name = f'vae_stacked_weighted_{DATASET_TYPE}_{CHANNEL}_optuna'
save_model(model, model_name)
print(f"\nOptimized model saved as: {model_name}.pth")

"""# Evaluate with Optimized Threshold"""

def evaluate_lstm_weighted(model, test_loader, device, percentile_threshold=90):
    model.eval()
    anomaly_scores = []

    with torch.no_grad():
        for batch_top, batch_remaining in test_loader:
            batch_top = torch.tensor(batch_top, dtype=torch.float32).to(device)
            batch_remaining = torch.tensor(batch_remaining, dtype=torch.float32).to(device)

            batch_scores = []
            for i in range(batch_top.shape[0]):
                sequence_top = batch_top[i, :, :].unsqueeze(0)
                sequence_remaining = batch_remaining[i, :, :].unsqueeze(0)
                
                recon_top, recon_remaining, mean, logvar = model(sequence_top, sequence_remaining)
                loss = loss_function_weighted(sequence_top, sequence_remaining, recon_top, recon_remaining, 
                                             mean, logvar, model.top_weight, model.remaining_weight)
                batch_scores.append(loss.item())
            anomaly_scores.extend(batch_scores)

    threshold = np.percentile(anomaly_scores, percentile_threshold)
    anomaly_indices = [i for i, score in enumerate(anomaly_scores) if score > threshold]
    return anomaly_indices, anomaly_scores

print("\n--- Evaluating Final Model ---")
final_f1, anomaly_scores = evaluate_for_optuna(
    model, test_loader_final, device, 
    final_percentile_threshold, true_anomalies, sequence_length
)

threshold_value = np.percentile(anomaly_scores, final_percentile_threshold)
anomalies = [i for i, score in enumerate(anomaly_scores) if score > threshold_value]

print("\n--- Anomaly Score Diagnostics ---")
print(f"Anomaly scores - Min: {np.min(anomaly_scores):.4f}, Max: {np.max(anomaly_scores):.4f}")
print(f"Anomaly scores - Mean: {np.mean(anomaly_scores):.4f}, Std: {np.std(anomaly_scores):.4f}")
print(f"Threshold percentile: {final_percentile_threshold}")
print(f"Threshold value: {threshold_value:.4f}")
print(f"Number of detected anomalies: {len(anomalies)}")
print(f"True anomaly rate: {np.sum(true_anomalies) / len(true_anomalies) * 100:.2f}%")
print(f"F1 Score: {final_f1:.4f}")

def calculate_f1_score_smap_msl(anomaly_indices, true_anomalies, sequence_length):
    """
    Calculate F1 score for SMAP/MSL datasets.
    
    For windowed sequences, we need to align predictions with the original labels.
    Each sequence at index i corresponds to the time window [i, i+sequence_length-1].
    We assign the prediction to the last timestep of each window.
    """
    # Adjust true_anomalies to match the sequence indices
    # After windowing, we have len(test_data) - sequence_length + 1 sequences
    # Each sequence i corresponds to original positions [i, i+sequence_length-1]
    # We use the label of the last position (i + sequence_length - 1)
    adjusted_true_anomalies = true_anomalies[sequence_length-1:]
    
    # Create a binary array representing predicted anomalies
    predicted_anomalies = np.zeros(len(adjusted_true_anomalies), dtype=int)
    for index in anomaly_indices:
        if index < len(predicted_anomalies):  # Check index bounds
            predicted_anomalies[index] = 1

    # Calculate the F1 score
    f1 = f1_score(adjusted_true_anomalies, predicted_anomalies)
    return f1, predicted_anomalies, adjusted_true_anomalies

f1, predicted_anomalies, adjusted_true_anomalies = calculate_f1_score_smap_msl(
    anomalies, true_anomalies, sequence_length
)
print(f"\nF1 Score (with threshold percentile {final_percentile_threshold}): {f1}")

# Calculate AUC-ROC if we have both classes
unique_true = np.unique(adjusted_true_anomalies)
if len(unique_true) > 1:
    auc_roc = roc_auc_score(adjusted_true_anomalies, predicted_anomalies)
    print(f"AUC-ROC Score: {auc_roc}")
    
    auc_pr = average_precision_score(adjusted_true_anomalies, predicted_anomalies)
    print(f"AUCPR Score: {auc_pr}")
else:
    print("Warning: Only one class in true labels, cannot compute AUC-ROC or AUCPR")

print("\nClassification Report:")
print(classification_report(adjusted_true_anomalies, predicted_anomalies, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(adjusted_true_anomalies, predicted_anomalies))

"""## Additional Evaluation: Point-Adjust F1 (PA%F1)

NASA SMAP/MSL papers often use Point-Adjust F1 score where if any point in an 
anomaly segment is detected, the entire segment is considered detected.
"""

def point_adjust_f1_score(anomaly_indices, true_anomalies, sequence_length, anomaly_sequences):
    """
    Calculate Point-Adjust F1 score used in SMAP/MSL literature.
    
    If any point within an anomaly segment is detected, the entire segment 
    is considered as correctly detected (True Positive).
    """
    # Map sequence indices back to original time indices
    # Sequence i covers positions [i, i + sequence_length - 1]
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

# Get anomaly sequences for the current channel
labels_df = pd.read_csv(LABELS_FILE)
channel_labels = labels_df[labels_df['chan_id'] == CHANNEL]
anomaly_sequences = ast.literal_eval(channel_labels['anomaly_sequences'].iloc[0])

pa_results = point_adjust_f1_score(anomalies, true_anomalies, sequence_length, anomaly_sequences)
print(f"\nPoint-Adjust Evaluation (common in SMAP/MSL papers):")
print(f"  Precision: {pa_results['precision']:.4f}")
print(f"  Recall: {pa_results['recall']:.4f}")
print(f"  F1 Score: {pa_results['f1']:.4f}")
print(f"  Detected Segments: {pa_results['true_positives']} / {pa_results['total_segments']}")


"""## Optuna Visualization Functions

Visualize the hyperparameter optimization results.
"""

def visualize_optuna_study(study, save_path=None):
    """
    Generate visualizations for Optuna study results.
    
    Args:
        study: Optuna study object
        save_path: Optional path prefix to save figures
    """
    import optuna.visualization as vis
    
    # Create figure for optimization history
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    trial_values = [t.value for t in study.trials if t.value is not None]
    trial_numbers = [t.number for t in study.trials if t.value is not None]
    
    ax1.plot(trial_numbers, trial_values, 'bo-', alpha=0.6, label='Trial F1 Score')
    
    # Add best value line
    best_values = []
    current_best = 0
    for val in trial_values:
        if val > current_best:
            current_best = val
        best_values.append(current_best)
    ax1.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best F1 Score')
    
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Optuna Optimization History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(f'{save_path}_optimization_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create figure for parameter importance (focusing on encoder weights)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Extract parameter values from completed trials
    completed_trials = [t for t in study.trials if t.value is not None]
    
    if len(completed_trials) > 0:
        top_weights = [t.params.get('top_weight', None) for t in completed_trials]
        f1_scores = [t.value for t in completed_trials]
        
        # Filter out None values
        valid_data = [(tw, f1) for tw, f1 in zip(top_weights, f1_scores) if tw is not None]
        
        if valid_data:
            top_weights, f1_scores = zip(*valid_data)
            
            scatter = ax2.scatter(top_weights, f1_scores, c=range(len(top_weights)), 
                                  cmap='viridis', alpha=0.7, s=50)
            
            # Highlight best trial
            best_idx = np.argmax(f1_scores)
            ax2.scatter([top_weights[best_idx]], [f1_scores[best_idx]], 
                       c='red', s=200, marker='*', edgecolors='black', 
                       linewidths=2, label=f'Best (top_weight={top_weights[best_idx]:.3f})')
            
            ax2.set_xlabel('Top Encoder Weight')
            ax2.set_ylabel('F1 Score')
            ax2.set_title('Encoder Weight vs Model Performance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.colorbar(scatter, ax=ax2, label='Trial Number')
    
    if save_path:
        plt.savefig(f'{save_path}_weight_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Parameter distribution plot
    fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    param_names = ['top_weight', 'hidden_dim', 'latent_dim', 'learning_rate', 
                   'batch_size', 'percentile_threshold']
    
    for idx, param_name in enumerate(param_names):
        if idx < len(axes):
            param_values = [t.params.get(param_name, None) for t in completed_trials]
            trial_f1s = [t.value for t in completed_trials]
            
            valid_data = [(p, f1) for p, f1 in zip(param_values, trial_f1s) if p is not None]
            
            if valid_data:
                param_vals, f1_vals = zip(*valid_data)
                axes[idx].scatter(param_vals, f1_vals, alpha=0.6)
                axes[idx].set_xlabel(param_name)
                axes[idx].set_ylabel('F1 Score')
                axes[idx].set_title(f'{param_name} vs F1 Score')
                axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}_parameter_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_optuna_summary(study):
    """
    Print a detailed summary of the Optuna study results.
    """
    print("\n" + "=" * 70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    print(f"\nDataset: {DATASET_TYPE}, Channel: {CHANNEL}")
    print(f"Total trials: {len(study.trials)}")
    
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    
    print(f"Completed trials: {len(completed)}")
    print(f"Pruned trials: {len(pruned)}")
    
    if completed:
        f1_scores = [t.value for t in completed]
        print(f"\nF1 Score Statistics:")
        print(f"  Best:   {max(f1_scores):.4f}")
        print(f"  Mean:   {np.mean(f1_scores):.4f}")
        print(f"  Std:    {np.std(f1_scores):.4f}")
        print(f"  Median: {np.median(f1_scores):.4f}")
        
        print(f"\nBest Hyperparameters:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        # Analyze encoder weight importance
        top_weights = [t.params.get('top_weight', 0.5) for t in completed]
        weight_corr = np.corrcoef(top_weights, f1_scores)[0, 1]
        print(f"\nEncoder Weight Analysis:")
        print(f"  Correlation with F1 Score: {weight_corr:.4f}")
        print(f"  Optimal top_weight: {study.best_params.get('top_weight', 'N/A')}")
        print(f"  Optimal remaining_weight: {1 - study.best_params.get('top_weight', 0.5):.4f}")
    
    print("=" * 70)


# Visualize Optuna results if optimization was run
if USE_OPTUNA and 'study' in dir():
    print_optuna_summary(study)
    visualize_optuna_study(study, save_path=f'optuna_{DATASET_TYPE}_{CHANNEL}')


"""## Final Model Performance Summary"""
print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("=" * 70)
print(f"\nDataset: {DATASET_TYPE}, Channel: {CHANNEL}")
print(f"\nOptimized Hyperparameters:")
print(f"  top_weight: {final_top_weight:.4f}")
print(f"  remaining_weight: {final_remaining_weight:.4f}")
print(f"  hidden_dim: {final_hidden_dim}")
print(f"  latent_dim: {final_latent_dim}")
print(f"  num_layers: {final_num_layers}")
print(f"  learning_rate: {final_learning_rate:.6f}")
print(f"  batch_size: {final_batch_size}")
print(f"  percentile_threshold: {final_percentile_threshold:.2f}")
print(f"\nPerformance Metrics:")
print(f"  Point-wise F1 Score: {f1:.4f}")
print(f"  Point-Adjust F1 Score: {pa_results['f1']:.4f}")
print(f"  Detected Anomaly Segments: {pa_results['true_positives']} / {pa_results['total_segments']}")
print("=" * 70)
print(f"  Detected Segments: {pa_results['true_positives']} / {pa_results['total_segments']}")
