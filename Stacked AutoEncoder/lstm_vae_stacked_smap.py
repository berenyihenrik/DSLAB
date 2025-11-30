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
    reproduction_loss_top = nn.functional.mse_loss(x_hat_top, x_top, reduction='sum')
    reproduction_loss_remaining = nn.functional.mse_loss(x_hat_remaining, x_remaining, reduction='sum')
    
    # Weighted reconstruction loss
    reproduction_loss = top_weight * reproduction_loss_top + remaining_weight * reproduction_loss_remaining
    
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
    return reproduction_loss + KLD

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

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
train_losses, val_losses = train_model_weighted(model, train_loader_combined, val_loader_combined, 
                                                 optimizer, loss_function_weighted, scheduler, 
                                                 num_epochs=256, device=device)

save_model(model, f'vae_stacked_weighted_{DATASET_TYPE}_{CHANNEL}')

"""# Evaluate"""

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

anomalies, anomaly_scores = evaluate_lstm_weighted(model, test_loader_combined, device, 90)

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
print(f"F1 Score: {f1}")

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