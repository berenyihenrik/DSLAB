#!/usr/bin/env python
# coding: utf-8

# # Setup Environment and Read Data

# In[1]:


import torch
import numpy as np
import pandas as pd
import pickle
import copy
from tqdm import trange,tqdm
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix


# ## Setup the dataset

# In[2]:


DRIVE = "drive/MyDrive/Colab Notebooks/ELTE/DSLAB/swat/"
DRIVE = "swat/"
TRAIN_FILE_NAME = "SWaT_Dataset_Normal_v0 1.csv"
TRAIN_DATASET = DRIVE + TRAIN_FILE_NAME
TEST_FILE_NAME = "SWaT_Dataset_Attack_v0 1.csv"
TEST_DATASET = DRIVE + TEST_FILE_NAME

data_train = pd.read_csv(TRAIN_DATASET, header=None)
data_test = pd.read_csv(TEST_DATASET, header=None)


# In[3]:


data_train.replace("Normal", 0, inplace=True)
data_test.replace("Normal", 0, inplace=True)
data_test.replace("Attack", 1, inplace=True)
data_test.replace("A ttack", 1, inplace=True)

train = data_train.iloc[:, 1:].copy()
train = train.iloc[:, :-1].copy()
test = data_test.iloc[:, 1:].copy()
test = test.iloc[:, :-1].copy()


# In[4]:


true_anomalies = data_test[52].to_numpy()
true_anomalies = np.delete(true_anomalies, 0)
true_anomalies = true_anomalies.astype(int)


# In[5]:


true_anomalies


# In[6]:


train


# In[7]:


test


# # Preprocess the Dataset

# In[8]:


# remove headers
if train.iloc[0].dtype == 'object':
    train = train[1:].copy()
    test = test[1:].copy()

# convert all columns to numeric
train = train.apply(pd.to_numeric, errors='coerce')
test = test.apply(pd.to_numeric, errors='coerce')

# interpolate/fill NaNs after conversion if necessary
train.interpolate(inplace=True)
train.bfill(inplace=True)
test.interpolate(inplace=True)
test.bfill(inplace=True)

# calculate variance, crop columns with variance close to zero.
variance_threshold = 1e-6
variances = train.var()
low_variance_cols = variances[variances < variance_threshold].index

print(f"Columns with very low variance: {list(low_variance_cols)}")

# drop low variance columns
train_filtered = train.drop(columns=low_variance_cols)
test_filtered = test.drop(columns=low_variance_cols)

print(f"Original number of features: {train.shape[1]}")
print(f"Number of features after removing low variance columns: {train_filtered.shape[1]}")

# egenerate the tensors and dataloaders with the filtered data
train_tensor = train_filtered.values
test_tensor = test_filtered.values

sequence_length = 30
sequences = []
for i in range(train_tensor.shape[0] - sequence_length + 1):
  sequences.append(train_tensor[i:i + sequence_length])

train_data, val_data = train_test_split(sequences, test_size=0.3, random_state=42, shuffle=False) # 70% train, 30% temp

test_sequences = []
for i in range(test_tensor.shape[0] - sequence_length + 1):
  test_sequences.append(test_tensor[i:i + sequence_length])

batch_size = 32
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_sequences, batch_size=batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[9]:


sequences[0].shape


# # Define the Network

# ## LSTM

# In[10]:


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch, hidden_dim)
        h = h_n[-1]  # take the output of the last layer
        return self.fc_mean(h), self.fc_logvar(h)


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, sequence_length, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.sequence_length = sequence_length
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # Repeat z for each timestep
        hidden = self.latent_to_hidden(z).unsqueeze(1).repeat(1, self.sequence_length, 1)
        out, _ = self.lstm(hidden)
        return self.output_layer(out)


class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, sequence_length, num_layers=1, device='cpu'):
        super(LSTMVAE, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers).to(device)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, sequence_length, num_layers).to(device)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar


# In[11]:


input_dim = train_filtered.shape[1]
hidden_dim = 128
latent_dim = 32
num_layers = 1

model = LSTMVAE(input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                sequence_length=sequence_length,
                num_layers=num_layers,
                device=device).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)


# ## Support functions

# In[12]:


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


# In[ ]:


def save_model(model):
    model_state = {
        'input_dim':input_dim,
        'latent_dim':latent_dim,
        'hidden_dim':hidden_dim,
        'state_dict':model.state_dict()
    }
    torch.save(model_state,'vae_swat.pth')


# # Train

# ## LSTM

# In[14]:


torch.cuda.empty_cache()

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

# SPO optimizer - optuna
# bayesian hyperparameter tuning
# grid search - slow for DL

def train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, num_epochs=10, device='cpu'):
    torch.cuda.empty_cache()
    train_losses = []
    val_losses = []

    early_stop_tolerant_count = 0
    early_stop_tolerant = 10
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for batch in train_loader:
            batch = torch.tensor(batch, dtype=torch.float32).to(device)

            optimizer.zero_grad()

            recon_batch, mean, logvar = model(batch)
            loss = loss_fn(recon_batch, batch, mean, logvar)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = torch.tensor(batch, dtype=torch.float32).to(device)
                recon_batch, mean, logvar = model(batch)
                loss = loss_fn(recon_batch, batch, mean, logvar)
                valid_loss += loss.item()

        valid_loss /= len(val_loader)
        val_losses.append(valid_loss)

        scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_tolerant_count = 0
        else:
            early_stop_tolerant_count += 1

        print(f"Epoch {epoch+1:04d}: train loss {train_loss:.4f}, valid loss {valid_loss:.4f}")

        if early_stop_tolerant_count >= early_stop_tolerant:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model_wts)
    print("Finished Training.")
    return train_losses, val_losses

train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, loss_function, scheduler, num_epochs=100, device=device)


# # Evaluate

# In[15]:


def evaluate_lstm(model, test_loader, device, percentile_threshold=90):
    model.eval()
    anomaly_scores = []

    with torch.no_grad():
        for batch in test_loader:
            batch = torch.tensor(batch, dtype=torch.float32).to(device)

            batch_scores = []
            for i in range(batch.shape[0]): #Iterate through each sequence in the batch
                sequence = batch[i, :, :].unsqueeze(0)  # Select a single sequence
                recon_batch, mean, logvar = model(sequence)
                loss = loss_function(recon_batch, sequence, mean, logvar)
                batch_scores.append(loss.item())
            anomaly_scores.extend(batch_scores)  # Append scores for all sequences in the batch


    # Calculate the threshold based on the specified percentile
    threshold = np.percentile(anomaly_scores, percentile_threshold)

    # Identify anomaly indices
    anomaly_indices = [i for i, score in enumerate(anomaly_scores) if score > threshold]
    return anomaly_indices
anomalies = evaluate_lstm(model, test_loader, device, 90)


# In[16]:


def calculate_f1_score(anomaly_indices, true_anomalies):
    # Create a binary array representing predicted anomalies
    predicted_anomalies = np.zeros_like(true_anomalies)
    for index in anomaly_indices:
        if index < len(predicted_anomalies):  # Check index bounds
          predicted_anomalies[index] = 1

    # Calculate the F1 score
    f1 = f1_score(true_anomalies, predicted_anomalies)
    return f1, predicted_anomalies

# Example usage (assuming 'anomalies' and 'true_anomalies' are defined)
f1, predicted_anomalies = calculate_f1_score(anomalies, true_anomalies)
print(f"F1 Score: {f1}")


# In[17]:


print(classification_report(true_anomalies, predicted_anomalies))


# In[18]:


print(confusion_matrix(true_anomalies, predicted_anomalies))

