#!/usr/bin/env python
# coding: utf-8

# # Setup Environment and Read Data

# In[ ]:


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


# ## Setup the Dataset

# In[ ]:


DRIVE = "drive/MyDrive/Colab Notebooks/ELTE/DSLAB/ServerMachineDataset/"
MACHINE = "machine-1-1.txt"
TRAIN_DATASET = DRIVE + "train/" + MACHINE
TEST_DATASET = DRIVE + "test/" + MACHINE
TEST_LABEL_DATASET = DRIVE + "test_label/" + MACHINE

metric = pd.read_csv(TRAIN_DATASET, header=None)
metric_test = pd.read_csv(TEST_DATASET, header=None)
true_anomalies = pd.read_csv(TEST_LABEL_DATASET, header=None)[0].to_numpy()


# In[ ]:


metric


# ## Preprocess the Dataset

# ### Non-scaled

# In[ ]:


# create train and test dataloaders
metric.interpolate(inplace=True)
metric.bfill(inplace=True)
metric_tensor = metric.values

metric_test.interpolate(inplace=True)
metric_test.bfill(inplace=True)
metric_test_tensor = metric_test.values

sequence_length = 30
sequences = []
for i in range(metric_tensor.shape[0] - sequence_length + 1):
  sequences.append(metric_tensor[i:i + sequence_length])

train_data, val_data = train_test_split(sequences, test_size=0.3, random_state=42) # 70% train, 30% temp

test_sequences = []
for i in range(metric_test_tensor.shape[0] - sequence_length + 1):
  test_sequences.append(metric_test_tensor[i:i + sequence_length])

batch_size = 32
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_sequences, batch_size=batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


sequences[0].shape


# # Define the Networks

# In[ ]:


input_dim = 38
hidden_dim = 128
latent_dim = 32


# In[ ]:


def save_model(model_name, model, input_dim, latent_dim, hidden_dim):
    model_state = {
        'input_dim':input_dim,
        'latent_dim':latent_dim,
        'hidden_dim':hidden_dim,
        'state_dict':model.state_dict()
    }
    torch.save(model_state, f'drive/MyDrive/Colab Notebooks/ELTE/DSLAB/{model_name}.pth')


# ## AutoEncoder

# In[ ]:


class FeedforwardEncoder(nn.Module):
    def __init__(self, input_dim, sequence_length, hidden_dim, latent_dim):
        super(FeedforwardEncoder, self).__init__()
        self.flatten_dim = input_dim * sequence_length
        self.encoder = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, sequence_length * input_dim)
        z = self.encoder(x)
        return z

class FeedforwardDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, sequence_length):
        super(FeedforwardDecoder, self).__init__()
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * sequence_length)
        )

    def forward(self, z):
        x_recon = self.decoder(z)
        return x_recon.view(z.size(0), self.sequence_length, self.output_dim)

class AE(nn.Module):
    def __init__(self, input_dim, sequence_length, hidden_dim, latent_dim, device='cpu'):
        super(AE, self).__init__()
        self.encoder = FeedforwardEncoder(input_dim, sequence_length, hidden_dim, latent_dim).to(device)
        self.decoder = FeedforwardDecoder(latent_dim, hidden_dim, input_dim, sequence_length).to(device)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


# In[ ]:


def loss_function_ae(x, x_hat):
    return nn.functional.mse_loss(x_hat, x, reduction='sum')


# In[ ]:


model_ae = AE(input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                sequence_length=sequence_length,
                device=device).to(device)
optimizer_ae = Adam(model_ae.parameters(), lr=1e-3)
scheduler_ae = ReduceLROnPlateau(optimizer_ae, 'min', patience=5, factor=0.1, verbose=True)


# ## LSTM AutoEncoder

# In[ ]:


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]  # Take last layer's hidden state
        z = self.fc(h)
        return z

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, sequence_length, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.sequence_length = sequence_length
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        hidden = self.latent_to_hidden(z).unsqueeze(1).repeat(1, self.sequence_length, 1)
        out, _ = self.lstm(hidden)
        return self.output_layer(out)


class LSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, sequence_length, num_layers=1, device='cpu'):
        super(LSTMAE, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers).to(device)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, sequence_length, num_layers).to(device)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


# In[ ]:


def loss_function_lstm_ae(x, x_hat):
    return nn.functional.mse_loss(x_hat, x, reduction='sum')


# In[ ]:


model_lstm_ae = LSTMAE(input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                sequence_length=sequence_length,
                num_layers=1,
                device=device).to(device)
optimizer_lstm_ae = Adam(model_lstm_ae.parameters(), lr=1e-3)
scheduler_lstm_ae = ReduceLROnPlateau(optimizer_lstm_ae, 'min', patience=5, factor=0.1, verbose=True)


# ## LSTM Variational AutoEncoder

# In[ ]:


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


# In[ ]:


def loss_function_lstm_vae(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


# In[ ]:


model_lstm_vae = LSTMVAE(input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                sequence_length=sequence_length,
                num_layers=1,
                device=device).to(device)
optimizer_lstm_vae = Adam(model_lstm_vae.parameters(), lr=1e-3)
scheduler_lstm_vae = ReduceLROnPlateau(optimizer_lstm_vae, 'min', patience=5, factor=0.1, verbose=True)


# # Train

# In[ ]:


def train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, variational=False, num_epochs=10, device='cpu'):
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

            if variational:
                recon_batch, mean, logvar = model(batch)
                loss = loss_fn(recon_batch, batch, mean, logvar)
            else:
                recon_batch = model(batch)
                loss = loss_fn(batch, recon_batch)

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
                if variational:
                    recon_batch, mean, logvar = model(batch)
                    loss = loss_fn(recon_batch, batch, mean, logvar)
                else:
                    recon_batch = model(batch)
                    loss = loss_fn(batch, recon_batch)
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


# ## AutoEncoder

# In[ ]:


train_losses_ae, val_losses_ae = train_model(model_ae, train_loader, val_loader, optimizer_ae, loss_function_ae, scheduler_ae, False, num_epochs=100, device=device)


# ## LSTM AutoEncoder

# In[ ]:


train_losses_lstm_ae, val_losses_lstm_ae = train_model(model_lstm_ae, train_loader, val_loader, optimizer_lstm_ae, loss_function_lstm_ae, scheduler_lstm_ae, False, num_epochs=100, device=device)


# ## LSTM Variational AutoEncoder

# In[ ]:


train_losses_lstm_vae, val_losses_lstm_vae = train_model(model_lstm_vae, train_loader, val_loader, optimizer_lstm_vae, loss_function_lstm_vae, scheduler_lstm_vae, True, num_epochs=100, device=device)


# # Evaluation

# In[ ]:


def evaluate_model(model, variational, test_loader, device, loss_fn, percentile_threshold=90):
    model.eval()
    anomaly_scores = []

    with torch.no_grad():
        for batch in test_loader:
            batch = torch.tensor(batch, dtype=torch.float32).to(device)

            batch_scores = []
            for i in range(batch.shape[0]):  # Iterate through each sequence in the batch
                sequence = batch[i].unsqueeze(0)  # Shape: (1, seq_len, features)
                if variational:
                    recon_sequence, mean, logvar = model(sequence)
                    loss = loss_fn(recon_sequence, sequence, mean, logvar)
                else:
                    recon_sequence = model(sequence)
                    loss = loss_fn(sequence, recon_sequence)
                batch_scores.append(loss.item())

            anomaly_scores.extend(batch_scores)

    # Calculate threshold and identify anomalies
    threshold = np.percentile(anomaly_scores, percentile_threshold)
    anomaly_indices = [i for i, score in enumerate(anomaly_scores) if score > threshold]
    return anomaly_indices


# In[ ]:


def calculate_f1_score(anomaly_indices, true_anomalies):
    # Create a binary array representing predicted anomalies
    predicted_anomalies = np.zeros_like(true_anomalies)
    for index in anomaly_indices:
        if index < len(predicted_anomalies):  # Check index bounds
          predicted_anomalies[index] = 1

    # Calculate the F1 score
    f1 = f1_score(true_anomalies, predicted_anomalies)
    return f1, predicted_anomalies


# ## AutoEncoder

# In[ ]:


anomalies_ae = evaluate_model(model_ae, False, test_loader, device, loss_function_ae, 90)


# In[ ]:


f1_ae, predicted_anomalies_ae = calculate_f1_score(anomalies_ae, true_anomalies)
print(f"F1 Score: {f1_ae}")


# In[ ]:


print(classification_report(true_anomalies, predicted_anomalies_ae))


# In[ ]:


print(confusion_matrix(true_anomalies, predicted_anomalies_ae))


# ## LSTM AutoEncoder

# In[ ]:


anomalies_lstm_ae = evaluate_model(model_lstm_ae, False, test_loader, device, loss_function_lstm_ae, 90)


# In[ ]:


f1_lstm_ae, predicted_anomalies_lstm_ae = calculate_f1_score(anomalies_lstm_ae, true_anomalies)
print(f"F1 Score: {f1_lstm_ae}")


# In[ ]:


print(classification_report(true_anomalies, predicted_anomalies_lstm_ae))


# In[ ]:


print(confusion_matrix(true_anomalies, predicted_anomalies_lstm_ae))


# ## LSTM Variational AutoEncoder

# In[ ]:


anomalies_lstm_vae = evaluate_model(model_lstm_vae, True, test_loader, device, loss_function_lstm_vae, 90)


# In[ ]:


f1_lstm_vae, predicted_anomalies_lstm_vae = calculate_f1_score(anomalies_lstm_vae, true_anomalies)
print(f"F1 Score: {f1_lstm_vae}")


# In[ ]:


print(classification_report(true_anomalies, predicted_anomalies_lstm_vae))


# In[ ]:


print(confusion_matrix(true_anomalies, predicted_anomalies_lstm_vae))

