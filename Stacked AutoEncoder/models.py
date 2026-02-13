# -*- coding: utf-8 -*-
"""Model architectures for LSTM VAE with stacked weighted encoders."""

import torch
import torch.nn as nn


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
