# -*- coding: utf-8 -*-
"""Model architectures for LSTM VAE with grouped encoders."""

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
    def __init__(self, input_features_dim, hidden_dim, output_features_dim, sequence_length, num_layers=1):
        super(SharedDecoder, self).__init__()
        self.sequence_length = sequence_length
        self.latent_to_hidden = nn.Linear(input_features_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_features_dim)

    def forward(self, z):
        hidden = self.latent_to_hidden(z).unsqueeze(1).repeat(1, self.sequence_length, 1)
        out, _ = self.lstm(hidden)
        return self.output_layer(out)


class LSTMVAE_Grouped(nn.Module):
    def __init__(self, encoder_groups, hidden_dim, latent_dim, sequence_length,
                 num_layers=1, device='cpu', group_weights=None):
        """
        Args:
            encoder_groups: list[list[int]] — feature indices per encoder group
            hidden_dim: LSTM hidden dimension
            latent_dim: latent dimension per encoder
            sequence_length: sequence window length
            num_layers: LSTM layers
            device: torch device
            group_weights: optional list of floats (one per group) for loss weighting.
                          If None, all groups weighted equally.
        """
        super(LSTMVAE_Grouped, self).__init__()
        self.encoder_groups = encoder_groups
        self.sequence_length = sequence_length
        self.device = device
        self.n_total_features = sum(len(g) for g in encoder_groups)
        self.n_groups = len(encoder_groups)
        self.latent_dim = latent_dim
        self.total_latent_dim = self.n_groups * latent_dim
        n_groups = self.n_groups

        self.encoders = nn.ModuleList([
            LSTMEncoder(len(group), hidden_dim, latent_dim, num_layers)
            for group in encoder_groups
        ])

        if group_weights is None:
            weights = torch.ones(n_groups) / n_groups
        else:
            weights = torch.tensor(group_weights, dtype=torch.float32)
            weights = weights / weights.sum()
        self.register_buffer('group_weights', weights)

        # Build group_positions: map each group's features to positions in sorted output
        all_indices = sorted([idx for group in encoder_groups for idx in group])
        self.feature_order = all_indices
        index_to_pos = {idx: pos for pos, idx in enumerate(all_indices)}
        self.group_positions = [
            [index_to_pos[idx] for idx in group]
            for group in encoder_groups
        ]

        # Map each decoder output position to its owning group index
        pos_to_group = [0] * len(all_indices)
        for gi, positions in enumerate(self.group_positions):
            for p in positions:
                pos_to_group[p] = gi
        self.register_buffer('pos_to_group', torch.tensor(pos_to_group, dtype=torch.long))

        decoder_input_dim = n_groups * latent_dim
        self.decoder = SharedDecoder(
            decoder_input_dim, hidden_dim, self.n_total_features, sequence_length, num_layers
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x_groups):
        """
        Args:
            x_groups: list of tensors, one per group.
                      Each tensor shape: (batch, seq_len, group_size)
        Returns:
            x_recon: (batch, seq_len, n_total_features) in sorted feature order
            mean: (batch, n_groups * latent_dim)
            logvar: (batch, n_groups * latent_dim)
        """
        means = []
        logvars = []
        for encoder, x_g in zip(self.encoders, x_groups):
            mean_i, logvar_i = encoder(x_g)
            means.append(mean_i)
            logvars.append(logvar_i)

        mean = torch.cat(means, dim=1)
        logvar = torch.cat(logvars, dim=1)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)

        return x_recon, mean, logvar
