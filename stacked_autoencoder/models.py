# -*- coding: utf-8 -*-
"""Model architectures for LSTM VAE with grouped encoders."""

import torch
import torch.nn as nn


class ResidualMLPFusion(nn.Module):
    """Small residual MLP to fuse concatenated group representations."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 2 * dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(2 * dim, dim)

    def forward(self, x):
        residual = x
        out = self.norm(x)
        out = self.fc2(self.drop(self.act(self.fc1(out))))
        return residual + out


class GroupSelfAttentionFusion(nn.Module):
    """Self-attention fusion over per-group latent tokens."""

    def __init__(self, n_groups, latent_dim, dropout=0.1, num_heads=1):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")

        self.n_groups = n_groups
        self.latent_dim = latent_dim
        self.group_embed = nn.Parameter(torch.randn(1, n_groups, latent_dim) * 0.02)

        self.norm1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * latent_dim, latent_dim),
        )
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        tokens = x.reshape(batch_size, self.n_groups, self.latent_dim)
        tokens = tokens + self.group_embed

        attn_in = self.norm1(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + self.attn_drop(attn_out)

        ffn_in = self.norm2(tokens)
        ffn_out = self.ffn(ffn_in)
        tokens = tokens + self.ffn_drop(ffn_out)

        return tokens.reshape(batch_size, self.n_groups * self.latent_dim)


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
                 num_layers=1, device='cpu', group_weights=None, binary_group_flags=None,
                 fusion_type="none", fusion_dropout=0.1):
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
            binary_group_flags: optional list[bool] — one per group, True if all
                               features in that group are binary. Used to select
                               BCE loss instead of MSE for those groups.
            fusion_type: fusion strategy for concatenated latents. One of
                         "none", "mlp", "attn_mean", "attn_both".
            fusion_dropout: dropout rate for fusion MLP
        """
        super(LSTMVAE_Grouped, self).__init__()
        self.encoder_groups = encoder_groups
        self.sequence_length = sequence_length
        self.device = device
        self.n_total_features = sum(len(g) for g in encoder_groups)
        self.binary_group_flags = binary_group_flags
        valid_fusion_types = {"none", "mlp", "attn_mean", "attn_both"}
        if fusion_type not in valid_fusion_types:
            raise ValueError(
                f"fusion_type must be one of {sorted(valid_fusion_types)}, got {fusion_type!r}"
            )
        self.fusion_type = fusion_type
        n_groups = len(encoder_groups)

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
        index_to_pos = {idx: pos for pos, idx in enumerate(all_indices)}
        self.group_positions = [
            [index_to_pos[idx] for idx in group]
            for group in encoder_groups
        ]

        decoder_input_dim = n_groups * latent_dim
        self.decoder = SharedDecoder(
            decoder_input_dim, hidden_dim, self.n_total_features, sequence_length, num_layers
        )

        if self.fusion_type == "mlp":
            self.mean_fuser = ResidualMLPFusion(decoder_input_dim, dropout=fusion_dropout)
            self.logvar_fuser = ResidualMLPFusion(decoder_input_dim, dropout=fusion_dropout)
        elif self.fusion_type == "attn_mean":
            self.mean_fuser = GroupSelfAttentionFusion(
                n_groups=n_groups,
                latent_dim=latent_dim,
                dropout=fusion_dropout,
            )
        elif self.fusion_type == "attn_both":
            self.mean_fuser = GroupSelfAttentionFusion(
                n_groups=n_groups,
                latent_dim=latent_dim,
                dropout=fusion_dropout,
            )
            self.logvar_fuser = GroupSelfAttentionFusion(
                n_groups=n_groups,
                latent_dim=latent_dim,
                dropout=fusion_dropout,
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

        if self.fusion_type == "mlp":
            mean = self.mean_fuser(mean)
            logvar = self.logvar_fuser(logvar)
        elif self.fusion_type == "attn_mean":
            mean = self.mean_fuser(mean)
        elif self.fusion_type == "attn_both":
            mean = self.mean_fuser(mean)
            logvar = self.logvar_fuser(logvar)

        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)

        return x_recon, mean, logvar
