# -*- coding: utf-8 -*-
"""Training functions for LSTM VAE with grouped encoders."""

import copy
import time
import torch
import torch.nn as nn


def loss_function_grouped(x_groups, x_recon, mean, log_var, group_weights, group_positions, kl_weight=0.1):
    """
    Compute group-weighted reconstruction loss + KL divergence.
    
    Args:
        x_groups: list of tensors, one per group (batch, seq_len, group_size)
        x_recon: reconstructed output (batch, seq_len, n_total_features)
        mean: latent mean
        log_var: latent log variance
        group_weights: tensor of normalized weights per group
        group_positions: list of lists, positions in x_recon for each group
        kl_weight: beta for KL term
    """
    recon_loss = 0.0
    for i, (x_g, positions) in enumerate(zip(x_groups, group_positions)):
        x_recon_g = x_recon[:, :, positions]
        group_loss = nn.functional.mse_loss(x_recon_g, x_g, reduction='mean')
        recon_loss = recon_loss + group_weights[i] * group_loss

    KLD = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + kl_weight * KLD


def loss_function_grouped_decomposed(x_groups, x_recon, mean, log_var,
                                     group_weights, group_positions,
                                     latent_dim, kl_weight=0.1,
                                     return_timestep_errors=True):
    """
    Decomposed loss returning per-sample, per-group, and per-feature contributions.

    Args:
        x_groups: list of tensors, one per group (batch, seq_len, group_size)
        x_recon: reconstructed output (batch, seq_len, n_total_features)
        mean: latent mean (batch, total_latent_dim)
        log_var: latent log variance (batch, total_latent_dim)
        group_weights: tensor of normalized weights per group
        group_positions: list of lists, positions in x_recon for each group
        latent_dim: latent dimension per encoder group
        kl_weight: beta for KL term
        return_timestep_errors: if True, include (B, T, F) squared-error tensor

    Returns:
        total_loss_per_sample: (batch,)
        components: dict of attribution tensors
    """
    device = x_recon.device
    batch, seq_len, n_total_features = x_recon.shape
    n_groups = len(x_groups)
    total_latent_dim = mean.shape[1]

    # --- Reconstruction (per-feature / per-group) ---
    recon_sqerr_t_f = torch.zeros((batch, seq_len, n_total_features), device=device) if return_timestep_errors else None
    recon_mse_per_feature = torch.zeros((batch, n_total_features), device=device)
    recon_mse_per_group = torch.zeros((batch, n_groups), device=device)

    for gi, (x_g, positions) in enumerate(zip(x_groups, group_positions)):
        x_recon_g = x_recon[:, :, positions]
        sqerr = (x_recon_g - x_g) ** 2

        mse_feat = sqerr.mean(dim=1)  # (batch, group_size)
        recon_mse_per_feature[:, positions] = mse_feat
        recon_mse_per_group[:, gi] = mse_feat.mean(dim=1)

        if return_timestep_errors:
            recon_sqerr_t_f[:, :, positions] = sqerr

    recon_contrib_per_group = recon_mse_per_group * group_weights.unsqueeze(0)

    # Per-feature weights from owning group
    pos_to_group = torch.empty(n_total_features, dtype=torch.long, device=device)
    for gi, positions in enumerate(group_positions):
        for p in positions:
            pos_to_group[p] = gi
    per_feature_weight = group_weights[pos_to_group]
    recon_contrib_per_feature = recon_mse_per_feature * per_feature_weight.unsqueeze(0)

    recon_loss_per_sample = recon_contrib_per_group.sum(dim=1)

    # --- KL divergence (per-group decomposition) ---
    kld_per_dim = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
    kld_total_per_sample = kld_per_dim.mean(dim=1)

    kld_contrib_per_group = torch.zeros((batch, n_groups), device=device)
    for gi in range(n_groups):
        s = gi * latent_dim
        e = (gi + 1) * latent_dim
        kld_contrib_per_group[:, gi] = kld_per_dim[:, s:e].sum(dim=1) / total_latent_dim

    # --- Total ---
    total_loss_per_sample = recon_loss_per_sample + kl_weight * kld_total_per_sample

    components = {
        "recon_mse_per_group": recon_mse_per_group,
        "recon_contrib_per_group": recon_contrib_per_group,
        "recon_mse_per_feature": recon_mse_per_feature,
        "recon_contrib_per_feature": recon_contrib_per_feature,
        "recon_sqerr_t_f": recon_sqerr_t_f,
        "kld_total_per_sample": kld_total_per_sample,
        "kld_contrib_per_group": kld_contrib_per_group,
    }
    return total_loss_per_sample, components


def train_model_grouped(model, train_loader, val_loader, optimizer, loss_fn, scheduler, num_epochs=10,
                        device='cpu', use_amp=True):
    """
    Train the grouped LSTM VAE model.
    
    Args:
        model: LSTMVAE_Grouped model
        train_loader: Training data loader (yields tuples of tensors, one per group)
        val_loader: Validation data loader
        optimizer: Optimizer
        loss_fn: Loss function
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Device to train on
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    torch.cuda.empty_cache()
    device = torch.device(device) if isinstance(device, str) else device
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
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

        for batch in train_loader:
            x_groups = [torch.as_tensor(g, dtype=torch.float32).to(device, non_blocking=True) for g in batch]

            optimizer.zero_grad(set_to_none=True)

            # Time forward pass
            t0 = time.time()

            with torch.cuda.amp.autocast(enabled=use_amp):
                x_recon, mean, logvar = model(x_groups)
                loss = loss_fn(x_groups, x_recon, mean, logvar,
                               model.group_weights, model.group_positions)

            forward_time += time.time() - t0

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
            for batch in val_loader:
                x_groups = [torch.as_tensor(g, dtype=torch.float32).to(device, non_blocking=True) for g in batch]

                with torch.cuda.amp.autocast(enabled=use_amp):
                    x_recon, mean, logvar = model(x_groups)
                    loss = loss_fn(x_groups, x_recon, mean, logvar,
                                   model.group_weights, model.group_positions)
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


def save_model(model, name, input_dim, latent_dim, hidden_dim):
    """
    Save model state to a file.
    
    Args:
        model: Model to save
        name: Name for the saved file (without extension)
        input_dim: Input dimension
        latent_dim: Latent dimension
        hidden_dim: Hidden dimension
    """
    model_state = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'state_dict': model.state_dict()
    }
    torch.save(model_state, name + '.pth')
