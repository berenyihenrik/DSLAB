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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
