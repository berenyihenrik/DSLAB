# -*- coding: utf-8 -*-
"""Training functions for LSTM VAE with stacked weighted encoders."""

import copy
import time
import torch
import torch.nn as nn


def loss_function_weighted(x_top, x_remaining, x_hat_top, x_hat_remaining, mean, log_var, top_weight=0.7, remaining_weight=0.3):
    """
    Compute weighted loss for the stacked VAE.
    
    Args:
        x_top: Original top features
        x_remaining: Original remaining features
        x_hat_top: Reconstructed top features
        x_hat_remaining: Reconstructed remaining features
        mean: Latent mean
        log_var: Latent log variance
        top_weight: Weight for top features reconstruction loss
        remaining_weight: Weight for remaining features reconstruction loss
    
    Returns:
        Total loss (reconstruction + KLD)
    """
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


def train_model_weighted(model, train_loader, val_loader, optimizer, loss_fn, scheduler, num_epochs=10, device='cpu'):
    """
    Train the weighted LSTM VAE model.
    
    Args:
        model: LSTMVAE_Stacked_Weighted model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        loss_fn: Loss function
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Device to train on
    
    Returns:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
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
