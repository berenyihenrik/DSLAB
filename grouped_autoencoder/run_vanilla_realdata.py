# -*- coding: utf-8 -*-
"""Tune and evaluate vanilla single-group LSTM-VAE baselines on real datasets.

This runner is intentionally separate from the grouped-model experiment scripts.
It keeps the current no-leakage train/validation protocols while allowing a
plain single-encoder LSTM-VAE to use vanilla-style training and scoring.

Supported datasets:
    - SMD (`machine-1-1.txt` by default)
    - SWaT (with the same stride-based resampling ratios used in prior tests)

Supported scoring modes:
    - ``ecdf``: two-sided ECDF calibration from validation reconstruction scores
    - ``raw``: raw reconstruction scores with validation-derived threshold
    - ``raw_kl``: reconstruction + KL scores with validation-derived threshold

Examples:
    python run_vanilla_realdata.py --dataset smd --optuna-trials 20
    python run_vanilla_realdata.py --dataset swat --optuna-trials 30
    python run_vanilla_realdata.py --dataset swat --score-mode all --eval-seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import optuna
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from config import (
    DEFAULT_PARAMS_SMD,
    DEFAULT_PARAMS_SWAT,
    DEVICE,
    DATALOADER_WORKERS,
    MACHINE,
    PIN_MEMORY,
    SEQUENCE_LENGTH,
    SMD_DRIVE,
    SWAT_ATTACK_DATASET,
    SWAT_NORMAL_DATASET,
    SWAT_VAL_RATIO,
    USE_AMP,
)
from data_loader import (
    GroupedSequenceDataset,
    detect_binary_features,
    load_smd_data,
    load_swat_data,
    preprocess_data,
    standardize_continuous_features,
)
from feature_selection import split_features_by_groups
from models import LSTMVAE_Grouped


ALL_SCORE_MODES = ("ecdf", "raw", "raw_kl")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "vanilla_results")
DEFAULT_PATIENCE = 10


@dataclass
class DatasetBundle:
    """Container for dataset-specific training and evaluation artifacts.

    Args:
        dataset_name: Short dataset identifier.
        train_dataset: Dataset over normal training windows.
        val_dataset: Dataset over normal validation windows.
        test_dataset: Dataset over test windows.
        true_anomalies: Point-wise anomaly labels for the test split.
        encoder_groups: Single-group feature layout used by the model.
        binary_group_flags: Group loss-type flags passed to the model.
        sequence_length: Sliding window length.
        metadata: Additional dataset-specific details for reporting.
    """

    dataset_name: str
    train_dataset: Any
    val_dataset: Any
    test_dataset: Any
    true_anomalies: np.ndarray
    encoder_groups: list[list[int]]
    binary_group_flags: list[bool]
    sequence_length: int
    metadata: dict[str, Any]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible runs.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _validate_ratio(name: str, ratio: float) -> None:
    """Validate a subsampling ratio.

    Args:
        name: Human-readable parameter name.
        ratio: Ratio value to validate.

    Raises:
        ValueError: If ``ratio`` is outside ``(0, 1]``.
    """
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"{name} must be in (0, 1], got {ratio}")


def _loader_kwargs() -> dict[str, Any]:
    """Build DataLoader kwargs from shared config.

    Returns:
        DataLoader keyword arguments.
    """
    num_workers = min(DATALOADER_WORKERS, max(0, (os.cpu_count() or 2) // 2))
    return {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() and PIN_MEMORY,
        "persistent_workers": num_workers > 0,
    }


def _contiguous_window_split(dataset: Any, val_ratio: float) -> tuple[Subset, Subset]:
    """Split a window dataset into contiguous train and validation subsets.

    Args:
        dataset: Window dataset supporting ``len()`` and integer indexing.
        val_ratio: Fraction of windows assigned to validation.

    Returns:
        Tuple ``(train_subset, val_subset)``.
    """
    n_total = len(dataset)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError("Validation ratio leaves no samples for training")
    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_total))
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _strided_subsample(
    dataset: Any,
    ratio: float,
    tag: str,
    return_indices: bool = False,
) -> Any:
    """Subsample overlap-heavy windows with deterministic uniform stride.

    Args:
        dataset: Dataset to subsample.
        ratio: Fraction of windows to retain.
        tag: Human-readable label for error messages.
        return_indices: Whether to also return selected indices.

    Returns:
        Either ``(subset, used, total)`` or ``(subset, used, total, indices)``.
    """
    _validate_ratio(f"{tag} ratio", ratio)
    n_total = len(dataset)
    if ratio >= 1.0:
        indices = np.arange(n_total, dtype=np.int64)
        if return_indices:
            return dataset, n_total, n_total, indices
        return dataset, n_total, n_total

    n_target = max(1, int(round(n_total * ratio)))
    stride = max(1, n_total // n_target)
    indices = np.arange(0, n_total, stride, dtype=np.int64)[:n_target]
    subset = Subset(dataset, indices.tolist())
    if return_indices:
        return subset, len(indices), n_total, indices
    return subset, len(indices), n_total


def _normalize_binary_with_train_stats(
    train_data: np.ndarray,
    other_arrays: list[np.ndarray],
    binary_indices: set[int],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Normalize two-valued features to ``[0, 1]`` using train-only stats.

    Args:
        train_data: Training array, modified in place.
        other_arrays: Arrays transformed with the same train-only stats.
        binary_indices: Feature indices to normalize.

    Returns:
        Tuple containing the modified training array and list of transformed
        auxiliary arrays.
    """
    for idx in sorted(binary_indices):
        col_min = train_data[:, idx].min()
        col_max = train_data[:, idx].max()
        if col_max > col_min:
            train_data[:, idx] = (train_data[:, idx] - col_min) / (col_max - col_min)
            for arr in other_arrays:
                arr[:, idx] = (arr[:, idx] - col_min) / (col_max - col_min)
    return train_data, other_arrays


def _drop_low_variance_columns(
    train_data: np.ndarray,
    other_arrays: list[np.ndarray],
    variance_threshold: float,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    """Drop near-static columns using train-only variance.

    Args:
        train_data: Training array used to compute variances.
        other_arrays: Arrays to trim using the same retained columns.
        variance_threshold: Minimum allowed variance.

    Returns:
        Tuple ``(trimmed_train, trimmed_others, dropped_indices)``.
    """
    variances = train_data.var(axis=0)
    dropped = np.flatnonzero(variances < variance_threshold).tolist()
    if not dropped:
        return train_data, other_arrays, []

    keep_mask = variances >= variance_threshold
    trimmed_train = train_data[:, keep_mask]
    trimmed_others = [arr[:, keep_mask] for arr in other_arrays]
    return trimmed_train, trimmed_others, dropped


def prepare_smd_bundle(sequence_length: int, machine: str, val_ratio: float) -> DatasetBundle:
    """Prepare the SMD vanilla experiment bundle.

    Args:
        sequence_length: Sliding window length.
        machine: SMD machine filename.
        val_ratio: Contiguous validation ratio over training windows.

    Returns:
        Prepared dataset bundle.
    """
    metric_train, metric_test, true_anomalies = load_smd_data(machine, SMD_DRIVE)
    metric_train = preprocess_data(metric_train.astype(np.float32))
    metric_test = preprocess_data(metric_test.astype(np.float32))

    n_features = metric_train.shape[1]
    encoder_groups = [list(range(n_features))]
    data_groups_train = split_features_by_groups(metric_train, encoder_groups)
    data_groups_test = split_features_by_groups(metric_test, encoder_groups)

    all_train_dataset = GroupedSequenceDataset(data_groups_train, sequence_length)
    train_dataset, val_dataset = _contiguous_window_split(all_train_dataset, val_ratio)
    test_dataset = GroupedSequenceDataset(data_groups_test, sequence_length)

    metadata = {
        "machine": machine,
        "n_features": n_features,
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "test_windows": len(test_dataset),
    }
    return DatasetBundle(
        dataset_name="smd",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        true_anomalies=np.asarray(true_anomalies, dtype=np.int64),
        encoder_groups=encoder_groups,
        binary_group_flags=[False],
        sequence_length=sequence_length,
        metadata=metadata,
    )


def prepare_swat_bundle(
    sequence_length: int,
    drop_low_variance: bool,
    variance_threshold: float,
) -> DatasetBundle:
    """Prepare the SWaT vanilla experiment bundle.

    Args:
        sequence_length: Sliding window length.
        drop_low_variance: Whether to remove near-static features.
        variance_threshold: Variance cutoff used when dropping low-variance features.

    Returns:
        Prepared dataset bundle.
    """
    metric_train, metric_test, true_anomalies = load_swat_data(
        SWAT_NORMAL_DATASET,
        SWAT_ATTACK_DATASET,
    )

    split_idx = int(len(metric_train) * (1.0 - SWAT_VAL_RATIO))
    train_series = metric_train[:split_idx].astype(np.float32, copy=True)
    val_series = metric_train[split_idx:].astype(np.float32, copy=True)
    test_series = metric_test.astype(np.float32, copy=True)

    dropped_low_variance = []
    if drop_low_variance:
        train_series, [val_series, test_series], dropped_low_variance = _drop_low_variance_columns(
            train_series,
            [val_series, test_series],
            variance_threshold,
        )

    n_features = train_series.shape[1]
    binary_feature_indices = detect_binary_features(train_series)
    all_indices = set(range(n_features))
    continuous_indices = sorted(all_indices - binary_feature_indices)

    if binary_feature_indices:
        train_series, [val_series, test_series] = _normalize_binary_with_train_stats(
            train_series,
            [val_series, test_series],
            binary_feature_indices,
        )

    train_series, [val_series, test_series] = standardize_continuous_features(
        train_series,
        [val_series, test_series],
        continuous_indices,
    )

    encoder_groups = [list(range(n_features))]
    data_groups_train = split_features_by_groups(train_series, encoder_groups)
    data_groups_val = split_features_by_groups(val_series, encoder_groups)
    data_groups_test = split_features_by_groups(test_series, encoder_groups)

    train_dataset = GroupedSequenceDataset(data_groups_train, sequence_length)
    val_dataset = GroupedSequenceDataset(data_groups_val, sequence_length)
    test_dataset = GroupedSequenceDataset(data_groups_test, sequence_length)

    metadata = {
        "n_features": n_features,
        "binary_features": len(binary_feature_indices),
        "continuous_features": len(continuous_indices),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "test_windows": len(test_dataset),
        "dropped_low_variance": dropped_low_variance,
    }
    return DatasetBundle(
        dataset_name="swat",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        true_anomalies=np.asarray(true_anomalies, dtype=np.int64),
        encoder_groups=encoder_groups,
        binary_group_flags=[False],
        sequence_length=sequence_length,
        metadata=metadata,
    )


def build_model(bundle: DatasetBundle, params: dict[str, Any], device: torch.device) -> LSTMVAE_Grouped:
    """Construct a single-group vanilla model using the repository implementation.

    Args:
        bundle: Prepared dataset bundle.
        params: Model hyperparameters.
        device: Target device.

    Returns:
        Instantiated model.
    """
    return LSTMVAE_Grouped(
        encoder_groups=bundle.encoder_groups,
        hidden_dim=int(params["hidden_dim"]),
        latent_dim=int(params["latent_dim"]),
        sequence_length=bundle.sequence_length,
        num_layers=int(params["num_layers"]),
        device=device,
        binary_group_flags=bundle.binary_group_flags,
        fusion_type="none",
    ).to(device)


def vanilla_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mean: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float,
) -> torch.Tensor:
    """Compute vanilla LSTM-VAE loss using sum reductions.

    This matches the older standalone LSTM-VAE implementations more closely
    than the grouped-model mean-reduction loss.

    Args:
        x: Input batch of shape ``(batch, seq_len, features)``.
        x_recon: Reconstructed batch.
        mean: Latent means.
        log_var: Latent log-variances.
        kl_weight: Weight applied to the KL term.

    Returns:
        Scalar loss tensor.
    """
    recon_loss = torch.nn.functional.mse_loss(x_recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + kl_weight * kld


def train_vanilla_model(
    model: LSTMVAE_Grouped,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Adam,
    num_epochs: int,
    device: torch.device,
    kl_weight: float,
    use_amp: bool,
    verbose: bool,
    patience: int = DEFAULT_PATIENCE,
) -> tuple[list[float], list[float]]:
    """Train a vanilla single-group LSTM-VAE.

    Args:
        model: Model to train.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        optimizer: Optimizer instance.
        num_epochs: Maximum training epochs.
        device: Target device.
        kl_weight: KL loss weight.
        use_amp: Whether to enable AMP on CUDA.
        verbose: Whether to print per-epoch logs.
        patience: Early-stopping patience.

    Returns:
        Tuple of training and validation loss histories.
    """
    torch.cuda.empty_cache()
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    stagnant_epochs = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = torch.as_tensor(batch[0], dtype=torch.float32).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                x_recon, mean, log_var = model([x])
                loss = vanilla_loss(x, x_recon, mean, log_var, kl_weight)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = torch.as_tensor(batch[0], dtype=torch.float32).to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    x_recon, mean, log_var = model([x])
                    loss = vanilla_loss(x, x_recon, mean, log_var, kl_weight)
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1

        if verbose:
            duration = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1:04d}: train loss {train_loss:.4f}, "
                f"valid loss {val_loss:.4f}, time {duration:.2f}s"
            )

        if stagnant_epochs >= patience:
            if verbose:
                print("Early stopping triggered.")
            break

    model.load_state_dict(best_model_wts)
    if verbose:
        print("Finished Training.")
    return train_losses, val_losses


def compute_vanilla_components(
    model: LSTMVAE_Grouped,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-window reconstruction and KL components.

    Args:
        model: Trained vanilla model.
        loader: Evaluation dataloader.
        device: Target device.

    Returns:
        Tuple ``(reconstruction_scores, kl_scores)``.
    """
    recon_scores = []
    kl_scores = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = torch.as_tensor(batch[0], dtype=torch.float32).to(device, non_blocking=True)
            x_recon, mean, log_var = model([x])
            recon = (x_recon - x).pow(2).sum(dim=(1, 2))
            kl = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=1)
            recon_scores.append(recon.cpu().numpy())
            kl_scores.append(kl.cpu().numpy())

    return np.concatenate(recon_scores), np.concatenate(kl_scores)


def ecdf_scores(reference_recon: np.ndarray, query_recon: np.ndarray) -> np.ndarray:
    """Convert reconstruction losses to two-sided ECDF anomaly scores.

    Args:
        reference_recon: Reference normal reconstruction losses.
        query_recon: Reconstruction losses to score.

    Returns:
        ECDF-based anomaly scores.
    """
    eps = 1e-12
    sorted_ref = np.sort(reference_recon)
    u = np.searchsorted(sorted_ref, query_recon, side="right") / max(1, len(sorted_ref))
    p = 2.0 * np.minimum(u, 1.0 - u)
    return -np.log(p + eps)


def compute_scores_from_components(
    reference_recon: np.ndarray,
    recon_scores: np.ndarray,
    kl_scores: np.ndarray,
    score_mode: str,
    score_kl_weight: float,
) -> np.ndarray:
    """Convert score components into final anomaly scores.

    Args:
        reference_recon: Validation reconstruction scores used by ECDF mode.
        recon_scores: Reconstruction score array.
        kl_scores: KL score array.
        score_mode: One of ``ecdf``, ``raw``, or ``raw_kl``.
        score_kl_weight: Weight applied to KL in ``raw_kl`` mode.

    Returns:
        Final anomaly score array.
    """
    if score_mode == "ecdf":
        return ecdf_scores(reference_recon, recon_scores)
    if score_mode == "raw":
        return recon_scores
    if score_mode == "raw_kl":
        return recon_scores + score_kl_weight * kl_scores
    raise ValueError(f"Unsupported score mode: {score_mode}")


def evaluate_vanilla_model(
    model: LSTMVAE_Grouped,
    val_loader: DataLoader,
    test_loader: DataLoader,
    true_anomalies: np.ndarray,
    sequence_length: int,
    percentile_threshold: float,
    score_mode: str,
    score_kl_weight: float,
    device: torch.device,
    test_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate a trained vanilla model using validation-derived thresholds.

    Args:
        model: Trained vanilla model.
        val_loader: Validation dataloader over normal windows.
        test_loader: Test dataloader.
        true_anomalies: Point-wise test labels.
        sequence_length: Sliding window length.
        percentile_threshold: Validation percentile used as threshold.
        score_mode: Scoring mode.
        score_kl_weight: KL score weight for ``raw_kl`` mode.
        device: Target device.
        test_indices: Optional indices for subsampled test windows.

    Returns:
        Metric dictionary.
    """
    val_recon, val_kl = compute_vanilla_components(model, val_loader, device)
    test_recon, test_kl = compute_vanilla_components(model, test_loader, device)

    val_scores = compute_scores_from_components(
        val_recon,
        val_recon,
        val_kl,
        score_mode,
        score_kl_weight,
    )
    test_scores = compute_scores_from_components(
        val_recon,
        test_recon,
        test_kl,
        score_mode,
        score_kl_weight,
    )

    threshold = float(np.percentile(val_scores, percentile_threshold))
    adjusted_true = np.asarray(true_anomalies[sequence_length - 1 :], dtype=np.int64)
    if test_indices is not None:
        adjusted_true = adjusted_true[np.asarray(test_indices, dtype=np.int64)]

    preds = (test_scores[: len(adjusted_true)] > threshold).astype(int)
    score_slice = np.asarray(test_scores[: len(adjusted_true)], dtype=np.float64)
    f1 = f1_score(adjusted_true, preds, zero_division=0)
    aucpr = (
        average_precision_score(adjusted_true, score_slice)
        if len(np.unique(adjusted_true)) > 1
        else float("nan")
    )

    normal_mask = adjusted_true == 0
    anomaly_mask = adjusted_true == 1
    score_sep = (
        float(score_slice[anomaly_mask].mean() - score_slice[normal_mask].mean())
        if anomaly_mask.any()
        else float("nan")
    )

    return {
        "f1": float(f1),
        "aucpr": float(aucpr),
        "score_sep": score_sep,
        "threshold": threshold,
        "score_mode": score_mode,
        "score_kl_weight": float(score_kl_weight),
        "n_test_windows": int(len(score_slice)),
    }


def choose_score_modes(score_mode_arg: str) -> list[str]:
    """Resolve CLI score-mode selection.

    Args:
        score_mode_arg: CLI value.

    Returns:
        List of score modes to evaluate or tune over.
    """
    if score_mode_arg == "all":
        return list(ALL_SCORE_MODES)
    if score_mode_arg not in ALL_SCORE_MODES:
        raise ValueError(f"score_mode must be one of {ALL_SCORE_MODES} or 'all'")
    return [score_mode_arg]


def dataset_default_epochs(dataset: str) -> int:
    """Return the default epoch count for each dataset.

    Args:
        dataset: Dataset identifier.

    Returns:
        Default epoch count.
    """
    return 256 if dataset == "smd" else 30


def dataset_default_params(dataset: str) -> dict[str, Any]:
    """Return a mutable parameter dictionary for a dataset.

    Args:
        dataset: Dataset identifier.

    Returns:
        Default hyperparameter dictionary.
    """
    if dataset == "smd":
        params = DEFAULT_PARAMS_SMD.copy()
        params.setdefault("score_mode", "raw")
        params.setdefault("score_kl_weight", params.get("kl_weight", 0.1))
        return params
    params = DEFAULT_PARAMS_SWAT.copy()
    params.setdefault("score_mode", "raw")
    params.setdefault("score_kl_weight", params.get("kl_weight", 0.1))
    return params


def suggest_params(trial: optuna.Trial, dataset: str, score_modes: list[str]) -> dict[str, Any]:
    """Sample vanilla hyperparameters for Optuna.

    Args:
        trial: Active Optuna trial.
        dataset: Dataset identifier.
        score_modes: Allowed score modes.

    Returns:
        Sampled parameter dictionary.
    """
    if dataset == "smd":
        hidden_choices = [128, 192, 256, 384, 512]
        latent_choices = [8, 16, 24, 32, 48]
        batch_choices = [256, 512, 896, 1024]
        percentile_low, percentile_high = 85, 95
    else:
        hidden_choices = [128, 192, 256, 320, 384]
        latent_choices = [8, 16, 24, 32, 48]
        batch_choices = [256, 512, 768, 1024]
        percentile_low, percentile_high = 95, 99

    params = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", hidden_choices),
        "latent_dim": trial.suggest_categorical("latent_dim", latent_choices),
        "num_layers": trial.suggest_int("num_layers", 1, 2),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", batch_choices),
        "kl_weight": trial.suggest_float("kl_weight", 1e-4, 1.0, log=True),
        "percentile_threshold": trial.suggest_int("percentile_threshold", percentile_low, percentile_high),
        "score_mode": trial.suggest_categorical("score_mode", score_modes),
    }
    params["score_kl_weight"] = params["kl_weight"]
    return params


def build_dataloaders(
    bundle: DatasetBundle,
    params: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    """Create train/validation/test dataloaders for a run.

    Args:
        bundle: Dataset bundle.
        params: Run hyperparameters.
        args: Parsed CLI arguments.

    Returns:
        Tuple ``(train_loader, val_loader, test_loader, info)``.
    """
    if bundle.dataset_name == "swat":
        train_data, train_used, train_total = _strided_subsample(
            bundle.train_dataset, args.swat_train_ratio, "swat train"
        )
        val_data, val_used, val_total = _strided_subsample(
            bundle.val_dataset, args.swat_val_ratio, "swat val"
        )
        test_data, test_used, test_total, test_indices = _strided_subsample(
            bundle.test_dataset,
            args.swat_test_ratio,
            "swat test",
            return_indices=True,
        )
    else:
        train_data, train_used, train_total = _strided_subsample(
            bundle.train_dataset, args.smd_train_ratio, "smd train"
        )
        val_data, val_used, val_total = _strided_subsample(
            bundle.val_dataset, args.smd_val_window_ratio, "smd val"
        )
        test_data, test_used, test_total, test_indices = _strided_subsample(
            bundle.test_dataset,
            args.smd_test_ratio,
            "smd test",
            return_indices=True,
        )

    lk = _loader_kwargs()
    batch_size = int(params["batch_size"])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **lk)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, **lk)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, **lk)
    info = {
        "train_windows": int(train_used),
        "train_windows_total": int(train_total),
        "val_windows": int(val_used),
        "val_windows_total": int(val_total),
        "test_windows": int(test_used),
        "test_windows_total": int(test_total),
        "test_indices": test_indices,
    }
    return train_loader, val_loader, test_loader, info


def run_single_experiment(
    bundle: DatasetBundle,
    params: dict[str, Any],
    seed: int,
    num_epochs: int,
    device: torch.device,
    verbose: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Train and evaluate one vanilla experiment run.

    Args:
        bundle: Dataset bundle.
        params: Hyperparameters for the run.
        seed: Random seed.
        num_epochs: Training epochs.
        device: Target device.
        verbose: Whether to print training logs.
        args: Parsed CLI arguments.

    Returns:
        Metric dictionary for the run.
    """
    set_seed(seed)
    train_loader, val_loader, test_loader, loader_info = build_dataloaders(bundle, params, args)

    model = build_model(bundle, params, device)
    optimizer = Adam(model.parameters(), lr=float(params["learning_rate"]))

    if verbose:
        print("=" * 72)
        print(
            f"dataset={bundle.dataset_name} seed={seed} score_mode={params['score_mode']} "
            f"hidden={params['hidden_dim']} latent={params['latent_dim']} "
            f"layers={params['num_layers']} batch={params['batch_size']}"
        )
        print(
            f"windows train={loader_info['train_windows']}/{loader_info['train_windows_total']} "
            f"val={loader_info['val_windows']}/{loader_info['val_windows_total']} "
            f"test={loader_info['test_windows']}/{loader_info['test_windows_total']}"
        )

    train_fn = train_vanilla_model
    if verbose:
        train_fn(
            model,
            train_loader,
            val_loader,
            optimizer,
            num_epochs,
            device,
            float(params["kl_weight"]),
            USE_AMP,
            verbose=True,
        )
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            train_fn(
                model,
                train_loader,
                val_loader,
                optimizer,
                num_epochs,
                device,
                float(params["kl_weight"]),
                USE_AMP,
                verbose=False,
            )

    metrics = evaluate_vanilla_model(
        model=model,
        val_loader=val_loader,
        test_loader=test_loader,
        true_anomalies=bundle.true_anomalies,
        sequence_length=bundle.sequence_length,
        percentile_threshold=float(params["percentile_threshold"]),
        score_mode=str(params["score_mode"]),
        score_kl_weight=float(params.get("score_kl_weight", params["kl_weight"])),
        device=device,
        test_indices=loader_info["test_indices"],
    )
    metrics.update(
        {
            "seed": int(seed),
            "hidden_dim": int(params["hidden_dim"]),
            "latent_dim": int(params["latent_dim"]),
            "num_layers": int(params["num_layers"]),
            "learning_rate": float(params["learning_rate"]),
            "batch_size": int(params["batch_size"]),
            "kl_weight": float(params["kl_weight"]),
            "percentile_threshold": float(params["percentile_threshold"]),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            **{k: v for k, v in loader_info.items() if k != "test_indices"},
        }
    )

    del model
    torch.cuda.empty_cache()
    return metrics


def parse_seed_list(raw: str) -> list[int]:
    """Parse a comma-separated seed list.

    Args:
        raw: Comma-separated integer string.

    Returns:
        Parsed seed list.
    """
    seeds = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    return seeds


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-run metrics.

    Args:
        results: List of per-run metric dictionaries.

    Returns:
        Aggregate summary dictionary.
    """
    f1s = np.asarray([r["f1"] for r in results], dtype=np.float64)
    aucprs = np.asarray([r["aucpr"] for r in results], dtype=np.float64)
    seps = np.asarray([r["score_sep"] for r in results], dtype=np.float64)
    thresholds = np.asarray([r["threshold"] for r in results], dtype=np.float64)
    return {
        "f1_mean": float(np.nanmean(f1s)),
        "f1_std": float(np.nanstd(f1s)),
        "aucpr_mean": float(np.nanmean(aucprs)),
        "aucpr_std": float(np.nanstd(aucprs)),
        "score_sep_mean": float(np.nanmean(seps)),
        "score_sep_std": float(np.nanstd(seps)),
        "threshold_mean": float(np.nanmean(thresholds)),
        "threshold_std": float(np.nanstd(thresholds)),
        "n_runs": len(results),
    }


def run_optuna_search(
    bundle: DatasetBundle,
    args: argparse.Namespace,
    device: torch.device,
    num_epochs: int,
    score_modes: list[str],
) -> dict[str, Any]:
    """Run Optuna search for vanilla hyperparameters.

    Args:
        bundle: Dataset bundle.
        args: Parsed CLI arguments.
        device: Target device.
        num_epochs: Number of epochs used during tuning.
        score_modes: Allowed score modes.

    Returns:
        Best parameter dictionary.
    """
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, bundle.dataset_name, score_modes)
        result = run_single_experiment(
            bundle=bundle,
            params=params,
            seed=args.tune_seed,
            num_epochs=num_epochs,
            device=device,
            verbose=args.verbose_training,
            args=args,
        )
        trial.set_user_attr("score_mode", params["score_mode"])
        trial.set_user_attr("aucpr", result["aucpr"])
        trial.set_user_attr("threshold", result["threshold"])
        return result["f1"]

    sampler = optuna.samplers.TPESampler(seed=args.tune_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.optuna_trials)
    best_params = study.best_params.copy()
    best_params["score_kl_weight"] = best_params.get(
        "kl_weight",
        dataset_default_params(bundle.dataset_name)["kl_weight"],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_path = os.path.join(
        args.output_dir,
        f"vanilla_{bundle.dataset_name}_optuna_{timestamp}_best_params.json",
    )
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, sort_keys=True)
    print(f"Saved best params to {best_path}")
    return best_params


def evaluate_param_set(
    bundle: DatasetBundle,
    args: argparse.Namespace,
    params: dict[str, Any],
    seeds: list[int],
    num_epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate one parameter set across multiple seeds.

    Args:
        bundle: Dataset bundle.
        args: Parsed CLI arguments.
        params: Hyperparameters to evaluate.
        seeds: Evaluation seeds.
        num_epochs: Training epochs.
        device: Target device.

    Returns:
        Dictionary containing per-seed and aggregate metrics.
    """
    per_seed = []
    for seed in seeds:
        result = run_single_experiment(
            bundle=bundle,
            params=params,
            seed=seed,
            num_epochs=num_epochs,
            device=device,
            verbose=args.verbose_training,
            args=args,
        )
        per_seed.append(result)
        print(
            f"seed={seed} mode={params['score_mode']} "
            f"F1={result['f1']:.4f} AUCPR={result['aucpr']:.4f} "
            f"Sep={result['score_sep']:.4f} Thr={result['threshold']:.4f}"
        )
    return {
        "params": params,
        "per_seed": per_seed,
        "aggregate": aggregate_results(per_seed),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["smd", "swat"], required=True)
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--tune-epochs", type=int, default=None)
    parser.add_argument("--tune-seed", type=int, default=42)
    parser.add_argument("--eval-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--score-mode", type=str, default="all")
    parser.add_argument("--machine", type=str, default=MACHINE)
    parser.add_argument("--smd-val-ratio", type=float, default=0.30)
    parser.add_argument("--smd-train-ratio", type=float, default=1.0)
    parser.add_argument("--smd-val-window-ratio", type=float, default=1.0)
    parser.add_argument("--smd-test-ratio", type=float, default=1.0)
    parser.add_argument("--swat-train-ratio", type=float, default=0.40)
    parser.add_argument("--swat-val-ratio", type=float, default=0.50)
    parser.add_argument("--swat-test-ratio", type=float, default=0.50)
    parser.add_argument("--drop-low-variance", action="store_true")
    parser.add_argument("--variance-threshold", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verbose-training", action="store_true")
    return parser.parse_args()


def normalize_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply derived defaults and validate CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Normalized argument namespace.
    """
    if args.num_epochs is None:
        args.num_epochs = dataset_default_epochs(args.dataset)
    if args.tune_epochs is None:
        args.tune_epochs = args.num_epochs

    _validate_ratio("SMD validation split", args.smd_val_ratio)
    _validate_ratio("SMD train window ratio", args.smd_train_ratio)
    _validate_ratio("SMD validation window ratio", args.smd_val_window_ratio)
    _validate_ratio("SMD test window ratio", args.smd_test_ratio)
    _validate_ratio("SWaT train ratio", args.swat_train_ratio)
    _validate_ratio("SWaT val ratio", args.swat_val_ratio)
    _validate_ratio("SWaT test ratio", args.swat_test_ratio)

    return args


def report_bundle(bundle: DatasetBundle) -> None:
    """Print dataset preparation details.

    Args:
        bundle: Prepared dataset bundle.
    """
    print(f"Using device: {DEVICE}")
    print(f"Dataset: {bundle.dataset_name}")
    for key, value in bundle.metadata.items():
        print(f"  {key}: {value}")


def write_results_file(
    bundle: DatasetBundle,
    args: argparse.Namespace,
    evaluations: list[dict[str, Any]],
) -> str:
    """Persist experiment outputs to JSON.

    Args:
        bundle: Dataset bundle.
        args: Parsed CLI arguments.
        evaluations: Evaluation payloads.

    Returns:
        Path to the written JSON file.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.output_dir, f"vanilla_{bundle.dataset_name}_{timestamp}_results.json")
    payload = {
        "dataset": bundle.dataset_name,
        "metadata": bundle.metadata,
        "args": vars(args),
        "evaluations": evaluations,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def main() -> None:
    """Run vanilla real-data experiments."""
    args = normalize_cli_args(parse_args())
    device = DEVICE

    if args.dataset == "smd":
        bundle = prepare_smd_bundle(
            sequence_length=SEQUENCE_LENGTH,
            machine=args.machine,
            val_ratio=args.smd_val_ratio,
        )
    else:
        bundle = prepare_swat_bundle(
            sequence_length=SEQUENCE_LENGTH,
            drop_low_variance=args.drop_low_variance,
            variance_threshold=args.variance_threshold,
        )

    report_bundle(bundle)
    score_modes = choose_score_modes(args.score_mode)
    eval_seeds = parse_seed_list(args.eval_seeds)

    base_params = dataset_default_params(bundle.dataset_name)
    if len(score_modes) == 1:
        base_params["score_mode"] = score_modes[0]
    base_params["score_kl_weight"] = base_params.get("kl_weight", 0.1)

    if args.optuna_trials > 0:
        best_params = run_optuna_search(
            bundle=bundle,
            args=args,
            device=device,
            num_epochs=args.tune_epochs,
            score_modes=score_modes,
        )
        param_sets = [best_params]
    elif len(score_modes) == 1:
        param_sets = [base_params]
    else:
        param_sets = []
        for mode in score_modes:
            params = base_params.copy()
            params["score_mode"] = mode
            params["score_kl_weight"] = params.get("kl_weight", 0.1)
            param_sets.append(params)

    evaluations = []
    for params in param_sets:
        print("\n" + "=" * 78)
        print(f"Evaluating score_mode={params['score_mode']}")
        print("=" * 78)
        evaluation = evaluate_param_set(
            bundle=bundle,
            args=args,
            params=params,
            seeds=eval_seeds,
            num_epochs=args.num_epochs,
            device=device,
        )
        evaluations.append(evaluation)
        agg = evaluation["aggregate"]
        print(
            f"Aggregate F1={agg['f1_mean']:.4f} +/- {agg['f1_std']:.4f}  "
            f"AUCPR={agg['aucpr_mean']:.4f} +/- {agg['aucpr_std']:.4f}  "
            f"Sep={agg['score_sep_mean']:.4f} +/- {agg['score_sep_std']:.4f}"
        )

    results_path = write_results_file(bundle, args, evaluations)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
