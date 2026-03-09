# -*- coding: utf-8 -*-
"""Synthetic grouped time-series generator with spatial/temporal anomalies."""

from dataclasses import dataclass
import numpy as np


ANOMALY_TYPES = [
    "spatial_regime_desync_g1",
    "spatial_routing_swap_g3",
    "spatial_group_transplant_g1",
    "spatial_sign_flip_g1",
    "temporal_lag_shift_g1",
    "temporal_phase_jump_g1",
    "temporal_time_warp_g1",
    "temporal_event_order_shift_g2",
    "control_spike",
    "control_dropout",
]


@dataclass
class SyntheticConfig:
    train_T: int = 12000
    test_T: int = 6000
    seq_len: int = 30
    noise_std: float = 0.05
    tau1: int = 3
    tau2: int = 2
    regime_dwell: tuple[int, int] = (40, 120)
    event_rate: float = 0.02
    anomaly_len: tuple[int, int] = (45, 90)


def _build_regime(T, dwell, rng):
    lo, hi = dwell
    out = np.zeros(T, dtype=np.int64)
    t = 0
    state = int(rng.integers(0, 2))
    while t < T:
        d = int(rng.integers(lo, hi + 1))
        out[t:min(T, t + d)] = state
        state = 1 - state
        t += d
    return out


def generate_hidden_processes(T, cfg: SyntheticConfig, rng):
    """Generate hidden regime, phase and sparse pulse processes."""
    regime = _build_regime(T, cfg.regime_dwell, rng)

    # Regime-dependent oscillator frequency with small stochastic drift.
    omega = np.where(regime == 0, 0.16, 0.10) + rng.normal(0.0, 0.01, size=T)
    theta = np.cumsum(omega)

    pulse = (rng.random(T) < cfg.event_rate).astype(np.float32)
    return {
        "regime": regime,
        "theta": theta,
        "pulse": pulse,
    }


def _lagged_copy(arr, lag):
    out = np.empty_like(arr)
    for t in range(arr.shape[0]):
        src = max(0, t - int(lag[t]))
        out[t] = arr[src]
    return out


def _time_warp_segment(segment, factor):
    n, d = segment.shape
    src_t = np.arange(n, dtype=np.float32)
    warped_t = np.clip(np.arange(n, dtype=np.float32) * factor, 0, n - 1)
    out = np.zeros_like(segment)
    for j in range(d):
        out[:, j] = np.interp(src_t, warped_t, segment[:, j])
    return out


def _cosine_ramp_weights(length, ramp):
    ramp = max(1, min(ramp, (length - 1) // 2))
    w = np.ones(length, dtype=np.float32)
    edge = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, ramp)))
    w[:ramp] = edge
    w[-ramp:] = edge[::-1]
    return w


def _build_g0(theta, noise_std, rng):
    g0 = np.stack([
        np.sin(theta),
        np.cos(theta),
        0.7 * np.sin(theta) + 0.3 * np.sin(2.0 * theta + 0.4),
    ], axis=1)
    return g0 + rng.normal(0.0, noise_std, size=g0.shape)


def _build_g1(g0, theta, regime, cfg, noise_std, rng, tau1=None, tau2=None, phase_offset=0.0, sign_flip=False):
    tau1 = cfg.tau1 if tau1 is None else tau1
    tau2 = cfg.tau2 if tau2 is None else tau2
    lag = np.where(regime == 0, tau1, tau2)
    g0_lag = _lagged_copy(g0, lag)
    theta_lag = _lagged_copy((theta + phase_offset)[:, None], lag)[:, 0]

    gate = np.where(regime == 0, 0.85, 1.15).astype(np.float32)
    g1 = np.stack([
        gate * (0.85 * g0_lag[:, 0] + 0.15 * g0_lag[:, 2]),
        gate * (-0.4 * g0_lag[:, 0] + 0.8 * g0_lag[:, 1]),
        gate * (0.65 * g0_lag[:, 2] + 0.25 * np.sin(theta_lag)),
    ], axis=1)

    if sign_flip:
        g1 = -g1

    return g1 + rng.normal(0.0, noise_std, size=g1.shape)


def _build_g2(regime, pulse, noise_std, rng):
    kernel = np.array([0.55, 0.3, 0.15], dtype=np.float32)
    pulse_sm = np.convolve(pulse, kernel, mode="same")
    g2 = np.stack([
        regime.astype(np.float32),
        pulse,
        0.6 * regime.astype(np.float32) + 0.8 * pulse_sm,
    ], axis=1)
    return g2 + rng.normal(0.0, noise_std * 0.5, size=g2.shape)


def _build_g3(g0, g1, regime, noise_std, rng, routing_swap=False):
    route_a = np.stack([
        g0[:, 0] * g1[:, 0],
        g0[:, 1] * g1[:, 2],
        g0[:, 2] * g1[:, 1],
    ], axis=1)
    route_b = np.stack([
        g0[:, 0] * g1[:, 1] - 0.5 * g0[:, 2] * g1[:, 2],
        np.tanh(g0[:, 1] + g1[:, 0]),
        0.5 * g0[:, 2] ** 2 - 0.3 * g1[:, 2] ** 2,
    ], axis=1)

    mask = regime[:, None] == (0 if routing_swap else 1)
    g3 = np.where(mask, route_a, route_b)
    return g3 + rng.normal(0.0, noise_std, size=g3.shape)


def _compose_series(hidden, cfg, rng, routing_swap=False, tau1_override=None, phase_offset=0.0, sign_flip=False):
    regime = hidden["regime"]
    theta = hidden["theta"]
    pulse = hidden["pulse"]

    g0 = _build_g0(theta, cfg.noise_std, rng)
    g1 = _build_g1(
        g0, theta, regime, cfg, cfg.noise_std, rng,
        tau1=tau1_override if tau1_override is not None else cfg.tau1,
        tau2=cfg.tau2,
        phase_offset=phase_offset,
        sign_flip=sign_flip,
    )
    g2 = _build_g2(regime, pulse, cfg.noise_std, rng)
    g3 = _build_g3(g0, g1, regime, cfg.noise_std, rng, routing_swap=routing_swap)
    return np.concatenate([g0, g1, g2, g3], axis=1).astype(np.float32)


def generate_normal_series(hidden, cfg: SyntheticConfig, rng):
    """Create one normal multivariate series with fixed 4 groups (12 features)."""
    return _compose_series(hidden, cfg, rng)


def inject_anomaly_segment(series, hidden, anomaly_type, start, length, cfg: SyntheticConfig, rng):
    """Inject one anomaly segment with cosine-ramp boundaries."""
    end = start + length
    segment = series[start:end].copy()
    anom = segment.copy()

    local_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))

    if anomaly_type == "spatial_regime_desync_g1":
        hidden_alt = {k: v.copy() for k, v in hidden.items()}
        hidden_alt["regime"][start:end] = 1 - hidden_alt["regime"][start:end]
        alt = _compose_series(hidden_alt, cfg, local_rng)
        anom[:, 3:6] = alt[start:end, 3:6]
    elif anomaly_type == "spatial_routing_swap_g3":
        alt = _compose_series(hidden, cfg, local_rng, routing_swap=True)
        anom[:, 9:12] = alt[start:end, 9:12]
    elif anomaly_type == "spatial_group_transplant_g1":
        max_start = series.shape[0] - length
        src = int(local_rng.integers(0, max_start + 1))
        if abs(src - start) < length:
            src = (src + 4 * length) % max_start
        anom[:, 3:6] = series[src:src + length, 3:6]
    elif anomaly_type == "spatial_sign_flip_g1":
        anom[:, 3:6] = -segment[:, 3:6]
    elif anomaly_type == "temporal_lag_shift_g1":
        alt = _compose_series(hidden, cfg, local_rng, tau1_override=7)
        anom[:, 3:6] = alt[start:end, 3:6]
    elif anomaly_type == "temporal_phase_jump_g1":
        alt = _compose_series(hidden, cfg, local_rng, phase_offset=np.pi / 3.0)
        anom[:, 3:6] = alt[start:end, 3:6]
    elif anomaly_type == "temporal_time_warp_g1":
        factor = float(local_rng.choice([0.7, 1.3]))
        anom[:, 3:6] = _time_warp_segment(segment[:, 3:6], factor)
    elif anomaly_type == "temporal_event_order_shift_g2":
        shift = int(local_rng.choice([-10, -8, -6, 6, 8, 10]))
        hidden_alt = {k: v.copy() for k, v in hidden.items()}
        pulse = hidden_alt["pulse"].copy()
        pulse[start:end] = np.roll(pulse[start:end], shift)
        hidden_alt["pulse"] = pulse
        alt = _compose_series(hidden_alt, cfg, local_rng)
        anom[:, 6:9] = alt[start:end, 6:9]
    elif anomaly_type == "control_spike":
        center = length / 2.0
        spread = max(3.0, length / 6.0)
        t = np.arange(length)
        envelope = np.exp(-((t - center) ** 2) / (2.0 * spread ** 2))[:, None]
        spike = local_rng.normal(0.0, 4.0 * cfg.noise_std, size=anom.shape)
        anom = anom + spike * envelope
    elif anomaly_type == "control_dropout":
        if start > 0:
            hold = np.repeat(series[start - 1:start], length, axis=0)
        else:
            hold = np.zeros_like(anom)
        anom = hold
    else:
        raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    ramp = int(local_rng.integers(3, 6))
    weights = _cosine_ramp_weights(length, ramp)[:, None]
    series[start:end] = segment * (1.0 - weights) + anom * weights
    return ramp


def build_test_with_anomalies(test_normal, hidden, cfg: SyntheticConfig, rng, n_segments=10):
    """Create anomalous test series and labels with non-overlapping segments."""
    T = test_normal.shape[0]
    out = test_normal.copy()
    labels = np.zeros(T, dtype=np.int64)
    type_labels = {k: np.zeros(T, dtype=np.int64) for k in ANOMALY_TYPES}

    anomaly_types = ANOMALY_TYPES.copy()
    if n_segments <= len(anomaly_types):
        local_types = anomaly_types[:n_segments]
    else:
        extra = [anomaly_types[i % len(anomaly_types)] for i in range(n_segments - len(anomaly_types))]
        local_types = anomaly_types + extra
    rng.shuffle(local_types)

    occupied = np.zeros(T, dtype=bool)
    segments = []

    for anomaly_type in local_types:
        length = int(rng.integers(cfg.anomaly_len[0], cfg.anomaly_len[1] + 1))
        ok = False
        start = 0
        for _ in range(5000):
            start = int(rng.integers(0, T - length + 1))
            lo = max(0, start - 6)
            hi = min(T, start + length + 6)
            if not occupied[lo:hi].any():
                occupied[start:start + length] = True
                ok = True
                break
        if not ok:
            raise RuntimeError("Could not place non-overlapping anomaly segments.")

        ramp = inject_anomaly_segment(out, hidden, anomaly_type, start, length, cfg, rng)
        end = start + length
        labels[start:end] = 1
        type_labels[anomaly_type][start:end] = 1
        segments.append({
            "type": anomaly_type,
            "start": start,
            "end": end,
            "length": length,
            "ramp": ramp,
        })

    segments.sort(key=lambda s: s["start"])
    return out.astype(np.float32), labels, type_labels, segments


def standardize_train_test(train_data, test_data):
    """Z-score normalize test data using train statistics."""
    mean = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    train = (train_data - mean) / std
    test = (test_data - mean) / std
    return train.astype(np.float32), test.astype(np.float32)
