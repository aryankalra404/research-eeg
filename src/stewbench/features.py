"""Hand-crafted EEG features for the classical baselines.

All features are computed per window from band-passed (not normalized)
signals, so they depend on nothing outside the window. Scaling statistics
are fitted later inside each fold's pipeline on training subjects only.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from . import constants as C


def band_powers(X: np.ndarray, sfreq: int = C.SFREQ) -> tuple[np.ndarray, list[str]]:
    """Welch PSD band power per channel/band -> (N, C, B) absolute power."""
    freqs, psd = welch(X, fs=sfreq, nperseg=min(256, X.shape[-1]), axis=-1)
    df = freqs[1] - freqs[0]
    powers = np.stack([psd[..., (freqs >= lo) & (freqs < hi)].sum(-1) * df
                       for lo, hi in C.FREQ_BANDS.values()], axis=-1)
    return powers, list(C.FREQ_BANDS)


def hjorth(X: np.ndarray) -> np.ndarray:
    """Hjorth mobility and complexity per channel -> (N, C, 2)."""
    d1 = np.diff(X, axis=-1)
    d2 = np.diff(d1, axis=-1)
    var0, var1, var2 = X.var(-1), d1.var(-1), d2.var(-1)
    mobility = np.sqrt(var1 / np.maximum(var0, 1e-12))
    complexity = np.sqrt(var2 / np.maximum(var1, 1e-12)) / np.maximum(mobility, 1e-12)
    return np.stack([mobility, complexity], axis=-1)


def spectral_features(X: np.ndarray, sfreq: int = C.SFREQ) -> tuple[np.ndarray, list[str]]:
    """Log absolute band power, relative band power, workload ratios
    (theta/alpha, engagement index beta/(alpha+theta); Pope et al., 1995),
    hemispheric alpha asymmetry, and Hjorth parameters."""
    powers, bands = band_powers(X, sfreq)
    total = powers.sum(-1, keepdims=True)
    log_abs = np.log(powers + 1e-12)
    relative = powers / np.maximum(total, 1e-12)
    b = {name: powers[..., i] for i, name in enumerate(bands)}
    theta_alpha = np.log((b["theta"] + 1e-12) / (b["alpha"] + 1e-12))
    engagement = np.log((b["beta"] + 1e-12) / (b["alpha"] + b["theta"] + 1e-12))
    left = [C.CHANNELS.index(ch) for ch in C.LEFT_HEMISPHERE]
    right = [C.CHANNELS.index(ch) for ch in C.RIGHT_HEMISPHERE]
    asymmetry = log_abs[:, right, :] - log_abs[:, left, :]  # (N, 7, B)
    hj = hjorth(X)

    blocks, names = [], []
    def add(values, label_fn):
        values = values.reshape(len(X), -1)
        blocks.append(values)
        names.extend(label_fn(i) for i in range(values.shape[1]))
    add(log_abs, lambda i: f"logpow_{C.CHANNELS[i // len(bands)]}_{bands[i % len(bands)]}")
    add(relative, lambda i: f"relpow_{C.CHANNELS[i // len(bands)]}_{bands[i % len(bands)]}")
    add(theta_alpha, lambda i: f"theta_alpha_{C.CHANNELS[i]}")
    add(engagement, lambda i: f"engagement_{C.CHANNELS[i]}")
    add(asymmetry, lambda i: f"asym_{C.RIGHT_HEMISPHERE[i // len(bands)]}-{C.LEFT_HEMISPHERE[i // len(bands)]}_{bands[i % len(bands)]}")
    add(hj, lambda i: f"hjorth_{C.CHANNELS[i // 2]}_{('mobility', 'complexity')[i % 2]}")
    return np.concatenate(blocks, axis=1).astype(np.float64), names
