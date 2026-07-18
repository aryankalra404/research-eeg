"""Quantitative real-vs-synthetic EEG checks saved with every GAN run."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

from . import config


def _safe_relative_error(real: np.ndarray, synth: np.ndarray) -> float:
    denominator = np.linalg.norm(real) + 1e-8
    return float(np.linalg.norm(synth - real) / denominator)


def _lag_one_autocorrelation(x: np.ndarray) -> np.ndarray:
    left = x[:, :-1, :].reshape(-1, x.shape[2])
    right = x[:, 1:, :].reshape(-1, x.shape[2])
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    numerator = np.mean(left * right, axis=0)
    denominator = left.std(axis=0) * right.std(axis=0) + 1e-8
    return numerator / denominator


def _channel_covariance(x: np.ndarray) -> np.ndarray:
    return np.cov(x.reshape(-1, x.shape[2]), rowvar=False)


def _band_powers(x: np.ndarray, fs: int) -> dict[str, np.ndarray]:
    frequencies, psd = welch(x, fs=fs, axis=1, nperseg=min(256, x.shape[1]))
    mean_psd = psd.mean(axis=0)
    powers = {}
    for name, (low, high) in config.FREQ_BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        powers[name] = trapezoid(mean_psd[mask], frequencies[mask], axis=0)
    return powers


def evaluate_synthetic_quality(
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    fs: int = config.EEG_SAMPLING_RATE,
    max_samples_per_class: int = 1000,
) -> dict:
    """Return interpretable distribution checks for every generated class."""
    report: dict[str, object] = {
        "note": (
            "These diagnostics detect obvious distribution mismatch; they do not "
            "prove that synthetic EEG is physiologically valid."
        ),
        "classes": {},
    }

    for class_id in sorted(np.unique(y_synth).astype(int).tolist()):
        real = X_real[y_real == class_id][:max_samples_per_class]
        synth = X_synth[y_synth == class_id][:max_samples_per_class]
        if len(real) == 0 or len(synth) == 0:
            continue

        real_bands = _band_powers(real, fs)
        synth_bands = _band_powers(synth, fs)
        band_errors = {
            band: _safe_relative_error(real_bands[band], synth_bands[band])
            for band in config.FREQ_BANDS
        }

        report["classes"][str(class_id)] = {
            "n_real_evaluated": int(len(real)),
            "n_synthetic_evaluated": int(len(synth)),
            "global_mean_real": float(real.mean()),
            "global_mean_synthetic": float(synth.mean()),
            "global_std_real": float(real.std()),
            "global_std_synthetic": float(synth.std()),
            "channel_covariance_relative_error": _safe_relative_error(
                _channel_covariance(real), _channel_covariance(synth)
            ),
            "lag1_autocorrelation_relative_error": _safe_relative_error(
                _lag_one_autocorrelation(real), _lag_one_autocorrelation(synth)
            ),
            "band_power_relative_error": band_errors,
        }

    return report
