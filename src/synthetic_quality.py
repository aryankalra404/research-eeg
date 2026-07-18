"""Quantitative real-vs-synthetic EEG checks saved with every GAN run."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from . import config


def _safe_relative_error(real: np.ndarray, synth: np.ndarray) -> float:
    denominator = np.linalg.norm(real) + 1e-8
    return float(np.linalg.norm(synth - real) / denominator)


def _even_subsample(x: np.ndarray, max_samples: int) -> np.ndarray:
    if len(x) <= max_samples:
        return x
    indices = np.linspace(0, len(x) - 1, max_samples, dtype=int)
    return x[indices]


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


def _quality_features(x: np.ndarray, fs: int) -> np.ndarray:
    """Compact time/frequency features used only for distribution diagnostics."""
    frequencies, psd = welch(x, fs=fs, axis=1, nperseg=min(256, x.shape[1]))
    features = [x.mean(axis=1), x.std(axis=1)]
    for low, high in config.FREQ_BANDS.values():
        mask = (frequencies >= low) & (frequencies < high)
        window_power = trapezoid(psd[:, mask, :], frequencies[mask], axis=1)
        features.append(np.log1p(np.maximum(window_power, 0.0)))
    return np.concatenate(features, axis=1)


def _rbf_mmd(real_features: np.ndarray, synth_features: np.ndarray) -> float:
    combined = np.concatenate([real_features, synth_features], axis=0)
    distances = pairwise_distances(combined, metric="sqeuclidean")
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    kernel = np.exp(-distances / max(2.0 * bandwidth, 1e-12))
    n_real = len(real_features)
    k_rr = kernel[:n_real, :n_real]
    k_ss = kernel[n_real:, n_real:]
    k_rs = kernel[:n_real, n_real:]
    return float(max(0.0, k_rr.mean() + k_ss.mean() - 2.0 * k_rs.mean()))


def _density_and_coverage(
    real_features: np.ndarray,
    synth_features: np.ndarray,
    nearest_neighbors: int = 5,
) -> tuple[float, float]:
    k = min(nearest_neighbors, len(real_features) - 1)
    if k < 1:
        return 0.0, 0.0
    real_distances = pairwise_distances(real_features)
    np.fill_diagonal(real_distances, np.inf)
    radii = np.partition(real_distances, k - 1, axis=1)[:, k - 1]
    real_synth_distances = pairwise_distances(real_features, synth_features)
    density = np.mean((real_synth_distances <= radii[:, None]).sum(axis=0) / k)
    coverage = np.mean(real_synth_distances.min(axis=1) <= radii)
    return float(density), float(coverage)


def _nearest_neighbor_summary(source: np.ndarray, target: np.ndarray) -> dict:
    nearest = pairwise_distances(source, target).min(axis=1)
    return {
        "mean": float(nearest.mean()),
        "median": float(np.median(nearest)),
        "p05": float(np.percentile(nearest, 5)),
        "p95": float(np.percentile(nearest, 95)),
    }


def _utility_report(
    real_features: np.ndarray,
    y_real: np.ndarray,
    synth_features: np.ndarray,
    y_synth: np.ndarray,
    reference_features: np.ndarray,
    y_reference: np.ndarray,
) -> dict:
    def fit_and_score(train_x, train_y) -> dict:
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
        classifier.fit(train_x, train_y)
        prediction = classifier.predict(reference_features)
        return {
            "accuracy": float(accuracy_score(y_reference, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y_reference, prediction)),
            "f1_macro": float(f1_score(y_reference, prediction, average="macro", zero_division=0)),
        }

    return {
        "trtr_train_real_test_real_reference": fit_and_score(real_features, y_real),
        "tstr_train_synthetic_test_real_reference": fit_and_score(synth_features, y_synth),
        "note": "A lightweight logistic-regression utility diagnostic; final claims use the held-out deep-classifier comparison.",
    }


def evaluate_synthetic_quality(
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    *,
    fs: int,
    max_samples_per_class: int = 1000,
    X_reference: np.ndarray | None = None,
    y_reference: np.ndarray | None = None,
) -> dict:
    """Return interpretable distribution checks for every generated class."""
    if fs <= 0:
        raise ValueError(f"Sampling rate must be positive, got {fs}.")
    report: dict[str, object] = {
        "sampling_rate_hz": int(fs),
        "note": (
            "These diagnostics detect obvious distribution mismatch; they do not "
            "prove that synthetic EEG is physiologically valid."
        ),
        "classes": {},
    }

    feature_scaler = StandardScaler().fit(_quality_features(X_real, fs))

    for class_id in sorted(np.unique(y_synth).astype(int).tolist()):
        real = _even_subsample(X_real[y_real == class_id], max_samples_per_class)
        synth = _even_subsample(X_synth[y_synth == class_id], max_samples_per_class)
        if len(real) == 0 or len(synth) == 0:
            continue

        real_bands = _band_powers(real, fs)
        synth_bands = _band_powers(synth, fs)
        real_features = feature_scaler.transform(_quality_features(real, fs))
        synth_features = feature_scaler.transform(_quality_features(synth, fs))
        density, coverage = _density_and_coverage(real_features, synth_features)
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
            "feature_rbf_mmd": _rbf_mmd(real_features, synth_features),
            "feature_density": density,
            "feature_coverage": coverage,
            "synthetic_to_gan_train_nearest_neighbor": _nearest_neighbor_summary(
                synth_features, real_features
            ),
        }

        if X_reference is not None and y_reference is not None:
            reference = _even_subsample(
                X_reference[y_reference == class_id], max_samples_per_class
            )
            if len(reference):
                reference_features = feature_scaler.transform(
                    _quality_features(reference, fs)
                )
                train_nn = report["classes"][str(class_id)][
                    "synthetic_to_gan_train_nearest_neighbor"
                ]
                reference_nn = _nearest_neighbor_summary(
                    synth_features, reference_features
                )
                report["classes"][str(class_id)][
                    "synthetic_to_real_reference_nearest_neighbor"
                ] = reference_nn
                report["classes"][str(class_id)]["memorization_distance_ratio"] = float(
                    train_nn["median"] / max(reference_nn["median"], 1e-12)
                )

    if X_reference is not None and y_reference is not None:
        real_features = feature_scaler.transform(_quality_features(X_real, fs))
        synth_features = feature_scaler.transform(_quality_features(X_synth, fs))
        reference_features = feature_scaler.transform(_quality_features(X_reference, fs))
        report["utility"] = _utility_report(
            real_features,
            y_real,
            synth_features,
            y_synth,
            reference_features,
            y_reference,
        )
        report["memorization_note"] = (
            "A ratio far below 1 means generated samples are closer to GAN-training "
            "windows than to unseen real-reference windows. This is a heuristic, not "
            "a formal privacy guarantee."
        )

    return report


def save_synthetic_quality_plots(
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_synth: np.ndarray,
    y_synth: np.ndarray,
    *,
    fs: int,
    output_dir,
    class_names: tuple[str, str],
) -> None:
    """Save PSD and spatial-correlation comparisons for both classes."""
    project_cache = Path(__file__).resolve().parent.parent / ".cache"
    os.environ.setdefault("XDG_CACHE_HOME", str(project_cache))
    os.environ.setdefault("MPLCONFIGDIR", str(project_cache / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for class_id, class_name in enumerate(class_names):
        for data, label, linestyle in (
            (X_real[y_real == class_id], f"Real: {class_name}", "-"),
            (X_synth[y_synth == class_id], f"Synthetic: {class_name}", "--"),
        ):
            frequencies, psd = welch(data, fs=fs, axis=1, nperseg=min(256, data.shape[1]))
            axes[class_id].semilogy(
                frequencies,
                psd.mean(axis=(0, 2)),
                linestyle=linestyle,
                label=label,
            )
        axes[class_id].set(
            title=class_name,
            xlabel="Frequency (Hz)",
            ylabel="Power spectral density",
            xlim=(0, min(45, fs / 2)),
        )
        axes[class_id].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "real_vs_synthetic_psd.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(9, 8))
    for class_id, class_name in enumerate(class_names):
        for column, (data, source) in enumerate((
            (X_real[y_real == class_id], "Real"),
            (X_synth[y_synth == class_id], "Synthetic"),
        )):
            correlation = np.corrcoef(data.reshape(-1, data.shape[2]), rowvar=False)
            image = axes[class_id, column].imshow(
                correlation, vmin=-1, vmax=1, cmap="coolwarm"
            )
            axes[class_id, column].set_title(f"{source}: {class_name}")
    figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025)
    figure.savefig(output_dir / "real_vs_synthetic_channel_correlation.png", dpi=200)
    plt.close(figure)
