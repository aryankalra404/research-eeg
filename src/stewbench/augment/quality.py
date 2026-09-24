"""Quantitative real-vs-synthetic EEG checks.

Fidelity : per-class PSD log-spectral distance, band-power relative error,
           channel-correlation error, lag-1 autocorrelation error, feature MMD.
Diversity: density and coverage (Naeem et al., ICML 2020).
Utility  : train-synthetic/test-real (TSTR) vs train-real/test-real (TRTR)
           with a logistic regression on spectral features, tested on the
           fold's inner-validation subjects (never the test subjects).
Privacy  : nearest-neighbour distance of synthetic windows to the generator's
           training windows vs. to unseen real windows (heuristic only).

These diagnostics can expose mismatch; they do not prove physiological
validity. Downstream held-out classification is the primary evidence.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, pairwise_distances
from sklearn.preprocessing import StandardScaler

from .. import constants as C
from ..features import spectral_features


def _subsample(x: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(x) <= n:
        return x
    return x[np.random.default_rng(seed).choice(len(x), n, replace=False)]


def _psd(x):
    freqs, psd = welch(x, fs=C.SFREQ, nperseg=min(256, x.shape[-1]), axis=-1)
    keep = (freqs >= 1) & (freqs <= 45)
    return freqs[keep], psd[..., keep]


def _rel_err(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def _mmd(a, b):
    z = np.concatenate([a, b])
    d = pairwise_distances(z, metric="sqeuclidean")
    bw = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    k = np.exp(-d / (2 * bw))
    n = len(a)
    return float(max(0.0, k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()))


def _density_coverage(real, fake, k=5):
    d_rr = pairwise_distances(real)
    np.fill_diagonal(d_rr, np.inf)
    radii = np.partition(d_rr, k - 1, axis=1)[:, k - 1]
    d_rf = pairwise_distances(real, fake)
    density = float(((d_rf <= radii[:, None]).sum(0) / k).mean())
    coverage = float((d_rf.min(1) <= radii).mean())
    return density, coverage


def evaluate_quality(X_train, y_train, X_synth, y_synth, X_ref=None, y_ref=None,
                     max_per_class: int = 1000, seed: int = 0) -> dict:
    """``X_train``: generator training windows; ``X_ref``: unseen real windows
    (inner-validation subjects) used for utility and memorization checks."""
    report = {"classes": {}}
    feat_train, _ = spectral_features(X_train)
    scaler = StandardScaler().fit(feat_train)
    for c in (0, 1):
        real = _subsample(X_train[y_train == c], max_per_class, seed)
        fake = _subsample(X_synth[y_synth == c], max_per_class, seed)
        if not len(real) or not len(fake):
            continue
        _, psd_r = _psd(real)
        _, psd_f = _psd(fake)
        mean_r, mean_f = np.log10(psd_r.mean(0)), np.log10(psd_f.mean(0))
        fr, ff = scaler.transform(spectral_features(real)[0]), scaler.transform(spectral_features(fake)[0])
        corr_r = np.corrcoef(real.transpose(1, 0, 2).reshape(real.shape[1], -1))
        corr_f = np.corrcoef(fake.transpose(1, 0, 2).reshape(fake.shape[1], -1))
        ac = lambda x: np.mean(np.sum(x[..., 1:] * x[..., :-1], -1) / np.maximum(np.sum(x * x, -1), 1e-12), 0)
        density, coverage = _density_coverage(fr, ff)
        entry = {
            "n_real": int(len(real)), "n_synthetic": int(len(fake)),
            "log_spectral_distance_db": float(10 * np.sqrt(np.mean((mean_r - mean_f) ** 2))),
            "channel_correlation_rel_error": _rel_err(corr_r, corr_f),
            "lag1_autocorrelation_rel_error": _rel_err(ac(real), ac(fake)),
            "feature_mmd": _mmd(fr, ff),
            "density": density, "coverage": coverage,
            "nn_distance_to_train_median": float(np.median(pairwise_distances(ff, fr).min(1))),
        }
        if X_ref is not None:
            ref = _subsample(X_ref[y_ref == c], max_per_class, seed)
            if len(ref):
                fref = scaler.transform(spectral_features(ref)[0])
                entry["nn_distance_to_unseen_median"] = float(np.median(pairwise_distances(ff, fref).min(1)))
                entry["memorization_ratio"] = entry["nn_distance_to_train_median"] / max(
                    entry["nn_distance_to_unseen_median"], 1e-12)
        report["classes"][str(c)] = entry
    if X_ref is not None and len(np.unique(y_ref)) == 2:
        f_ref = scaler.transform(spectral_features(X_ref)[0])
        def fit_score(F, y):
            clf = LogisticRegression(max_iter=3000, C=0.1).fit(scaler.transform(F), y)
            return float(balanced_accuracy_score(y_ref, clf.predict(f_ref)))
        report["utility"] = {
            "trtr_balanced_accuracy": fit_score(feat_train, y_train),
            "tstr_balanced_accuracy": fit_score(spectral_features(X_synth)[0], y_synth),
        }
    return report


def psd_summary(X, y) -> dict:
    """Mean log10 PSD per class (for real-vs-synthetic figures)."""
    freqs, psd = _psd(X)
    return {"freqs": freqs.tolist(),
            **{str(c): np.log10(psd[y == c].mean(axis=(0, 1)) + 1e-20).tolist() for c in (0, 1) if np.any(y == c)}}
