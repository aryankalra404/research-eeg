"""Turn filtered windows into model inputs.

* ``window_zscore``: every window/channel is standardized with its own mean
  and standard deviation. No statistic crosses a window boundary, so the
  transform is identical at training and test time and cannot leak labels.
  Note: this removes absolute amplitude; relative spectral shape is kept.

* Euclidean Alignment (He & Wu, IEEE TBME 2020): per subject, whiten windows
  with the inverse square root of that subject's mean spatial covariance.
  It uses no labels, but it does use all (unlabelled) windows of the test
  subject, i.e. it is transductive. Report it as a separate condition.

Per-recording normalization is deliberately NOT offered: in STEW each
recording is exactly one class, so recording-wise statistics encode the label.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import fractional_matrix_power

from ..settings import InputConfig


def window_zscore(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    return ((X - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def euclidean_alignment(X: np.ndarray, subject: np.ndarray) -> np.ndarray:
    aligned = np.empty_like(X, dtype=np.float32)
    for sid in np.unique(subject):
        mask = subject == sid
        Xs = X[mask].astype(np.float64)
        covariance = np.einsum("nct,ndt->cd", Xs, Xs) / (Xs.shape[0] * Xs.shape[2])
        covariance += 1e-6 * np.trace(covariance) / len(covariance) * np.eye(len(covariance))
        whitening = np.real(fractional_matrix_power(covariance, -0.5))
        aligned[mask] = np.einsum("cd,ndt->nct", whitening, Xs).astype(np.float32)
    return aligned


def model_inputs(X: np.ndarray, subject: np.ndarray, config: InputConfig) -> np.ndarray:
    out = window_zscore(X) if config.normalization == "window_zscore" else X.astype(np.float32)
    if config.alignment == "euclidean":
        out = euclidean_alignment(out, subject)
    return out
