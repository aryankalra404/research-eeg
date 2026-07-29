"""Leakage-safe conventional EEG augmentation controls."""

from __future__ import annotations

import numpy as np


def augmentation_counts_by_class(
    y: np.ndarray,
    fraction: float,
    *,
    class_ids: tuple[int, ...] = (0, 1),
) -> dict[int, int]:
    """Return a predeclared augmentation count for every class."""
    if not np.isfinite(fraction) or fraction < 0:
        raise ValueError("augmentation fraction must be finite and non-negative.")

    counts = {}
    for class_id in class_ids:
        real_count = int(np.count_nonzero(y == class_id))
        if real_count == 0:
            raise ValueError(f"No real training windows available for class {class_id}.")
        counts[class_id] = int(round(real_count * fraction))
    return counts


def _renormalize_window_channels(windows: np.ndarray) -> np.ndarray:
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    return ((windows - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def generate_simple_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    n_by_class: dict[int, int],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample training windows and apply noise, time-shift, or channel dropout."""
    rng = np.random.default_rng(seed)
    augmented, labels = [], []
    for class_id, n_samples in sorted(n_by_class.items()):
        class_windows = X[y == class_id]
        if n_samples <= 0:
            continue
        if len(class_windows) == 0:
            raise ValueError(f"No real training windows available for class {class_id}.")
        selected = class_windows[
            rng.choice(len(class_windows), size=n_samples, replace=True)
        ].copy()
        methods = rng.integers(0, 3, size=n_samples)
        for index, method in enumerate(methods):
            if method == 0:
                selected[index] += rng.normal(
                    0.0, 0.05, size=selected[index].shape
                ).astype(np.float32)
            elif method == 1:
                shift = int(rng.integers(-selected.shape[1] // 8, selected.shape[1] // 8 + 1))
                selected[index] = np.roll(selected[index], shift=shift, axis=0)
            else:
                channel = int(rng.integers(0, selected.shape[2]))
                selected[index, :, channel] = 0.0
        augmented.append(_renormalize_window_channels(selected))
        labels.append(np.full(n_samples, class_id, dtype=np.int64))
    if not augmented:
        return (
            np.empty((0, X.shape[1], X.shape[2]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.concatenate(augmented), np.concatenate(labels)
