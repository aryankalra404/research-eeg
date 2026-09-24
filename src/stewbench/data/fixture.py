"""Synthetic STEW-formatted files for tests and smoke runs ONLY.

The signals are coloured noise plus class-dependent occipital alpha and
frontal theta with random subject-specific gains. They exist so the full
pipeline can be exercised without the real dataset. Never report numbers
obtained on this fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import constants as C


def _pink_noise(rng, n_channels: int, n_samples: int) -> np.ndarray:
    spectrum = rng.normal(size=(n_channels, n_samples // 2 + 1)) + 1j * rng.normal(
        size=(n_channels, n_samples // 2 + 1))
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / C.SFREQ)
    spectrum /= np.maximum(freqs, 1.0) ** 0.5
    signal = np.fft.irfft(spectrum, n=n_samples, axis=-1)
    return signal / signal.std(axis=-1, keepdims=True)


def write_fixture(out_dir: Path, n_subjects: int = 12, seconds: float = 40.0,
                  effect: float = 0.6, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(seconds * C.SFREQ)
    t = np.arange(n_samples) / C.SFREQ
    occipital = np.array([ch in ("O1", "O2", "P7", "P8") for ch in C.CHANNELS], float)
    frontal = np.array([ch in ("AF3", "AF4", "F3", "F4", "F7", "F8") for ch in C.CHANNELS], float)
    ratings = []
    for subject in range(1, n_subjects + 1):
        gain = rng.uniform(5.0, 20.0, size=C.N_CHANNELS)
        subject_alpha = rng.uniform(9.0, 11.5)
        for condition, tag in ((0, "lo"), (1, "hi")):
            noise = _pink_noise(rng, C.N_CHANNELS, n_samples)
            alpha_amp = (1.0 - effect * condition) * rng.uniform(0.8, 1.2)
            theta_amp = (0.3 + effect * condition) * rng.uniform(0.8, 1.2)
            alpha = np.sin(2 * np.pi * subject_alpha * t + rng.uniform(0, 2 * np.pi))
            theta = np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 2 * np.pi))
            signal = noise + np.outer(occipital, alpha) * alpha_amp + np.outer(frontal, theta) * theta_amp
            signal = signal * gain[:, None] + 4200.0
            # Occasional large artifact to exercise rejection.
            if rng.random() < 0.5:
                start = int(rng.integers(0, n_samples - C.SFREQ))
                signal[:2, start:start + C.SFREQ // 4] += 800.0
            np.savetxt(out_dir / f"sub{subject:02d}_{tag}.txt", signal.T, fmt="%.3f")
        if subject not in C.MISSING_RATING_SUBJECTS:
            ratings.append((subject, int(rng.integers(1, 4)), int(rng.integers(5, 10))))
    np.savetxt(out_dir / "ratings.txt", np.array(ratings), fmt="%d", delimiter=", ")
    return out_dir
