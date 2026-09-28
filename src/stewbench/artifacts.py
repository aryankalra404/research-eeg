"""Artifact bank and artifact-counterfactual generation (Model E / ACCT).

Everything is estimated from the *training* subjects of a fold, pooled over
both classes so that the artifacts carry no label information, and operates
on model inputs (per-window z-scored windows, shape (N, C, T)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from . import constants as C

FRONTAL = ("AF3", "AF4", "F7", "F8", "F3", "F4")
OCULAR_BAND = (0.0, 4.0)
EMG_BAND = (20.0, 45.0)


@dataclass
class ArtifactBank:
    ocular_topographies: np.ndarray  # (K, C) unit-norm spatial patterns
    ocular_waveforms: np.ndarray     # (K, M, T) real component time courses
    emg_profile: np.ndarray          # (C,) relative excess high-frequency amplitude, max 1
    emg_amplitude: float             # typical EMG amplitude (input units)

    def summary(self) -> dict:
        return {
            "ocular_topographies": {f"component_{k}": dict(zip(C.CHANNELS, np.round(t, 3).tolist()))
                                    for k, t in enumerate(self.ocular_topographies)},
            "ocular_waveform_bank_size": int(self.ocular_waveforms.shape[1]),
            "emg_profile": dict(zip(C.CHANNELS, np.round(self.emg_profile, 3).tolist())),
            "emg_amplitude": float(self.emg_amplitude),
        }


def _band(X: np.ndarray, band: tuple[float, float], sfreq: int) -> np.ndarray:
    lo, hi = band
    if lo <= 0:
        sos = butter(4, hi, btype="lowpass", fs=sfreq, output="sos")
    else:
        sos = butter(4, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def fit_artifact_bank(X: np.ndarray, n_ocular: int = 2, bank_fraction: float = 0.3,
                      sfreq: int = C.SFREQ, max_windows: int = 4000, seed: int = 0) -> ArtifactBank:
    """``X``: training-subject model inputs (both classes pooled)."""
    rng = np.random.default_rng(seed)
    if len(X) > max_windows:
        X = X[rng.choice(len(X), max_windows, replace=False)]
    X = X.astype(np.float64)

    # --- ocular: frontal-dominant principal components of <4 Hz activity
    low = _band(X, OCULAR_BAND, sfreq)
    covariance = np.einsum("nct,ndt->cd", low, low) / (low.shape[0] * low.shape[2])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:6]
    frontal = np.array([ch in FRONTAL for ch in C.CHANNELS])
    candidates = []
    for idx in order:
        v = eigenvectors[:, idx]
        frontal_share = float((v[frontal] ** 2).sum() / (v ** 2).sum())
        candidates.append((frontal_share, idx))
    chosen = [idx for share, idx in sorted(candidates, reverse=True)[:n_ocular]]
    topographies = np.stack([eigenvectors[:, idx] for idx in chosen])
    # Sign convention: positive at the most frontal-loaded channel (arbitrary but fixed).
    for k in range(len(topographies)):
        if topographies[k][np.argmax(np.abs(topographies[k]))] < 0:
            topographies[k] *= -1

    projections = np.einsum("kc,nct->knt", topographies, low)  # (K, N, T)
    waveforms = []
    for k in range(len(topographies)):
        strength = projections[k].std(axis=-1)
        keep = strength >= np.quantile(strength, 1.0 - bank_fraction)
        wave = projections[k][keep]
        waveforms.append(wave - wave.mean(axis=-1, keepdims=True))
    n_bank = min(len(w) for w in waveforms)
    waveforms = np.stack([w[:n_bank] for w in waveforms]).astype(np.float32)

    # --- EMG: channels with excess 20-45 Hz amplitude in their noisiest windows
    high = _band(X, EMG_BAND, sfreq)
    amplitude = high.std(axis=-1)  # (N, C)
    excess = np.quantile(amplitude, 0.9, axis=0) - np.median(amplitude, axis=0)
    excess = np.clip(excess, 0, None)
    profile = excess / max(excess.max(), 1e-8)
    emg_amplitude = float(np.quantile(amplitude, 0.9, axis=0).max())

    return ArtifactBank(topographies.astype(np.float32), waveforms, profile.astype(np.float32), emg_amplitude)


def _fft_band_mask(n_times: int, band: tuple[float, float], sfreq: int, device) -> torch.Tensor:
    freqs = torch.fft.rfftfreq(n_times, d=1.0 / sfreq).to(device)
    lo, hi = band
    return ((freqs >= lo) & (freqs <= hi)).float()


class ArtifactCounterfactual:
    """Label-independent counterfactual transform for a batch (B, C, T):
    add ocular artifacts, remove the ocular subspace from low-frequency
    content, and/or add EMG, then re-z-score every window/channel."""

    def __init__(self, bank: ArtifactBank, p_add_ocular: float = 0.5, p_remove_ocular: float = 0.3,
                 p_emg: float = 0.5, magnitude: float = 1.0, sfreq: int = C.SFREQ):
        self.bank = bank
        self.p_add, self.p_remove, self.p_emg = p_add_ocular, p_remove_ocular, p_emg
        self.magnitude = magnitude
        self.sfreq = sfreq
        self._cache: dict = {}

    def _tensors(self, device, n_times):
        key = (str(device), n_times)
        if key not in self._cache:
            topo = torch.as_tensor(self.bank.ocular_topographies, device=device)
            waves = torch.as_tensor(self.bank.ocular_waveforms[..., :n_times], device=device)
            profile = torch.as_tensor(self.bank.emg_profile, device=device)
            low_mask = _fft_band_mask(n_times, OCULAR_BAND, self.sfreq, device)
            emg_mask = _fft_band_mask(n_times, EMG_BAND, self.sfreq, device)
            self._cache[key] = (topo, waves, profile, low_mask, emg_mask)
        return self._cache[key]

    def __call__(self, x: torch.Tensor, generator: torch.Generator, magnitude: float | None = None) -> torch.Tensor:
        magnitude = self.magnitude if magnitude is None else magnitude
        batch, channels, n_times = x.shape
        device = x.device
        topo, waves, profile, low_mask, emg_mask = self._tensors(device, n_times)
        out = x.float().clone()

        def draw(*shape):
            return torch.rand(shape, generator=generator).to(device)

        # 1) remove the ocular subspace from the low-frequency content
        remove = draw(batch) < self.p_remove
        if remove.any():
            low = torch.fft.irfft(torch.fft.rfft(out, dim=-1) * low_mask, n=n_times, dim=-1)
            basis = torch.linalg.qr(topo.T).Q  # (C, K) orthonormal
            ocular = basis @ (basis.T @ low)  # (B, C, T)
            out = torch.where(remove[:, None, None], out - ocular, out)

        # 2) add real ocular waveforms along the bank topographies
        add = draw(batch) < self.p_add
        if add.any():
            k_count, n_bank = waves.shape[0], waves.shape[1]
            picks = (draw(batch, k_count) * n_bank).long().clamp_max(n_bank - 1)
            amplitudes = (0.5 + draw(batch, k_count)) * magnitude * (draw(batch, k_count) < 0.7).float()
            signs = torch.where(draw(batch, k_count) < 0.5, -1.0, 1.0)
            chosen = waves[torch.arange(k_count, device=device)[None, :], picks]  # (B, K, T)
            ocular = torch.einsum("kc,bkt->bct", topo, chosen * (amplitudes * signs)[..., None])
            out = torch.where(add[:, None, None], out + ocular, out)

        # 3) add band-limited EMG with a smooth random envelope
        emg = draw(batch) < self.p_emg
        if emg.any():
            noise = torch.randn((batch, channels, n_times), generator=generator).to(device)
            noise = torch.fft.irfft(torch.fft.rfft(noise, dim=-1) * emg_mask, n=n_times, dim=-1)
            noise = noise / noise.std(dim=-1, keepdim=True).clamp_min(1e-6)
            knots = draw(batch, 1, 5)
            envelope = torch.nn.functional.interpolate(knots, size=n_times, mode="linear", align_corners=True)
            scale = self.bank.emg_amplitude * magnitude * (0.5 + draw(batch, 1, 1))
            burst = noise * profile[None, :, None] * envelope * scale
            out = torch.where(emg[:, None, None], out + burst, out)

        mean = out.mean(dim=-1, keepdim=True)
        std = out.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return ((out - mean) / std).to(x.dtype)
