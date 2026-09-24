"""Online (per-batch) EEG data augmentations.

Definitions follow the systematic comparison of Rommel et al. (J. Neural
Eng. 2022) and the braindecode implementations. Each transform is applied
independently to every sample with probability ``p``; ``magnitude`` in
[0, 1] scales its strength. Inputs are (B, C, T) tensors that have already
been normalized, and randomness is drawn from a CPU ``torch.Generator``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _apply_mask(p: float, batch: int, generator: torch.Generator, device) -> torch.Tensor:
    return (torch.rand(batch, generator=generator) < p).to(device)


def _uniform(shape, low, high, generator, device):
    return (torch.rand(shape, generator=generator) * (high - low) + low).to(device)


def gaussian_noise(x, y, g, p=0.5, magnitude=0.5):
    std = 0.2 * magnitude  # inputs are z-scored, so 0.1 = 10% of signal SD
    noise = torch.randn(x.shape, generator=g).to(x.device) * std
    return torch.where(_apply_mask(p, len(x), g, x.device)[:, None, None], x + noise, x), y


def smooth_time_mask(x, y, g, p=0.5, magnitude=0.5, sfreq=128):
    batch, _, times = x.shape
    length = max(1, int(magnitude * 0.25 * times))  # up to 25% of the window
    start = (torch.rand(batch, generator=g) * (times - length)).to(x.device)[:, None, None]
    t = torch.arange(times, device=x.device, dtype=x.dtype)[None, None, :]
    slope = 2.0  # sigmoid transition width of a few samples
    mask = torch.sigmoid(slope * (t - start - length)) + torch.sigmoid(-slope * (t - start))
    return torch.where(_apply_mask(p, batch, g, x.device)[:, None, None], x * mask, x), y


def channel_dropout(x, y, g, p=0.5, magnitude=0.5):
    keep = (torch.rand(x.shape[:2], generator=g) >= 0.4 * magnitude).to(x.device, x.dtype)
    return torch.where(_apply_mask(p, len(x), g, x.device)[:, None, None], x * keep[..., None], x), y


def ft_surrogate(x, y, g, p=0.5, magnitude=0.5):
    """Fourier-transform surrogate (Schwabedal et al., 2018): random phase
    perturbation shared across channels, preserving the power spectrum."""
    spectrum = torch.fft.rfft(x.float(), dim=-1)
    phases = _uniform((len(x), 1, spectrum.shape[-1]), 0, 2 * math.pi * magnitude, g, x.device)
    phases[..., 0] = 0.0  # keep DC
    if x.shape[-1] % 2 == 0:
        phases[..., -1] = 0.0  # keep Nyquist real
    surrogate = torch.fft.irfft(spectrum * torch.exp(1j * phases), n=x.shape[-1], dim=-1).to(x.dtype)
    return torch.where(_apply_mask(p, len(x), g, x.device)[:, None, None], surrogate, x), y


def frequency_shift(x, y, g, p=0.5, magnitude=0.5, sfreq=128):
    """Shift all frequencies by delta ~ U(-2m, 2m) Hz via the analytic signal."""
    times = x.shape[-1]
    spectrum = torch.fft.fft(x.float(), dim=-1)
    h = torch.zeros(times, device=x.device)
    h[0] = 1.0
    if times % 2 == 0:
        h[times // 2] = 1.0
        h[1:times // 2] = 2.0
    else:
        h[1:(times + 1) // 2] = 2.0
    analytic = torch.fft.ifft(spectrum * h, dim=-1)
    shift = _uniform((len(x), 1, 1), -2.0 * magnitude, 2.0 * magnitude, g, x.device)
    t = torch.arange(times, device=x.device) / sfreq
    shifted = (analytic * torch.exp(2j * math.pi * shift * t)).real.to(x.dtype)
    return torch.where(_apply_mask(p, len(x), g, x.device)[:, None, None], shifted, x), y


def sign_flip(x, y, g, p=0.5, magnitude=0.5):
    return torch.where(_apply_mask(p, len(x), g, x.device)[:, None, None], -x, x), y


def time_reverse(x, y, g, p=0.5, magnitude=0.5):
    return torch.where(_apply_mask(p, len(x), g, x.device)[:, None, None], x.flip(-1), x), y


def mixup(x, y, g, p=0.5, magnitude=0.5, n_classes=2):
    """Mixup (Zhang et al., ICLR 2018) with soft targets; alpha = magnitude."""
    alpha = max(magnitude, 1e-3)
    lam = torch.distributions.Beta(alpha, alpha).sample((len(x),)).to(x.device)
    lam = torch.where(_apply_mask(p, len(x), g, x.device), torch.maximum(lam, 1 - lam), torch.ones_like(lam))
    perm = torch.randperm(len(x), generator=g).to(x.device)
    onehot = F.one_hot(y, n_classes).float()
    mixed = lam[:, None, None] * x + (1 - lam[:, None, None]) * x[perm]
    target = lam[:, None] * onehot + (1 - lam[:, None]) * onehot[perm]
    return mixed, target


ONLINE_AUGMENTATIONS = {
    "gaussian_noise": gaussian_noise,
    "smooth_time_mask": smooth_time_mask,
    "channel_dropout": channel_dropout,
    "ft_surrogate": ft_surrogate,
    "frequency_shift": frequency_shift,
    "sign_flip": sign_flip,
    "time_reverse": time_reverse,
    "mixup": mixup,
}
GENERATIVE_AUGMENTATIONS = ("cwgan_gp", "ddpm")


def make_online(name: str, p: float, magnitude: float):
    if name not in ONLINE_AUGMENTATIONS:
        raise KeyError(f"Unknown online augmentation {name!r}")
    fn = ONLINE_AUGMENTATIONS[name]

    def augment(x, y, generator):
        return fn(x, y, generator, p=p, magnitude=magnitude)
    return augment
