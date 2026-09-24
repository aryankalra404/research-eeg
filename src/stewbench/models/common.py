"""Building blocks shared by several EEG architectures.

All models take input of shape (batch, channels, times) and return logits of
shape (batch, n_classes).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import constants as C


def scaled_kernel(samples_at_250hz: int, sfreq: float) -> int:
    """Rescale a kernel defined for 250 Hz data (BCI-IV-2a, where most EEG
    CNNs were designed) so it spans the same duration at ``sfreq``."""
    return max(1, int(round(samples_at_250hz * sfreq / 250.0)))


class Conv2dMaxNorm(nn.Conv2d):
    """Conv2d whose per-filter weight norm is clipped (Lawhern et al., 2018)."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        with torch.no_grad():
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LinearMaxNorm(nn.Linear):
    def __init__(self, *args, max_norm: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        with torch.no_grad():
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation,
                         padding=(kernel_size - 1) * dilation, **kwargs)
        self._chomp = (kernel_size - 1) * dilation

    def forward(self, x):
        out = super().forward(x)
        return out[..., :-self._chomp] if self._chomp else out


class TCNBlock(nn.Module):
    """Residual block of a temporal convolutional network (Bai et al., 2018)
    with batch norm and ELU as used in EEG-TCNet / ATCNet."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout)
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        out = self.drop(F.elu(self.bn1(self.conv1(x))))
        out = self.drop(F.elu(self.bn2(self.conv2(out))))
        return F.elu(out + self.residual(x))


class TCN(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, depth, dropout):
        super().__init__()
        self.blocks = nn.Sequential(*[
            TCNBlock(in_channels if i == 0 else filters, filters, kernel_size, 2 ** i, dropout)
            for i in range(depth)
        ])

    def forward(self, x):  # (B, F, T)
        return self.blocks(x)


class STFTSpectrogram(nn.Module):
    """Deterministic log-magnitude STFT computed inside the model so real and
    synthetic raw EEG always receive the identical transform."""

    def __init__(self, sfreq: float = C.SFREQ, n_fft: int = 64, hop_length: int = 16,
                 max_frequency_hz: float = 45.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_bin = min(n_fft // 2 + 1, int(max_frequency_hz * n_fft / sfreq) + 1)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, x):  # (B, C, T) -> (B, C, F, frames)
        batch, channels, times = x.shape
        spectrum = torch.stft(x.reshape(batch * channels, times).float(), n_fft=self.n_fft,
                              hop_length=self.hop_length, window=self.window, return_complex=True)
        spectrum = torch.log1p(spectrum.abs())[:, : self.max_bin]
        spectrum = spectrum.reshape(batch, channels, spectrum.shape[-2], spectrum.shape[-1])
        mean = spectrum.mean(dim=(-2, -1), keepdim=True)
        std = spectrum.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (spectrum - mean) / std


class LogBandPower(nn.Module):
    """Differentiable per-channel log band power (Hann-windowed periodogram)
    in the canonical bands; equivalent up to a constant to differential
    entropy for Gaussian signals (Duan et al., 2013)."""

    def __init__(self, n_times: int, sfreq: float = C.SFREQ, bands=None):
        super().__init__()
        bands = bands or C.FREQ_BANDS
        freqs = torch.fft.rfftfreq(n_times, d=1.0 / sfreq)
        masks = torch.stack([((freqs >= lo) & (freqs < hi)).float() for lo, hi in bands.values()])
        self.register_buffer("masks", masks / masks.sum(dim=1, keepdim=True).clamp_min(1.0),
                             persistent=False)
        self.register_buffer("window", torch.hann_window(n_times), persistent=False)

    def forward(self, x):  # (B, C, T) -> (B, C, n_bands)
        spectrum = torch.fft.rfft(x.float() * self.window, dim=-1).abs() ** 2
        return torch.log(spectrum @ self.masks.T + 1e-8)


def electrode_adjacency(neighbors: int = 3) -> torch.Tensor:
    """Symmetric k-nearest-neighbour graph over the 14 Emotiv electrodes
    (MNE standard_1020 coordinates) with self-loops, symmetrically normalized."""
    positions = torch.tensor(C.CHANNEL_POSITIONS, dtype=torch.float32)
    n = len(positions)
    nearest = torch.cdist(positions, positions).topk(k=neighbors + 1, largest=False).indices
    adjacency = torch.zeros(n, n).scatter_(1, nearest, 1.0)
    adjacency = torch.maximum(adjacency, adjacency.T)
    adjacency.fill_diagonal_(1.0)
    inv_sqrt = adjacency.sum(1).rsqrt()
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


def infer_output_size(module: nn.Module, forward_fn, input_shape) -> int:
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            out = forward_fn(torch.zeros(input_shape))
    finally:
        module.train(was_training)
    return int(math.prod(out.shape[1:]))
