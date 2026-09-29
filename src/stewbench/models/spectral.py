"""SpecNet: a differentiable version of the hand-crafted spectral pipeline
that won the benchmark, so it can be trained end-to-end with Model E's
artifact-counterfactual consistency.

Features computed inside the network per window (same families as
``features.spectral_features``): log band power, log relative band power,
log theta/alpha ratio, log engagement index beta/(alpha+theta), and
hemispheric asymmetry per band; then BatchNorm -> small MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .. import constants as C
from .common import WelchLogBandPower


class SpecNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=2, sfreq=C.SFREQ, hidden=32, dropout=0.3):
        super().__init__()
        if n_channels != C.N_CHANNELS:
            raise ValueError("SpecNet asymmetry pairs are defined for the 14 STEW channels")
        self.bandpower = WelchLogBandPower(n_times, sfreq)
        bands = list(C.FREQ_BANDS)
        self.theta, self.alpha, self.beta = bands.index("theta"), bands.index("alpha"), bands.index("beta")
        self.register_buffer("left", torch.tensor([C.CHANNELS.index(c) for c in C.LEFT_HEMISPHERE]), persistent=False)
        self.register_buffer("right", torch.tensor([C.CHANNELS.index(c) for c in C.RIGHT_HEMISPHERE]), persistent=False)
        n_bands = len(bands)
        n_features = n_channels * (2 * n_bands + 2) + len(C.LEFT_HEMISPHERE) * n_bands
        self.head = nn.Sequential(
            nn.BatchNorm1d(n_features), nn.Dropout(dropout),
            nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def features(self, x):
        logp = self.bandpower(x)  # (B, C, bands)
        relative = logp - torch.logsumexp(logp, dim=-1, keepdim=True)
        theta_alpha = logp[..., self.theta] - logp[..., self.alpha]
        engagement = logp[..., self.beta] - torch.logaddexp(logp[..., self.alpha], logp[..., self.theta])
        asymmetry = logp[:, self.right] - logp[:, self.left]
        return torch.cat([logp.flatten(1), relative.flatten(1), theta_alpha, engagement, asymmetry.flatten(1)], dim=1)

    def forward(self, x):
        return self.head(self.features(x))
