"""Graph neural network baselines with electrodes as nodes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import constants as C
from .common import LogBandPower, electrode_adjacency


class ElectrodeGCN(nn.Module):
    """Per-electrode temporal CNN encoder followed by two GCN layers
    (Kipf & Welling, 2017) over a fixed 3-nearest-neighbour electrode graph."""

    def __init__(self, n_channels, n_times, n_classes=2, sfreq=C.SFREQ, hidden=64, dropout=0.3):
        super().__init__()
        if n_channels != C.N_CHANNELS:
            raise ValueError("ElectrodeGCN graph is defined for the 14 STEW electrodes")
        self.register_buffer("adjacency", electrode_adjacency(), persistent=False)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, hidden, 7, padding=3), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.gc1 = nn.Linear(hidden, hidden)
        self.gc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x):
        batch, channels, times = x.shape
        nodes = self.encoder(x.reshape(batch * channels, 1, times)).reshape(batch, channels, -1)
        nodes = self.drop(F.relu(self.gc1(self.adjacency @ nodes)))
        nodes = F.relu(self.gc2(self.adjacency @ nodes))
        return self.classifier(nodes.mean(dim=1))


class DGCNN(nn.Module):
    """Dynamical graph CNN (Song et al., IEEE Trans. Affective Computing 2018):
    Chebyshev graph convolution over band-power node features with a
    *learnable* adjacency matrix. Band powers are computed inside the model."""

    def __init__(self, n_channels, n_times, n_classes=2, sfreq=C.SFREQ,
                 K=2, hidden=32, dropout=0.3):
        super().__init__()
        self.features = LogBandPower(n_times, sfreq)
        n_bands = len(C.FREQ_BANDS)
        self.input_norm = nn.BatchNorm1d(n_channels * n_bands)
        self.adjacency = nn.Parameter(torch.empty(n_channels, n_channels))
        nn.init.xavier_uniform_(self.adjacency)
        self.K = K
        self.cheb = nn.ModuleList([nn.Linear(n_bands, hidden, bias=False) for _ in range(K)])
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(n_channels * hidden, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, n_classes))

    def _laplacian(self):
        adjacency = F.relu(self.adjacency)
        adjacency = (adjacency + adjacency.T) / 2
        degree = adjacency.sum(1)
        inv_sqrt = torch.where(degree > 0, degree.clamp_min(1e-6).rsqrt(), torch.zeros_like(degree))
        eye = torch.eye(len(adjacency), device=adjacency.device)
        return eye - inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]

    def forward(self, x):
        batch = x.shape[0]
        nodes = self.features(x)  # (B, C, bands)
        nodes = self.input_norm(nodes.reshape(batch, -1)).reshape(nodes.shape)
        laplacian = self._laplacian()
        # Chebyshev polynomials T0 = I, T1 = L, Tk = 2 L T(k-1) - T(k-2) (L not rescaled,
        # following the reference DGCNN implementation).
        terms = [nodes]
        if self.K > 1:
            terms.append(laplacian @ nodes)
        for _ in range(2, self.K):
            terms.append(2 * laplacian @ terms[-1] - terms[-2])
        out = sum(layer(term) for layer, term in zip(self.cheb, terms)) + self.bias
        return self.head(F.relu(out))
