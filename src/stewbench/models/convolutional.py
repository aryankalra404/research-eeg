"""Convolutional EEG baselines.

Kernel/pool lengths quoted in the original papers for 250 Hz data are
rescaled to STEW's 128 Hz with ``scaled_kernel`` so they cover the same
duration in seconds. EEGNet was designed for 128 Hz and is used unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import constants as C
from .common import (
    TCN,
    Conv2dMaxNorm,
    LinearMaxNorm,
    infer_output_size,
    scaled_kernel,
)


class CNN1D(nn.Module):
    """Generic three-block 1D CNN over time with channels as input features."""

    def __init__(self, n_channels, n_times, n_classes=2, sfreq=C.SFREQ, dropout=0.3):
        super().__init__()
        def block(cin, cout, k):
            return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout),
                                 nn.ReLU(), nn.MaxPool1d(2))
        self.features = nn.Sequential(block(n_channels, 32, 7), block(32, 64, 5), block(64, 128, 3),
                                      nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout))
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al., J. Neural Eng. 2018). Dropout 0.25 is the
    value the authors recommend for cross-subject evaluation."""

    def __init__(self, n_channels, n_times, n_classes=2, sfreq=C.SFREQ,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            Conv2dMaxNorm(F1, F1 * D, (n_channels, 1), groups=F1, bias=False, max_norm=1.0),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        size = infer_output_size(self, self._features, (1, n_channels, n_times))
        self.classifier = LinearMaxNorm(size, n_classes, max_norm=0.25)

    def _features(self, x):
        return self.block2(self.block1(x.unsqueeze(1))).flatten(1)

    def forward(self, x):
        return self.classifier(self._features(x))


class ShallowConvNet(nn.Module):
    """Shallow ConvNet (Schirrmeister et al., Hum. Brain Mapp. 2017):
    temporal conv -> spatial conv -> square -> mean pool -> log."""

    def __init__(self, n_channels, n_times, n_classes=2, sfreq=C.SFREQ,
                 n_filters=40, dropout=0.5):
        super().__init__()
        k, pool, stride = (scaled_kernel(25, sfreq), scaled_kernel(75, sfreq), scaled_kernel(15, sfreq))
        self.temporal = nn.Conv2d(1, n_filters, (1, k))
        self.spatial = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters, momentum=0.1)
        self.pool = nn.AvgPool2d((1, pool), stride=(1, stride))
        self.drop = nn.Dropout(dropout)
        size = infer_output_size(self, self._features, (1, n_channels, n_times))
        self.classifier = nn.Linear(size, n_classes)

    def _features(self, x):
        x = self.bn(self.spatial(self.temporal(x.unsqueeze(1))))
        x = torch.log(torch.clamp(self.pool(x * x), min=1e-6))
        return self.drop(x).flatten(1)

    def forward(self, x):
        return self.classifier(self._features(x))


class DeepConvNet(nn.Module):
    """Deep ConvNet (Schirrmeister et al., 2017): four conv-pool blocks."""

    def __init__(self, n_channels, n_times, n_classes=2, sfreq=C.SFREQ, dropout=0.5):
        super().__init__()
        k = scaled_kernel(10, sfreq)

        def block(cin, cout):
            return nn.Sequential(nn.Dropout(dropout), nn.Conv2d(cin, cout, (1, k), bias=False),
                                 nn.BatchNorm2d(cout), nn.ELU(), nn.MaxPool2d((1, 3), (1, 3)))
        self.first = nn.Sequential(
            nn.Conv2d(1, 25, (1, k)), nn.Conv2d(25, 25, (n_channels, 1), bias=False),
            nn.BatchNorm2d(25), nn.ELU(), nn.MaxPool2d((1, 3), (1, 3)),
        )
        self.blocks = nn.Sequential(block(25, 50), block(50, 100), block(100, 200))
        size = infer_output_size(self, self._features, (1, n_channels, n_times))
        self.classifier = nn.Linear(size, n_classes)

    def _features(self, x):
        return self.blocks(self.first(x.unsqueeze(1))).flatten(1)

    def forward(self, x):
        return self.classifier(self._features(x))


class EEGTCNet(nn.Module):
    """EEG-TCNet (Ingolfsson et al., IEEE SMC 2020): EEGNet front-end followed
    by a temporal convolutional network; the last TCN time step is classified."""

    def __init__(self, n_channels, n_times, n_classes=2, sfreq=C.SFREQ,
                 F1=8, D=2, F2=16, eegnet_dropout=0.2, tcn_filters=12, tcn_kernel=4,
                 tcn_depth=2, tcn_dropout=0.3):
        super().__init__()
        ke = scaled_kernel(32, sfreq)
        self.eegnet = nn.Sequential(
            nn.Conv2d(1, F1, (1, ke), padding="same", bias=False), nn.BatchNorm2d(F1),
            Conv2dMaxNorm(F1, F1 * D, (n_channels, 1), groups=F1, bias=False, max_norm=1.0),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(eegnet_dropout),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(eegnet_dropout),
        )
        self.tcn = TCN(F2, tcn_filters, tcn_kernel, tcn_depth, tcn_dropout)
        self.classifier = LinearMaxNorm(tcn_filters, n_classes, max_norm=0.25)

    def forward(self, x):
        features = self.eegnet(x.unsqueeze(1)).squeeze(2)  # (B, F2, T')
        return self.classifier(self.tcn(features)[..., -1])


class TSception(nn.Module):
    """TSception (Ding et al., IEEE Trans. Affective Computing 2023):
    multi-scale temporal kernels (0.5, 0.25, 0.125 s) plus global and
    hemispheric spatial kernels. Channels are reordered left-then-right so
    the hemisphere kernels see homologous electrodes (Emotiv has no midline)."""

    def __init__(self, n_channels, n_times, n_classes=2, sfreq=C.SFREQ,
                 num_T=15, num_S=15, hidden=32, dropout=0.5, pool=8):
        super().__init__()
        if n_channels != C.N_CHANNELS:
            raise ValueError("TSception hemisphere layout is defined for the 14 STEW channels")
        order = [C.CHANNELS.index(ch) for ch in C.LEFT_HEMISPHERE + C.RIGHT_HEMISPHERE]
        self.register_buffer("order", torch.tensor(order), persistent=False)

        def temporal(k):
            return nn.Sequential(nn.Conv2d(1, num_T, (1, k)), nn.LeakyReLU(),
                                 nn.AvgPool2d((1, pool), (1, pool)))
        self.t1, self.t2, self.t3 = (temporal(int(r * sfreq)) for r in (0.5, 0.25, 0.125))
        self.bn_t = nn.BatchNorm2d(num_T)
        half = n_channels // 2
        self.s1 = nn.Sequential(nn.Conv2d(num_T, num_S, (n_channels, 1)), nn.LeakyReLU(),
                                nn.AvgPool2d((1, pool // 4), (1, pool // 4)))
        self.s2 = nn.Sequential(nn.Conv2d(num_T, num_S, (half, 1), stride=(half, 1)), nn.LeakyReLU(),
                                nn.AvgPool2d((1, pool // 4), (1, pool // 4)))
        self.bn_s = nn.BatchNorm2d(num_S)
        self.fusion = nn.Sequential(nn.Conv2d(num_S, num_S, (3, 1)), nn.LeakyReLU(),
                                    nn.AvgPool2d((1, 4), (1, 4)))
        self.bn_f = nn.BatchNorm2d(num_S)
        self.head = nn.Sequential(nn.Linear(num_S, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_classes))

    def forward(self, x):
        x = x[:, self.order].unsqueeze(1)
        x = self.bn_t(torch.cat([self.t1(x), self.t2(x), self.t3(x)], dim=-1))
        x = self.bn_s(torch.cat([self.s1(x), self.s2(x)], dim=2))
        x = self.bn_f(self.fusion(x))
        return self.head(x.mean(dim=(-2, -1)))


class CNNLSTM(nn.Module):
    """Convolutional feature extractor followed by an LSTM, the hybrid most
    frequently reported on STEW."""

    def __init__(self, n_channels, n_times, n_classes=2, sfreq=C.SFREQ, hidden=64, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x):
        features = self.conv(x).transpose(1, 2)
        _, (h, _) = self.lstm(features)
        return self.classifier(F.dropout(h[-1], 0.3, self.training))
