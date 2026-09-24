"""Recurrent baselines over the raw multichannel sequence (one step = one
sample, 14 features per step)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .. import constants as C


class _Recurrent(nn.Module):
    cell = nn.RNN
    bidirectional = False

    def __init__(self, n_channels, n_times, n_classes=2, sfreq=C.SFREQ,
                 hidden=64, layers=2, dropout=0.3):
        super().__init__()
        kwargs = {"nonlinearity": "tanh"} if self.cell is nn.RNN else {}
        self.rnn = self.cell(n_channels, hidden, num_layers=layers, batch_first=True,
                             dropout=dropout if layers > 1 else 0.0,
                             bidirectional=self.bidirectional, **kwargs)
        self.classifier = nn.Linear(hidden * (2 if self.bidirectional else 1), n_classes)

    def forward(self, x):
        _, state = self.rnn(x.transpose(1, 2))
        h = state[0] if isinstance(state, tuple) else state
        last = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
        return self.classifier(last)


class VanillaRNN(_Recurrent):
    cell = nn.RNN


class LSTM(_Recurrent):
    cell = nn.LSTM


class BiLSTM(_Recurrent):
    cell = nn.LSTM
    bidirectional = True


class GRU(_Recurrent):
    cell = nn.GRU
