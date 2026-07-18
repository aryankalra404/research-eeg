"""
Six baseline architectures for binary EEG classification, sharing one
interface: forward(x) where x is (batch, T, C) float32, T=window_samples,
C=n_channels. All output raw logits of shape (batch, 2).

Internally each model reshapes/permutes x as needed for its own convention
(e.g. EEGNet/DeepConvNet/ShallowConvNet treat EEG as a (1, C, T) "image").

Models:
    1D-CNN          -- simple stacked Conv1d over time, channels as input depth
    VanillaLSTM     -- LSTM over time, channels as per-timestep features
    EEGNetAdapted        -- EEGNet-inspired compact depthwise/separable CNN
    DeepConvNetAdapted   -- adapted Schirrmeister et al. deep architecture
    ShallowConvNetAdapted -- adapted Schirrmeister et al. shallow architecture
    TemporalCNN      -- dilated causal-conv TCN-style network

Usage:
    from src.models import get_model
    model = get_model("eegnet_adapted", n_channels=14, n_timepoints=512, n_classes=2)
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _infer_flattened_size(
    module: nn.Module,
    feature_fn: Callable[[torch.Tensor], torch.Tensor],
    input_shape: tuple[int, ...],
) -> int:
    """Infer a feature size without mutating BatchNorm or dropout state."""
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            output = feature_fn(torch.zeros(input_shape))
    finally:
        module.train(was_training)
    return int(output.flatten(start_dim=1).shape[1])


# ---------------------------------------------------------------------------
# 1. 1D-CNN
# ---------------------------------------------------------------------------
class OneDCNN(nn.Module):
    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, T, C)
        x = x.permute(0, 2, 1)  # -> (B, C, T)
        x = self.net(x).squeeze(-1)  # (B, 128)
        return self.fc(x)


# ---------------------------------------------------------------------------
# 2. Vanilla LSTM
# ---------------------------------------------------------------------------
class VanillaLSTM(nn.Module):
    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int = 2,
                 hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):  # x: (B, T, C) -- already the right shape for LSTM
        out, (h_n, c_n) = self.lstm(x)
        # concat final forward + backward hidden states
        last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, hidden*2)
        return self.fc(last)


# ---------------------------------------------------------------------------
# 3. EEGNet (Lawhern et al. 2018)
# ---------------------------------------------------------------------------
class EEGNetAdapted(nn.Module):
    """EEGNet-inspired adaptation for 4-second, 14-channel STEW windows."""
    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int = 2,
                 F1: int = 8, D: int = 2, F2: int = 16, kernel_length: int = 64,
                 dropout: float = 0.5):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        # Block 1: temporal conv + depthwise spatial conv
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        # Block 2: separable conv
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        flat_size = _infer_flattened_size(
            self, self._forward_features, (1, 1, n_channels, n_timepoints)
        )
        self.classify = nn.Linear(flat_size, n_classes)

    def _forward_features(self, x):
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return x

    def forward(self, x):  # x: (B, T, C)
        x = x.permute(0, 2, 1).unsqueeze(1)  # -> (B, 1, C, T)
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        return self.classify(x)


# ---------------------------------------------------------------------------
# 4. DeepConvNet (Schirrmeister et al. 2017)
# ---------------------------------------------------------------------------
class DeepConvNetAdapted(nn.Module):
    """DeepConvNet adaptation; not an exact reproduction of the paper model."""
    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int = 2,
                 dropout: float = 0.5):
        super().__init__()

        def block(in_c, out_c, kernel_t, pool=True):
            layers = [
                nn.Conv2d(in_c, out_c, (1, kernel_t), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ELU(),
            ]
            if pool:
                layers += [nn.MaxPool2d((1, 3), stride=(1, 3))]
            layers += [nn.Dropout(dropout)]
            return nn.Sequential(*layers)

        self.temporal_conv = nn.Conv2d(1, 25, (1, 10), bias=False)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(25, 25, (n_channels, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3)),
            nn.Dropout(dropout),
        )
        self.block2 = block(25, 50, 10)
        self.block3 = block(50, 100, 10)
        self.block4 = block(100, 200, 10)

        flat_size = _infer_flattened_size(
            self, self._forward_features, (1, 1, n_channels, n_timepoints)
        )
        self.classify = nn.Linear(flat_size, n_classes)

    def _forward_features(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    def forward(self, x):  # x: (B, T, C)
        x = x.permute(0, 2, 1).unsqueeze(1)  # -> (B, 1, C, T)
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        return self.classify(x)


# ---------------------------------------------------------------------------
# 5. ShallowConvNet (Schirrmeister et al. 2017)
# ---------------------------------------------------------------------------
class ShallowConvNetAdapted(nn.Module):
    """ShallowConvNet adaptation for the repository's fixed window format."""
    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, 40, (1, 25), bias=False)
        self.spatial_conv = nn.Conv2d(40, 40, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropout)

        flat_size = _infer_flattened_size(
            self, self._forward_features, (1, 1, n_channels, n_timepoints)
        )
        self.classify = nn.Linear(flat_size, n_classes)

    def _square(self, x):
        return x ** 2

    def _log(self, x):
        return torch.log(torch.clamp(x, min=1e-6))

    def _forward_features(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self._square(x)
        x = self.pool(x)
        x = self._log(x)
        x = self.dropout(x)
        return x

    def forward(self, x):  # x: (B, T, C)
        x = x.permute(0, 2, 1).unsqueeze(1)  # -> (B, 1, C, T)
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        return self.classify(x)


# ---------------------------------------------------------------------------
# 6. TemporalCNN (dilated causal convolutions, TCN-style)
# ---------------------------------------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, dilation, dropout=0.3):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size, padding=padding, dilation=dilation)
        self.chomp = padding  # trim to keep causal (no future leakage)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None
        self.bn1 = nn.BatchNorm1d(out_c)
        self.bn2 = nn.BatchNorm1d(out_c)

    def forward(self, x):
        out = self.conv1(x)[:, :, : -self.chomp if self.chomp > 0 else None]
        out = self.relu(self.bn1(out))
        out = self.dropout(out)
        out = self.conv2(out)[:, :, : -self.chomp if self.chomp > 0 else None]
        out = self.relu(self.bn2(out))
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalCNN(nn.Module):
    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int = 2,
                 channels=(32, 64, 128), kernel_size: int = 5, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_c = n_channels
        for i, out_c in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_c, out_c, kernel_size, dilation, dropout))
            in_c = out_c
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], n_classes)

    def forward(self, x):  # x: (B, T, C)
        x = x.permute(0, 2, 1)  # -> (B, C, T)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "1dcnn": OneDCNN,
    "lstm": VanillaLSTM,
    "eegnet_adapted": EEGNetAdapted,
    "deepconvnet_adapted": DeepConvNetAdapted,
    "shallowconvnet_adapted": ShallowConvNetAdapted,
    "temporalcnn": TemporalCNN,
}

MODEL_ALIASES = {
    "eegnet": "eegnet_adapted",
    "deepconvnet": "deepconvnet_adapted",
    "shallowconvnet": "shallowconvnet_adapted",
}


def get_model(name: str, n_channels: int, n_timepoints: int, n_classes: int = 2, **kwargs):
    name = name.lower()
    name = MODEL_ALIASES.get(name, name)
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY) + list(MODEL_ALIASES)
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name](n_channels=n_channels, n_timepoints=n_timepoints,
                                  n_classes=n_classes, **kwargs)


if __name__ == "__main__":
    # Shape sanity check for every model with a fake batch
    B, T, C = 8, 512, 14
    x = torch.randn(B, T, C)

    for name in MODEL_REGISTRY:
        model = get_model(name, n_channels=C, n_timepoints=T, n_classes=2)
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:16s} output={tuple(out.shape)}  params={n_params:,}")
        assert out.shape == (B, 2), f"{name} produced wrong output shape: {out.shape}"

    print("\nAll models produce correct output shape (B, 2).")
