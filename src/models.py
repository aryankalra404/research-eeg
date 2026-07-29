"""EEG classifiers sharing a ``forward(B, T, C) -> (B, classes)`` interface.

The active research suite is listed in ``ACTIVE_MODEL_NAMES``. Older
architectures remain registered so their code and historical runs stay usable,
but they are no longer selected by the default baseline command.
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
# Recurrent baselines
# ---------------------------------------------------------------------------
class VanillaRNN(nn.Module):
    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int = 2,
                 hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1])


class VanillaLSTM(nn.Module):
    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int = 2,
                 hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class BidirectionalLSTM(nn.Module):
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

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(torch.cat([h_n[-2], h_n[-1]], dim=1))


class VanillaGRU(nn.Module):
    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int = 2,
                 hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])


# ---------------------------------------------------------------------------
# Graph baseline
# ---------------------------------------------------------------------------
# MNE standard_1020 montage coordinates in config.EMOTIV_14_CHANNELS order.
STEW_CHANNEL_POSITIONS = (
    (-0.033701, 0.076837, 0.021227),
    (-0.070263, 0.042474, -0.011420),
    (-0.050244, 0.053111, 0.042192),
    (-0.077215, 0.018643, 0.024460),
    (-0.084161, -0.016019, -0.009346),
    (-0.072434, -0.073453, -0.002487),
    (-0.029413, -0.112449, 0.008839),
    (0.029843, -0.112156, 0.008800),
    (0.073056, -0.073068, -0.002540),
    (0.085080, -0.015020, -0.009490),
    (0.079534, 0.019936, 0.024438),
    (0.051836, 0.054305, 0.040814),
    (0.073043, 0.044422, -0.012000),
    (0.035712, 0.077726, 0.021956),
)


def _normalized_electrode_adjacency(n_channels: int, neighbors: int = 3) -> torch.Tensor:
    """Build a symmetric k-nearest-neighbor graph with self-loops."""
    if n_channels == len(STEW_CHANNEL_POSITIONS):
        positions = torch.tensor(STEW_CHANNEL_POSITIONS, dtype=torch.float32)
        distances = torch.cdist(positions, positions)
        nearest = distances.topk(k=min(neighbors + 1, n_channels), largest=False).indices
        adjacency = torch.zeros(n_channels, n_channels)
        adjacency.scatter_(1, nearest, 1.0)
        adjacency = torch.maximum(adjacency, adjacency.T)
    else:
        adjacency = torch.eye(n_channels)
        if n_channels > 1:
            indices = torch.arange(n_channels - 1)
            adjacency[indices, indices + 1] = 1.0
            adjacency[indices + 1, indices] = 1.0
    adjacency.fill_diagonal_(1.0)
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inv_sqrt_degree = degree.rsqrt()
    return inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]


class ElectrodeGNN(nn.Module):
    """Temporal encoder plus two GCN layers over electrode spatial proximity."""
    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int = 2,
                 hidden_size: int = 64, dropout: float = 0.3):
        super().__init__()
        self.register_buffer(
            "normalized_adjacency",
            _normalized_electrode_adjacency(n_channels),
        )
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, hidden_size, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.graph1 = nn.Linear(hidden_size, hidden_size)
        self.graph2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        batch, timepoints, channels = x.shape
        nodes = x.permute(0, 2, 1).reshape(batch * channels, 1, timepoints)
        nodes = self.temporal_encoder(nodes).squeeze(-1)
        nodes = nodes.reshape(batch, channels, -1)
        nodes = torch.einsum("ij,bjf->bif", self.normalized_adjacency, nodes)
        nodes = F.relu(self.graph1(nodes))
        nodes = self.dropout(nodes)
        nodes = torch.einsum("ij,bjf->bif", self.normalized_adjacency, nodes)
        nodes = F.relu(self.graph2(nodes)).mean(dim=1)
        return self.classifier(nodes)


# ---------------------------------------------------------------------------
# STFT transformer baselines
# ---------------------------------------------------------------------------
class STFTSpectrogram(nn.Module):
    """Deterministic log-magnitude STFT shared by real and synthetic windows."""
    def __init__(self, n_fft: int = 64, hop_length: int = 16,
                 sampling_rate_hz: int = 128, max_frequency_hz: float = 45.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_bin = min(
            n_fft // 2 + 1,
            int(max_frequency_hz * n_fft / sampling_rate_hz) + 1,
        )
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, x):
        batch, timepoints, channels = x.shape
        flattened = x.permute(0, 2, 1).reshape(batch * channels, timepoints)
        spectrum = torch.stft(
            flattened,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        spectrum = torch.log1p(spectrum.abs())[:, :self.max_bin]
        spectrum = spectrum.reshape(batch, channels, spectrum.shape[-2], spectrum.shape[-1])
        mean = spectrum.mean(dim=(-2, -1), keepdim=True)
        std = spectrum.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (spectrum - mean) / std


class STFTVisionTransformer(nn.Module):
    """Compact ViT adaptation over multi-channel EEG spectrogram patches."""
    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int = 2,
                 embed_dim: int = 96, depth: int = 3, num_heads: int = 4,
                 patch_size: tuple[int, int] = (4, 4), dropout: float = 0.2):
        super().__init__()
        self.stft = STFTSpectrogram()
        self.patch_embed = nn.Conv2d(
            n_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        with torch.no_grad():
            spectrogram = self.stft(torch.zeros(1, n_timepoints, n_channels))
            patch_grid = self.patch_embed(spectrogram)
        n_patches = int(patch_grid.shape[-2] * patch_grid.shape[-1])
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        patches = self.patch_embed(self.stft(x)).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([class_token, patches], dim=1)
        tokens = self.encoder(tokens + self.position_embedding)
        return self.classifier(self.norm(tokens[:, 0]))


class ShiftedWindowBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, window_size: int,
                 shift_size: int, dropout: float):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        batch, height, width, channels = x.shape
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        padded_h, padded_w = x.shape[1:3]
        windows = (
            x.reshape(
                batch,
                padded_h // self.window_size,
                self.window_size,
                padded_w // self.window_size,
                self.window_size,
                channels,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, self.window_size * self.window_size, channels)
        )
        attention_mask = None
        if self.shift_size:
            region_mask = x.new_zeros((1, padded_h, padded_w, 1))
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            region_id = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    region_mask[:, height_slice, width_slice, :] = region_id
                    region_id += 1
            mask_windows = (
                region_mask.reshape(
                    1,
                    padded_h // self.window_size,
                    self.window_size,
                    padded_w // self.window_size,
                    self.window_size,
                    1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(-1, self.window_size * self.window_size)
            )
            attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attention_mask = attention_mask.masked_fill(
                attention_mask != 0, -100.0
            ).masked_fill(attention_mask == 0, 0.0)
            attention_mask = attention_mask.repeat(batch, 1, 1)
            attention_mask = attention_mask.repeat_interleave(
                self.num_heads, dim=0
            )
        windows, _ = self.attention(
            windows,
            windows,
            windows,
            attn_mask=attention_mask,
            need_weights=False,
        )
        x = (
            windows.reshape(
                batch,
                padded_h // self.window_size,
                padded_w // self.window_size,
                self.window_size,
                self.window_size,
                channels,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, padded_h, padded_w, channels)
        )
        x = x[:, :height, :width]
        if self.shift_size:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = residual + x
        return x + self.mlp(self.norm2(x))


class STFTSwinTransformer(nn.Module):
    """Compact shifted-window transformer adaptation for EEG spectrograms."""
    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int = 2,
                 embed_dim: int = 96, depth: int = 4, num_heads: int = 4,
                 patch_size: tuple[int, int] = (2, 4), window_size: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.stft = STFTSpectrogram()
        self.patch_embed = nn.Conv2d(
            n_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.blocks = nn.ModuleList([
            ShiftedWindowBlock(
                embed_dim,
                num_heads,
                window_size,
                shift_size=0 if index % 2 == 0 else window_size // 2,
                dropout=dropout,
            )
            for index in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.patch_embed(self.stft(x)).permute(0, 2, 3, 1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=(1, 2))
        return self.classifier(x)


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
    "rnn": VanillaRNN,
    "lstm": VanillaLSTM,
    "bilstm": BidirectionalLSTM,
    "gru": VanillaGRU,
    "gnn": ElectrodeGNN,
    "vit": STFTVisionTransformer,
    "swin": STFTSwinTransformer,
    "eegnet_adapted": EEGNetAdapted,
    "deepconvnet_adapted": DeepConvNetAdapted,
    "shallowconvnet_adapted": ShallowConvNetAdapted,
    "temporalcnn": TemporalCNN,
}

ACTIVE_MODEL_NAMES = (
    "1dcnn",
    "rnn",
    "lstm",
    "bilstm",
    "gru",
    "gnn",
    "vit",
    "swin",
)

OPTIONAL_MODEL_NAMES = (
    "eegnet_adapted",
    "deepconvnet_adapted",
    "shallowconvnet_adapted",
    "temporalcnn",
)

MODEL_INPUT_REPRESENTATIONS = {
    "1dcnn": "raw_eeg",
    "rnn": "raw_eeg_sequence",
    "lstm": "raw_eeg_sequence",
    "bilstm": "raw_eeg_sequence",
    "gru": "raw_eeg_sequence",
    "gnn": "raw_eeg_temporal_features_plus_electrode_knn_graph",
    "vit": "internal_log_magnitude_stft",
    "swin": "internal_log_magnitude_stft",
    "eegnet_adapted": "raw_eeg",
    "deepconvnet_adapted": "raw_eeg",
    "shallowconvnet_adapted": "raw_eeg",
    "temporalcnn": "raw_eeg",
}

MODEL_ALIASES = {
    "cnn": "1dcnn",
    "gcn": "gnn",
    "vision": "vit",
    "vision_transformer": "vit",
    "swin_transformer": "swin",
    "eegnet": "eegnet_adapted",
    "deepconvnet": "deepconvnet_adapted",
    "shallowconvnet": "shallowconvnet_adapted",
}


def canonical_model_name(name: str) -> str:
    name = name.lower()
    return MODEL_ALIASES.get(name, name)


def model_metadata(name: str) -> dict:
    canonical_name = canonical_model_name(name)
    if canonical_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'.")
    metadata = {
        "canonical_name": canonical_name,
        "input_representation": MODEL_INPUT_REPRESENTATIONS[canonical_name],
        "active_by_default": canonical_name in ACTIVE_MODEL_NAMES,
    }
    if canonical_name in ("vit", "swin"):
        metadata["stft"] = {
            "n_fft": 64,
            "hop_length": 16,
            "window": "hann",
            "magnitude": "log1p_absolute",
            "frequency_range_hz": [0.0, 45.0],
        }
        metadata["architecture_status"] = "EEG adaptation, not an exact image-model reproduction"
    if canonical_name == "gnn":
        metadata["graph"] = {
            "nodes": "14 STEW Emotiv electrodes",
            "coordinate_source": "MNE standard_1020 montage",
            "edges": "symmetric 3-nearest-neighbor graph from standard 3D electrode positions",
            "self_loops": True,
            "normalization": "symmetric_degree",
        }
    return metadata


def get_model(name: str, n_channels: int, n_timepoints: int, n_classes: int = 2, **kwargs):
    name = canonical_model_name(name)
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
