"""Attention-based EEG baselines: EEG Conformer, ATCNet and compact
spectrogram ViT / Swin adaptations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import constants as C
from .common import TCN, Conv2dMaxNorm, LinearMaxNorm, STFTSpectrogram, infer_output_size, scaled_kernel


# ---------------------------------------------------------------------------
# EEG Conformer (Song et al., IEEE TNSRE 2023)
# ---------------------------------------------------------------------------
class _PreNormResidual(nn.Module):
    def __init__(self, dim, fn, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.fn(self.norm(x)))


class _SelfAttention(nn.Module):
    def __init__(self, dim, heads, dropout, key_dim=None):
        super().__init__()
        key_dim = key_dim or dim // heads
        self.heads, self.key_dim = heads, key_dim
        self.qkv = nn.Linear(dim, 3 * heads * key_dim)
        self.out = nn.Linear(heads * key_dim, dim)
        self.dropout = dropout

    def forward(self, x):
        batch, tokens, _ = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.key_dim).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out(attended.transpose(1, 2).reshape(batch, tokens, -1))


class EEGConformer(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=2, sfreq=C.SFREQ,
                 emb=40, depth=6, heads=10, dropout=0.5):
        super().__init__()
        k, pool, stride = scaled_kernel(25, sfreq), scaled_kernel(75, sfreq), scaled_kernel(15, sfreq)
        self.patch = nn.Sequential(
            nn.Conv2d(1, 40, (1, k)), nn.Conv2d(40, 40, (n_channels, 1)), nn.BatchNorm2d(40), nn.ELU(),
            nn.AvgPool2d((1, pool), (1, stride)), nn.Dropout(dropout), nn.Conv2d(40, emb, 1),
        )
        self.encoder = nn.Sequential(*[
            nn.Sequential(
                _PreNormResidual(emb, _SelfAttention(emb, heads, dropout), dropout),
                _PreNormResidual(emb, nn.Sequential(nn.Linear(emb, 4 * emb), nn.GELU(),
                                                    nn.Dropout(dropout), nn.Linear(4 * emb, emb)), dropout),
            ) for _ in range(depth)
        ])
        size = infer_output_size(self, self._tokens, (1, n_channels, n_times))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(size, 256), nn.ELU(), nn.Dropout(0.5),
                                  nn.Linear(256, 32), nn.ELU(), nn.Dropout(0.3), nn.Linear(32, n_classes))

    def _tokens(self, x):
        tokens = self.patch(x.unsqueeze(1)).flatten(2).transpose(1, 2)  # (B, n, emb)
        return self.encoder(tokens)

    def forward(self, x):
        return self.head(self._tokens(x))


# ---------------------------------------------------------------------------
# ATCNet (Altaheri et al., IEEE Trans. Industrial Informatics 2023)
# ---------------------------------------------------------------------------
class ATCNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=2, sfreq=C.SFREQ,
                 F1=16, D=2, pool1=8, pool2=7, conv_dropout=0.3, n_windows=5,
                 heads=2, key_dim=8, attention_dropout=0.5, tcn_depth=2, tcn_kernel=4,
                 tcn_dropout=0.3):
        super().__init__()
        F2 = F1 * D
        ke = scaled_kernel(64, sfreq)
        self.conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, ke), padding="same", bias=False), nn.BatchNorm2d(F1),
            Conv2dMaxNorm(F1, F2, (n_channels, 1), groups=F1, bias=False, max_norm=1.0),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, pool1)), nn.Dropout(conv_dropout),
            nn.Conv2d(F2, F2, (1, 16), padding="same", bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, pool2)), nn.Dropout(conv_dropout),
        )
        steps = infer_output_size(self, lambda x: self.conv(x.unsqueeze(1)), (1, n_channels, n_times)) // F2
        self.n_windows = max(1, min(n_windows, steps))
        self.window_len = steps - self.n_windows + 1
        self.attention_norm = nn.ModuleList([nn.LayerNorm(F2) for _ in range(self.n_windows)])
        self.attention = nn.ModuleList([
            _SelfAttention(F2, heads, attention_dropout, key_dim=key_dim) for _ in range(self.n_windows)
        ])
        self.attention_drop = nn.Dropout(0.3)
        self.tcn = nn.ModuleList([TCN(F2, F2, tcn_kernel, tcn_depth, tcn_dropout) for _ in range(self.n_windows)])
        self.dense = nn.ModuleList([LinearMaxNorm(F2, n_classes, max_norm=0.25) for _ in range(self.n_windows)])

    def forward(self, x):
        sequence = self.conv(x.unsqueeze(1)).squeeze(2).transpose(1, 2)  # (B, T', F2)
        logits = []
        for i in range(self.n_windows):
            window = sequence[:, i:i + self.window_len]
            window = window + self.attention_drop(self.attention[i](self.attention_norm[i](window)))
            features = self.tcn[i](window.transpose(1, 2))[..., -1]
            logits.append(self.dense[i](features))
        return torch.stack(logits).mean(0)


# ---------------------------------------------------------------------------
# Spectrogram transformers (compact adaptations, not the original image models)
# ---------------------------------------------------------------------------
class ViTSTFT(nn.Module):
    """ViT (Dosovitskiy et al., 2021) over patches of the multichannel STFT."""

    def __init__(self, n_channels, n_times, n_classes=2, sfreq=C.SFREQ,
                 embed_dim=96, depth=3, heads=4, patch=(4, 4), dropout=0.2):
        super().__init__()
        self.stft = STFTSpectrogram(sfreq)
        self.patch_embed = nn.Conv2d(n_channels, embed_dim, patch, stride=patch)
        with torch.no_grad():
            grid = self.patch_embed(self.stft(torch.zeros(1, n_channels, n_times)))
        n_patches = grid.shape[-2] * grid.shape[-1]
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(embed_dim, heads, 4 * embed_dim, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        patches = self.patch_embed(self.stft(x)).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), patches], dim=1) + self.pos
        return self.classifier(self.norm(self.encoder(tokens)[:, 0]))


class _SwinBlock(nn.Module):
    def __init__(self, dim, heads, window, shift, dropout):
        super().__init__()
        self.window, self.shift, self.heads = window, shift, heads
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def _partition(self, x):
        b, h, w, c = x.shape
        s = self.window
        return x.reshape(b, h // s, s, w // s, s, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, s * s, c)

    def _merge(self, windows, b, h, w):
        s = self.window
        c = windows.shape[-1]
        return windows.reshape(b, h // s, w // s, s, s, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)

    def _mask(self, h, w, device):
        s, shift = self.window, self.shift
        region = torch.zeros(1, h, w, 1, device=device)
        slices = (slice(0, -s), slice(-s, -shift), slice(-shift, None))
        label = 0
        for hs in slices:
            for ws in slices:
                region[:, hs, ws, :] = label
                label += 1
        ids = self._partition(region).squeeze(-1)
        mask = ids.unsqueeze(1) - ids.unsqueeze(2)
        return mask.masked_fill(mask != 0, float("-inf")).masked_fill(mask == 0, 0.0)

    def forward(self, x):
        b, h0, w0, c = x.shape
        shortcut = x
        x = self.norm1(x)
        pad_h, pad_w = (-h0) % self.window, (-w0) % self.window
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        h, w = x.shape[1:3]
        mask = None
        if self.shift:
            x = torch.roll(x, (-self.shift, -self.shift), dims=(1, 2))
            mask = self._mask(h, w, x.device).repeat(b, 1, 1).repeat_interleave(self.heads, dim=0)
        windows = self._partition(x)
        windows, _ = self.attention(windows, windows, windows, attn_mask=mask, need_weights=False)
        x = self._merge(windows, b, h, w)
        if self.shift:
            x = torch.roll(x, (self.shift, self.shift), dims=(1, 2))
        x = shortcut + x[:, :h0, :w0]
        return x + self.mlp(self.norm2(x))


class SwinSTFT(nn.Module):
    """Shifted-window transformer (Liu et al., ICCV 2021) over the STFT."""

    def __init__(self, n_channels, n_times, n_classes=2, sfreq=C.SFREQ,
                 embed_dim=96, depth=4, heads=4, patch=(2, 4), window=4, dropout=0.2):
        super().__init__()
        self.stft = STFTSpectrogram(sfreq)
        self.patch_embed = nn.Conv2d(n_channels, embed_dim, patch, stride=patch)
        self.blocks = nn.ModuleList([
            _SwinBlock(embed_dim, heads, window, 0 if i % 2 == 0 else window // 2, dropout)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.patch_embed(self.stft(x)).permute(0, 2, 3, 1)
        for block in self.blocks:
            x = block(x)
        return self.classifier(self.norm(x).mean(dim=(1, 2)))
