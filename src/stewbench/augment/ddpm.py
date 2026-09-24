"""Class-conditional denoising diffusion probabilistic model for EEG windows
(Ho et al., NeurIPS 2020) with a compact 1D U-Net, cosine noise schedule
(Nichol & Dhariwal, ICML 2021), classifier-free guidance (Ho & Salimans,
2022) and deterministic DDIM sampling (Song et al., ICLR 2021).

Included as the modern alternative to the GAN: diffusion models are now the
dominant generative approach for EEG synthesis, and reviewers will ask for
the comparison. Trained only on the inner-training subjects of a fold.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x))) + self.emb(emb)[:, :, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class UNet1D(nn.Module):
    def __init__(self, n_channels: int, n_classes: int = 2, base: int = 64, mults=(1, 2, 4), emb_dim: int = 256):
        super().__init__()
        self.n_classes = n_classes
        self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.label_emb = nn.Embedding(n_classes + 1, emb_dim)  # last index = unconditional
        self.base = base
        self.inp = nn.Conv1d(n_channels, base, 3, padding=1)
        widths = [base * m for m in mults]
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        cin = base
        for w in widths:
            self.down.append(nn.ModuleList([ResBlock(cin, w, emb_dim), ResBlock(w, w, emb_dim)]))
            self.pool.append(nn.Conv1d(w, w, 4, stride=2, padding=1))
            cin = w
        self.mid = nn.ModuleList([ResBlock(cin, cin, emb_dim), ResBlock(cin, cin, emb_dim)])
        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for w in reversed(widths):
            self.upsample.append(nn.ConvTranspose1d(cin, cin, 4, stride=2, padding=1))
            self.up.append(nn.ModuleList([ResBlock(cin + w, w, emb_dim), ResBlock(w, w, emb_dim)]))
            cin = w
        self.out = nn.Sequential(nn.GroupNorm(8, cin), nn.SiLU(), nn.Conv1d(cin, n_channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, labels):
        emb = self.time_mlp(_timestep_embedding(t, self.base)) + self.label_emb(labels)
        h = self.inp(x)
        skips = []
        for (b1, b2), pool in zip(self.down, self.pool):
            h = b2(b1(h, emb), emb)
            skips.append(h)
            h = pool(h)
        for block in self.mid:
            h = block(h, emb)
        for (b1, b2), upsample in zip(self.up, self.upsample):
            h = upsample(h)
            h = b2(b1(torch.cat([h, skips.pop()], 1), emb), emb)
        return self.out(h)


def cosine_alpha_bar(steps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, steps, steps + 1) / steps
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0)


class ConditionalDDPM:
    name = "ddpm"

    def __init__(self, n_channels: int, n_times: int, device, seed: int = 0, steps: int = 1000,
                 lr: float = 2e-4, ema_decay: float = 0.999, p_uncond: float = 0.1,
                 guidance: float = 1.0, sample_steps: int = 50):
        if n_times % 8:
            raise ValueError("DDPM U-Net needs n_times divisible by 8")
        torch.manual_seed(seed)
        self.device, self.seed = device, seed
        self.n_channels, self.n_times = n_channels, n_times
        self.net = UNet1D(n_channels).to(device)
        self.ema = copy.deepcopy(self.net).eval()
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=0.0)
        self.alpha_bar = cosine_alpha_bar(steps).to(device)
        self.steps, self.ema_decay, self.p_uncond = steps, ema_decay, p_uncond
        self.guidance, self.sample_steps = guidance, sample_steps
        self.history: dict = {"loss": []}

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int = 64, verbose: bool = False):
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        g = torch.Generator().manual_seed(self.seed)
        started = time.perf_counter()
        for epoch in range(epochs):
            losses = []
            order = torch.randperm(len(X_t), generator=g).to(self.device)
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                x0, labels = X_t[idx], y_t[idx].clone()
                drop = (torch.rand(len(idx), generator=g) < self.p_uncond).to(self.device)
                labels[drop] = self.net.n_classes
                t = torch.randint(0, self.steps, (len(idx),), generator=g).to(self.device)
                noise = torch.randn(x0.shape, generator=g).to(self.device)
                ab = self.alpha_bar[t][:, None, None]
                xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
                loss = F.mse_loss(self.net(xt, t, labels), noise)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                with torch.no_grad():
                    for pe, p in zip(self.ema.parameters(), self.net.parameters()):
                        pe.lerp_(p, 1 - self.ema_decay)
                losses.append(loss.item())
            self.history["loss"].append(float(np.mean(losses)))
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"      ddpm epoch {epoch + 1}/{epochs} loss={self.history['loss'][-1]:.4f}")
        self.history["train_seconds"] = time.perf_counter() - started
        return self

    @torch.no_grad()
    def _eps(self, x, t, labels):
        t_batch = torch.full((len(x),), t, device=self.device, dtype=torch.long)
        eps_c = self.ema(x, t_batch, labels)
        if self.guidance == 1.0:
            return eps_c
        eps_u = self.ema(x, t_batch, torch.full_like(labels, self.ema.n_classes))
        return eps_u + self.guidance * (eps_c - eps_u)

    @torch.no_grad()
    def sample(self, n_per_class: dict[int, int], seed: int = 0, batch: int = 256):
        g = torch.Generator().manual_seed(seed)
        timesteps = torch.linspace(self.steps - 1, 0, self.sample_steps).long().tolist()
        X, y = [], []
        for label, n in n_per_class.items():
            for start in range(0, n, batch):
                m = min(batch, n - start)
                x = torch.randn(m, self.n_channels, self.n_times, generator=g).to(self.device)
                labels = torch.full((m,), label, dtype=torch.long, device=self.device)
                for i, t in enumerate(timesteps):  # DDIM, eta = 0
                    ab = self.alpha_bar[t]
                    ab_prev = self.alpha_bar[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=self.device)
                    eps = self._eps(x, t, labels)
                    x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-10, 10)
                    x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
                X.append(x.float().cpu().numpy())
                y.append(np.full(m, label, dtype=np.int64))
        return np.concatenate(X), np.concatenate(y)
