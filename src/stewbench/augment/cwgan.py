"""Conditional Wasserstein GAN with gradient penalty for multichannel EEG
windows (Gulrajani et al., NeurIPS 2017; EEG adaptation after Hartmann et
al., 2018).

Generator: noise + class embedding -> linear -> 4 transposed-conv upsampling
stages (T/16 -> T). Critic: mirrored strided convolutions with the class
embedding broadcast as extra input channels and per-sample LayerNorm
(GroupNorm with one group) instead of BatchNorm, as required by the gradient
penalty. Samples are drawn from an exponential moving average of the
generator weights.

Trained only on the inner-training subjects of a fold.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn

LATENT_DIM = 100
EMBED_DIM = 8


class Generator(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int = 2, latent_dim: int = LATENT_DIM):
        super().__init__()
        if n_times % 16:
            raise ValueError("CWGAN-GP generator needs n_times divisible by 16")
        self.latent_dim = latent_dim
        self.embed = nn.Embedding(n_classes, EMBED_DIM)
        self.init_len = n_times // 16
        self.fc = nn.Linear(latent_dim + EMBED_DIM, 256 * self.init_len)

        def up(cin, cout, last=False):
            layers = [nn.ConvTranspose1d(cin, cout, 4, stride=2, padding=1)]
            if not last:
                layers += [nn.BatchNorm1d(cout), nn.LeakyReLU(0.2)]
            return layers
        # Linear output: inputs are z-scored and unbounded, so no tanh.
        self.net = nn.Sequential(*up(256, 128), *up(128, 64), *up(64, 32), *up(32, n_channels, last=True))

    def forward(self, z, labels):
        h = self.fc(torch.cat([z, self.embed(labels)], 1)).view(len(z), 256, self.init_len)
        return self.net(h)


class Critic(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int = 2):
        super().__init__()
        self.embed = nn.Embedding(n_classes, EMBED_DIM)

        def down(cin, cout, norm=True):
            layers = [nn.Conv1d(cin, cout, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.GroupNorm(1, cout))
            return layers + [nn.LeakyReLU(0.2)]
        self.net = nn.Sequential(*down(n_channels + EMBED_DIM, 32, norm=False), *down(32, 64),
                                 *down(64, 128), *down(128, 256))
        self.fc = nn.Linear(256 * (n_times // 16), 1)

    def forward(self, x, labels):
        label_map = self.embed(labels)[:, :, None].expand(-1, -1, x.shape[-1])
        return self.fc(self.net(torch.cat([x, label_map], 1)).flatten(1))


def gradient_penalty(critic, real, fake, labels):
    eps = torch.rand(len(real), 1, 1, device=real.device)
    mixed = (eps * real + (1 - eps) * fake).requires_grad_(True)
    grad = torch.autograd.grad(critic(mixed, labels).sum(), mixed, create_graph=True)[0]
    return ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def _init(module):
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(module.weight, 0.0, 0.02)


class CWGANGP:
    name = "cwgan_gp"

    def __init__(self, n_channels: int, n_times: int, device, seed: int = 0, n_critic: int = 5,
                 lambda_gp: float = 10.0, lr: float = 1e-4, betas=(0.0, 0.9), ema_decay: float = 0.999):
        torch.manual_seed(seed)
        self.device = device
        self.generator = Generator(n_channels, n_times).to(device)
        self.critic = Critic(n_channels, n_times).to(device)
        self.generator.apply(_init)
        self.critic.apply(_init)
        self.ema = copy.deepcopy(self.generator).eval()
        self.n_critic, self.lambda_gp, self.ema_decay = n_critic, lambda_gp, ema_decay
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=betas)
        self.seed = seed
        self.history: dict = {"critic_loss": [], "generator_loss": [], "wasserstein": []}

    @torch.no_grad()
    def _update_ema(self):
        for ema_p, p in zip(self.ema.parameters(), self.generator.parameters()):
            ema_p.lerp_(p, 1 - self.ema_decay)
        for ema_b, b in zip(self.ema.buffers(), self.generator.buffers()):
            ema_b.copy_(b)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int = 64, verbose: bool = False):
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        g = torch.Generator().manual_seed(self.seed)
        steps = max(1, len(X_t) // batch_size)
        started = time.perf_counter()

        def batch():
            idx = torch.randint(0, len(X_t), (batch_size,), generator=g).to(self.device)
            return X_t[idx], y_t[idx]

        for epoch in range(epochs):
            c_losses, g_losses, w_est = [], [], []
            for _ in range(steps):
                for _ in range(self.n_critic):
                    real, labels = batch()
                    z = torch.randn(len(real), self.generator.latent_dim, device=self.device)
                    fake = self.generator(z, labels).detach()
                    c_real, c_fake = self.critic(real, labels).mean(), self.critic(fake, labels).mean()
                    loss_c = c_fake - c_real + self.lambda_gp * gradient_penalty(self.critic, real, fake, labels)
                    self.opt_c.zero_grad(set_to_none=True)
                    loss_c.backward()
                    self.opt_c.step()
                    c_losses.append(loss_c.item())
                    w_est.append((c_real - c_fake).item())
                labels = torch.randint(0, 2, (batch_size,), generator=g).to(self.device)
                z = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
                loss_g = -self.critic(self.generator(z, labels), labels).mean()
                self.opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                self.opt_g.step()
                self._update_ema()
                g_losses.append(loss_g.item())
            self.history["critic_loss"].append(float(np.mean(c_losses)))
            self.history["generator_loss"].append(float(np.mean(g_losses)))
            self.history["wasserstein"].append(float(np.mean(w_est)))
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"      cwgan epoch {epoch + 1}/{epochs} W={self.history['wasserstein'][-1]:.3f}")
        self.history["train_seconds"] = time.perf_counter() - started
        return self

    @torch.no_grad()
    def sample(self, n_per_class: dict[int, int], seed: int = 0, batch: int = 512):
        g = torch.Generator().manual_seed(seed)
        X, y = [], []
        for label, n in n_per_class.items():
            for start in range(0, n, batch):
                m = min(batch, n - start)
                z = torch.randn(m, self.ema.latent_dim, generator=g).to(self.device)
                labels = torch.full((m,), label, dtype=torch.long, device=self.device)
                X.append(self.ema(z, labels).float().cpu().numpy())
                y.append(np.full(m, label, dtype=np.int64))
        return np.concatenate(X), np.concatenate(y)
