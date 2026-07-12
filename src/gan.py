"""
Conditional WGAN-GP for raw EEG window generation, shape (T=512, C=14),
conditioned on the binary stress label.

Design (see conversation for rationale):
    - Generator: noise + label embedding -> transposed 1D convs, upsample
      32 -> 64 -> 128 -> 256 -> 512 timepoints, channels down to 14.
    - Critic: mirror -- strided 1D convs downsample 512 -> 32, label embedding
      concatenated as extra channels at the input. No batchnorm in critic
      (breaks the Lipschitz assumption needed for the gradient penalty) --
      uses InstanceNorm instead.
    - Conditioning: label embedded to a vector, broadcast across time,
      concatenated as extra channel(s).

Usage:
    from src.gan import Generator, Critic, gradient_penalty
"""

import torch
import torch.nn as nn


LATENT_DIM = 100
LABEL_EMBED_DIM = 8  # small embedding for binary label, broadcast across time


def weights_init(m):
    """
    DCGAN-style initialization, carried through in the official WGAN-GP paper
    (Gulrajani et al. 2017) and virtually all reference implementations.
    GANs are known to be sensitive to init; PyTorch's default init is not
    what the literature actually uses here.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("GroupNorm") != -1 or classname.find("BatchNorm") != -1:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, n_channels: int = 14, n_timepoints: int = 512,
                 latent_dim: int = LATENT_DIM, n_classes: int = 2,
                 label_embed_dim: int = LABEL_EMBED_DIM):
        super().__init__()
        assert n_timepoints == 512, "Upsampling schedule hardcoded for T=512 (32*2^4)"
        self.label_embed = nn.Embedding(n_classes, label_embed_dim)
        self.init_len = 32
        self.init_channels = 128

        in_dim = latent_dim + label_embed_dim
        self.fc = nn.Linear(in_dim, self.init_channels * self.init_len)

        def up_block(in_c, out_c, final=False):
            layers = [
                nn.ConvTranspose1d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            ]
            if not final:
                layers += [nn.BatchNorm1d(out_c), nn.ReLU(inplace=True)]
            # NOTE: no final activation (e.g. Tanh) here -- real data is
            # z-scored (mean 0, std ~1) but NOT bounded to [-1,1] (real
            # windows commonly range well beyond +/-3). A bounded final
            # activation like Tanh structurally caps the generator's output
            # range, making it impossible to match the real distribution no
            # matter how long you train -- this was causing severe waveform
            # saturation and mode-collapse-like t-SNE separation. Linear
            # output is correct for unbounded, roughly-Gaussian z-scored data.
            return nn.Sequential(*layers)

        # 32 -> 64 -> 128 -> 256 -> 512  (4 upsampling steps)
        self.net = nn.Sequential(
            up_block(128, 64),
            up_block(64, 32),
            up_block(32, 16),
            up_block(16, n_channels, final=True),
        )

    def forward(self, z, labels):
        # z: (B, latent_dim), labels: (B,) long
        label_vec = self.label_embed(labels)  # (B, label_embed_dim)
        x = torch.cat([z, label_vec], dim=1)  # (B, latent_dim + embed_dim)
        x = self.fc(x)  # (B, init_channels * init_len)
        x = x.view(-1, self.init_channels, self.init_len)  # (B, 128, 32)
        x = self.net(x)  # (B, n_channels, 512)
        return x.permute(0, 2, 1)  # -> (B, 512, n_channels) matching real data layout


class Critic(nn.Module):
    def __init__(self, n_channels: int = 14, n_timepoints: int = 512,
                 n_classes: int = 2, label_embed_dim: int = LABEL_EMBED_DIM):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, label_embed_dim)
        self.n_timepoints = n_timepoints

        in_c = n_channels + label_embed_dim  # label broadcast as extra channels

        def down_block(in_c, out_c, norm=True):
            layers = [nn.Conv1d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            if norm:
                # GroupNorm(num_groups=1, ...) normalizes jointly across all
                # channels+timesteps per sample -- this is the actual "Layer
                # Normalization" the official WGAN-GP paper (Gulrajani et al.
                # 2017) specifies for the critic, NOT InstanceNorm (which
                # normalizes each channel independently and is a different,
                # weaker form of per-sample normalization). Both preserve the
                # per-sample independence the gradient penalty needs (unlike
                # BatchNorm), but GroupNorm(1, C) is the one that actually
                # matches the reference implementation.
                layers += [nn.GroupNorm(1, out_c)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            return nn.Sequential(*layers)

        # 512 -> 256 -> 128 -> 64 -> 32
        self.net = nn.Sequential(
            down_block(in_c, 16, norm=False),  # no norm on first layer (WGAN-GP convention)
            down_block(16, 32),
            down_block(32, 64),
            down_block(64, 128),
        )
        self.fc = nn.Linear(128 * (n_timepoints // 16), 1)

    def forward(self, x, labels):
        # x: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        label_vec = self.label_embed(labels)  # (B, embed_dim)
        label_map = label_vec.unsqueeze(2).expand(-1, -1, x.size(2))  # (B, embed_dim, T)
        x = torch.cat([x, label_map], dim=1)  # (B, C+embed_dim, T)
        x = self.net(x)  # (B, 128, T/16)
        x = x.view(x.size(0), -1)
        return self.fc(x)  # (B, 1) raw critic score, no sigmoid (WGAN)


def gradient_penalty(critic, real, fake, labels, device):
    """
    Standard WGAN-GP gradient penalty: interpolate between real and fake
    samples, penalize the critic's gradient norm deviating from 1.
    """
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=device).expand_as(real)
    interpolated = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)

    critic_interpolated = critic(interpolated, labels)

    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty


if __name__ == "__main__":
    # Shape + gradient-penalty sanity check
    device = torch.device("cpu")
    B = 8
    gen = Generator().to(device)
    crit = Critic().to(device)

    z = torch.randn(B, LATENT_DIM, device=device)
    labels = torch.randint(0, 2, (B,), device=device)

    fake = gen(z, labels)
    print(f"Generator output shape: {tuple(fake.shape)} (expect (8, 512, 14))")
    assert fake.shape == (B, 512, 14)

    score = crit(fake, labels)
    print(f"Critic output shape: {tuple(score.shape)} (expect (8, 1))")
    assert score.shape == (B, 1)

    real = torch.randn(B, 512, 14, device=device)
    gp = gradient_penalty(crit, real, fake.detach(), labels, device)
    print(f"Gradient penalty value: {gp.item():.4f} (should be a finite positive scalar)")
    assert torch.isfinite(gp)

    n_gen_params = sum(p.numel() for p in gen.parameters())
    n_crit_params = sum(p.numel() for p in crit.parameters())
    print(f"Generator params: {n_gen_params:,}")
    print(f"Critic params: {n_crit_params:,}")
    print("\nAll GAN shape/gradient sanity checks passed.")
