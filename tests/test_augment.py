import numpy as np
import pytest
import torch
from scipy.signal import welch

from stewbench.augment.transforms import ONLINE_AUGMENTATIONS, make_online


@pytest.mark.parametrize("name", ONLINE_AUGMENTATIONS)
def test_online_augmentations_shape_and_finiteness(name):
    x = torch.randn(8, 14, 512)
    y = torch.tensor([0, 1] * 4)
    out, target = make_online(name, p=1.0, magnitude=0.5)(x, y, torch.Generator().manual_seed(0))
    assert out.shape == x.shape and torch.isfinite(out).all()
    if name == "mixup":
        assert target.shape == (8, 2) and torch.allclose(target.sum(1), torch.ones(8))
    else:
        assert torch.equal(target, y)


def test_probability_zero_is_identity():
    x = torch.randn(4, 14, 512)
    y = torch.tensor([0, 1, 0, 1])
    for name in ONLINE_AUGMENTATIONS:
        if name == "mixup":
            continue
        out, _ = make_online(name, p=0.0, magnitude=1.0)(x, y, torch.Generator().manual_seed(0))
        assert torch.equal(out, x), name


def test_ft_surrogate_preserves_power_spectrum():
    x = torch.randn(4, 14, 512)
    out, _ = make_online("ft_surrogate", p=1.0, magnitude=1.0)(x, torch.zeros(4, dtype=torch.long),
                                                               torch.Generator().manual_seed(0))
    np.testing.assert_allclose(torch.fft.rfft(out).abs().numpy(), torch.fft.rfft(x).abs().numpy(), rtol=1e-3, atol=1e-3)


def test_frequency_shift_moves_a_sinusoid():
    t = torch.arange(512) / 128
    x = torch.sin(2 * np.pi * 10 * t).repeat(1, 14, 1)
    from stewbench.augment.transforms import frequency_shift
    g = torch.Generator().manual_seed(3)
    out, _ = frequency_shift(x, torch.zeros(1, dtype=torch.long), g, p=1.0, magnitude=1.0)
    f, psd = welch(out[0, 0].numpy(), fs=128, nperseg=512)
    assert abs(f[psd.argmax()] - 10) > 0.2


@pytest.mark.parametrize("method", ["cwgan_gp", "ddpm"])
def test_generators_train_and_sample(method):
    from stewbench.experiments.augmentation import _generator
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 14, 512)).astype(np.float32)
    y = np.repeat([0, 1], 16)
    gen = _generator(method, 14, 512, torch.device("cpu"), seed=0)
    if method == "ddpm":
        gen.sample_steps = 3
    gen.fit(X, y, epochs=1, batch_size=16)
    Xs, ys = gen.sample({0: 3, 1: 2}, seed=0)
    assert Xs.shape == (5, 14, 512) and np.isfinite(Xs).all()
    assert ys.tolist() == [0, 0, 0, 1, 1]
