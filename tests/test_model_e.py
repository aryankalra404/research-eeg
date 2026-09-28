import numpy as np
import torch

from stewbench import constants as C
from stewbench.artifacts import OCULAR_BAND, ArtifactCounterfactual, _band, fit_artifact_bank
from stewbench.experiments.robust import dev_fold
from stewbench.settings import ProtocolConfig
from stewbench.splits import make_folds


def _windows_with_blinks(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 14, 512))
    blink = np.zeros(14)
    for ch, w in (("AF3", 1.0), ("AF4", 1.0), ("F7", 0.6), ("F8", 0.6), ("F3", 0.5), ("F4", 0.5)):
        blink[C.CHANNELS.index(ch)] = w
    t = np.arange(512)
    for i in range(n):
        centre = rng.integers(60, 450)
        X[i] += 8 * np.outer(blink, np.exp(-0.5 * ((t - centre) / 15) ** 2))
    X = (X - X.mean(-1, keepdims=True)) / X.std(-1, keepdims=True)
    return X.astype(np.float32), blink / np.linalg.norm(blink)


def test_bank_recovers_frontal_blink_topography():
    X, blink = _windows_with_blinks()
    bank = fit_artifact_bank(X, n_ocular=2)
    similarity = max(abs(float(t @ blink)) for t in bank.ocular_topographies)
    assert similarity > 0.9
    assert bank.ocular_waveforms.shape[0] == 2 and bank.ocular_waveforms.shape[-1] == 512


def test_counterfactual_shape_normalisation_and_ocular_removal():
    X, blink = _windows_with_blinks()
    bank = fit_artifact_bank(X, n_ocular=2)
    x = torch.as_tensor(X[:16])
    remove_only = ArtifactCounterfactual(bank, p_add_ocular=0.0, p_remove_ocular=1.0, p_emg=0.0)
    out = remove_only(x, torch.Generator().manual_seed(0))
    assert out.shape == x.shape and torch.isfinite(out).all()
    np.testing.assert_allclose(out.mean(-1).numpy(), 0, atol=1e-4)
    np.testing.assert_allclose(out.std(-1).numpy(), 1, atol=1e-3)
    low_before = _band(X[:16].astype(np.float64), OCULAR_BAND, 128)
    low_after = _band(out.numpy().astype(np.float64), OCULAR_BAND, 128)
    before = np.abs(np.einsum("c,nct->nt", blink, low_before)).mean()
    after = np.abs(np.einsum("c,nct->nt", blink, low_after)).mean()
    assert after < 0.5 * before


def test_counterfactual_is_deterministic_given_generator():
    X, _ = _windows_with_blinks()
    cf = ArtifactCounterfactual(fit_artifact_bank(X))
    x = torch.as_tensor(X[:8])
    a = cf(x, torch.Generator().manual_seed(3))
    b = cf(x, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_dev_mode_never_touches_test_subjects():
    subjects = np.repeat(np.arange(1, 49), 5)
    for fold in make_folds(subjects, ProtocolConfig()):
        dev = dev_fold(fold, 0.15, 2024)
        used = set(dev.train_subjects) | set(dev.val_subjects) | set(dev.test_subjects)
        assert used.isdisjoint(fold.test_subjects)
        assert set(dev.test_subjects) == set(fold.val_subjects)
        assert set(dev.train_subjects).isdisjoint(dev.val_subjects)


def test_consistency_training_runs():
    from stewbench.settings import TrainingConfig
    from stewbench.training import train_deep
    X, _ = _windows_with_blinks(64)
    y = np.repeat([0, 1], 32)
    cf = ArtifactCounterfactual(fit_artifact_bank(X))
    result = train_deep("eegnet", X[:48], y[:48], X[48:], y[48:],
                        TrainingConfig(max_epochs=2, patience=2, batch_size=16), 0, torch.device("cpu"),
                        counterfactual=cf, consistency_weight=1.0)
    assert result.epochs_run == 2
