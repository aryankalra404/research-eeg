import numpy as np
import pytest
import torch

from stewbench.models import ALL_MODELS, CLASSICAL_MODELS, DEEP_MODELS, SPECS, build_classical, build_deep


@pytest.mark.parametrize("name", DEEP_MODELS)
@pytest.mark.parametrize("n_times", [512, 256])
def test_deep_models_forward_backward(name, n_times):
    torch.manual_seed(0)
    model = build_deep(name, n_times=n_times)
    x = torch.randn(3, 14, n_times)
    logits = model(x)
    assert logits.shape == (3, 2)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_registry_is_consistent():
    assert set(ALL_MODELS) == set(SPECS)
    assert len(DEEP_MODELS) == 17 and len(CLASSICAL_MODELS) == 8


def test_recurrent_directionality():
    assert not build_deep("lstm").rnn.bidirectional
    assert build_deep("bilstm").rnn.bidirectional


def test_tsception_channel_order_pairs_hemispheres():
    from stewbench import constants as C
    model = build_deep("tsception")
    ordered = [C.CHANNELS[i] for i in model.order.tolist()]
    assert ordered[:7] == list(C.LEFT_HEMISPHERE) and ordered[7:] == list(C.RIGHT_HEMISPHERE)


@pytest.mark.parametrize("name", CLASSICAL_MODELS)
def test_classical_models_fit_predict(name, windows):
    train = np.isin(windows.subject, [1, 2, 3, 4, 5])
    val = np.isin(windows.subject, [6])
    test = np.isin(windows.subject, [7, 8])
    model = build_classical(name, seed=0)
    model.fit(windows.X[train], windows.y[train], windows.subject[train],
              windows.X[val], windows.y[val], windows.subject[val])
    proba = model.predict_proba(windows.X[test], windows.subject[test])
    assert proba.shape == (test.sum(), 2)
    np.testing.assert_allclose(proba.sum(1), 1, atol=1e-6)
