import numpy as np
import pytest

from stewbench import constants as C
from stewbench.data.inputs import euclidean_alignment, window_zscore
from stewbench.data.preprocess import artifact_mask, load_windows, sliding_windows
from stewbench.settings import PreprocessConfig


def test_windows_shape_labels_and_bookkeeping(windows):
    assert windows.X.shape[1:] == (C.N_CHANNELS, 512)
    assert set(np.unique(windows.y)) == {0, 1}
    assert len(windows.subjects) == 8
    for sid in windows.subjects:  # every subject keeps both conditions
        assert set(np.unique(windows.y[windows.subject == sid])) == {0, 1}


def test_sliding_windows_step_and_count():
    data = np.arange(2 * 1000, dtype=float).reshape(2, 1000)
    w, starts = sliding_windows(data, 256, 128)
    assert w.shape == (6, 2, 256)
    assert starts.tolist() == [0, 128, 256, 384, 512, 640]
    np.testing.assert_array_equal(w[1, 0], data[0, 128:384])


def test_artifact_rejection_is_label_free():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3, 64))
    X[7, 1] *= 200
    keep = artifact_mask(X, k=10)
    assert not keep[7] and keep.sum() == 99


def test_window_zscore_uses_only_the_window():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 14, 512)).astype(np.float32)
    before = window_zscore(X)[0]
    X[1:] *= 1000
    np.testing.assert_allclose(window_zscore(X)[0], before, atol=1e-5)
    np.testing.assert_allclose(before.mean(-1), 0, atol=1e-5)
    np.testing.assert_allclose(before.std(-1), 1, atol=1e-3)


def test_euclidean_alignment_whitens_each_subject():
    rng = np.random.default_rng(0)
    mixing = rng.normal(size=(14, 14))
    X = np.einsum("cd,ndt->nct", mixing, rng.normal(size=(40, 14, 256)))
    subject = np.repeat([1, 2], 20)
    aligned = euclidean_alignment(X, subject)
    for s in (1, 2):
        Xs = aligned[subject == s].astype(np.float64)
        cov = np.einsum("nct,ndt->cd", Xs, Xs) / (Xs.shape[0] * Xs.shape[2])
        np.testing.assert_allclose(cov, np.eye(14), atol=5e-3)  # ridge term shifts tiny eigenvalues


def test_cache_round_trip_and_config_guard(fixture_raw, tmp_path):
    config = PreprocessConfig()
    first = load_windows(config, raw_dir=fixture_raw, cache_dir=tmp_path, strict=False)
    second = load_windows(config, raw_dir=fixture_raw, cache_dir=tmp_path, strict=False)
    np.testing.assert_array_equal(first.X, second.X)
    other = PreprocessConfig(window_seconds=2.0)
    assert other.cache_key() != config.cache_key()
    assert load_windows(other, raw_dir=fixture_raw, cache_dir=tmp_path, strict=False).X.shape[-1] == 256


def test_strict_loader_rejects_incomplete_dataset(fixture_raw):
    from stewbench.data.raw import load_stew
    with pytest.raises(ValueError):
        load_stew(fixture_raw, strict=True)


def test_default_cache_key_unchanged_by_exclude_channels_field():
    import hashlib
    import json
    from dataclasses import asdict
    config = PreprocessConfig()
    legacy = {k: v for k, v in asdict(config).items() if k != "exclude_channels"}
    expected = hashlib.sha1(json.dumps(legacy, sort_keys=True).encode()).hexdigest()[:10]
    assert config.cache_key() == expected
    assert PreprocessConfig(exclude_channels=["F7"]).cache_key() != expected


def test_excluded_channels_carry_no_information(fixture_raw):
    from stewbench.data.preprocess import preprocess_recordings
    from stewbench.data.raw import load_stew
    config = PreprocessConfig(exclude_channels=["F7", "T8"])
    w = preprocess_recordings(load_stew(fixture_raw, strict=False), config)
    for channel in ("F7", "T8"):
        assert np.all(w.X[:, C.CHANNELS.index(channel)] == 0)
        assert np.all(window_zscore(w.X)[:, C.CHANNELS.index(channel)] == 0)
    assert np.any(w.X[:, C.CHANNELS.index("F8")] != 0)


def test_unknown_excluded_channel_is_rejected():
    from stewbench.settings import ExperimentConfig, validate
    config = ExperimentConfig(models=["bp_lr"])
    config.preprocess.exclude_channels = ["Cz"]
    with pytest.raises(ValueError):
        validate(config)
