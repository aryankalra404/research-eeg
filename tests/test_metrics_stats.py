import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, roc_auc_score

from stewbench.evaluation import stats
from stewbench.evaluation.metrics import binary_metrics, evaluate, expected_calibration_error, subject_bootstrap


def _data(seed=0):
    rng = np.random.default_rng(seed)
    subject = np.repeat(np.arange(1, 21), 40)
    y = np.tile(np.r_[np.zeros(20), np.ones(20)], 20).astype(int)
    p1 = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.2, len(y)), 0, 1)
    return y, p1, subject


def test_binary_metrics_match_sklearn():
    y, p1, _ = _data()
    m = binary_metrics(y, p1)
    pred = (p1 >= 0.5).astype(int)
    assert np.isclose(m["accuracy"], accuracy_score(y, pred))
    assert np.isclose(m["balanced_accuracy"], balanced_accuracy_score(y, pred))
    assert np.isclose(m["cohen_kappa"], cohen_kappa_score(y, pred))
    assert np.isclose(m["roc_auc"], roc_auc_score(y, p1))


def test_subject_bootstrap_ci_brackets_estimate():
    y, p1, subject = _data()
    point = binary_metrics(y, p1)
    ci = subject_bootstrap(y, p1, subject, n_boot=500)
    for metric in ("accuracy", "balanced_accuracy", "cohen_kappa", "roc_auc"):
        assert ci[metric][0] <= point[metric] <= ci[metric][1]


def test_ece_zero_for_perfectly_calibrated_extremes():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, y.astype(float)) < 1e-9


def test_recording_and_subject_levels():
    y, p1, subject = _data()
    result = evaluate(y, p1, subject, n_boot=0)
    assert result["recording"]["n"] == 40
    assert len(result["per_subject"]) == 20


def test_holm_is_monotone_and_bounded():
    adjusted = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(adjusted["a"], 0.03) and np.isclose(adjusted["c"], 0.06) and np.isclose(adjusted["b"], 0.06)


def test_friedman_detects_clear_ordering_and_cd_formula():
    rng = np.random.default_rng(0)
    scores = np.column_stack([rng.normal(0.8, 0.02, 30), rng.normal(0.6, 0.02, 30), rng.normal(0.5, 0.02, 30)])
    result = stats.friedman_nemenyi(scores, ["a", "b", "c"])
    assert result["friedman_p"] < 1e-6
    assert result["mean_ranks"]["a"] < result["mean_ranks"]["b"] < result["mean_ranks"]["c"]
    # Demsar (2006) Table 5: q_0.05 = 2.343 for k = 3
    assert abs(result["nemenyi_q_alpha"] - 2.343) < 0.01


def test_paired_comparison_sign_and_effect_size():
    base = np.linspace(0.5, 0.7, 20)
    better = base + 0.05
    result = stats.paired_comparison(better, base)
    assert result["mean_difference"] > 0 and result["rank_biserial"] == 1.0 and result["p_value"] < 0.001
    assert result["mean_difference_ci95"][0] > 0
