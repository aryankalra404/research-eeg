"""Classification metrics at window, recording and subject level.

Units of analysis
-----------------
* window: every 4 s window (overlapping windows are NOT independent, so
  confidence intervals resample whole subjects, never windows);
* recording: one STEW recording = one subject x condition (96 units),
  predicted by averaging its window probabilities;
* subject: per-subject accuracy / balanced accuracy over that subject's
  windows (48 paired values per model -> Friedman / Wilcoxon statistics).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

HEADLINE_METRICS = (
    "accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro",
    "sensitivity", "specificity", "cohen_kappa", "mcc", "roc_auc", "pr_auc",
    "brier", "log_loss", "ece",
)
HIGHER_IS_BETTER = {m: m not in {"brier", "log_loss", "ece"} for m in HEADLINE_METRICS}
BOOTSTRAP_METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa", "mcc", "roc_auc")


def expected_calibration_error(y_true: np.ndarray, p1: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins (Guo et al., ICML 2017)."""
    confidence = np.maximum(p1, 1 - p1)
    correct = ((p1 >= 0.5).astype(int) == y_true).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0.5 else (confidence >= lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def reliability_curve(y_true: np.ndarray, p1: np.ndarray, n_bins: int = 10) -> dict:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p1, edges[1:-1]), 0, n_bins - 1)
    mean_pred, frac_pos, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        counts.append(int(mask.sum()))
        mean_pred.append(float(p1[mask].mean()) if mask.any() else None)
        frac_pos.append(float(y_true[mask].mean()) if mask.any() else None)
    return {"mean_predicted": mean_pred, "fraction_positive": frac_pos, "count": counts}


def binary_metrics(y_true: np.ndarray, p1: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p1 = np.clip(np.asarray(p1, dtype=float), 1e-7, 1 - 1e-7)
    y_pred = (p1 >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if both else None,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,  # high-workload recall
        "specificity": float(tn / (tn + fp)) if tn + fp else None,  # rest recall
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)) if both else None,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, p1)) if both else None,
        "pr_auc": float(average_precision_score(y_true, p1)) if both else None,
        "brier": float(brier_score_loss(y_true, p1)),
        "log_loss": float(log_loss(y_true, p1, labels=[0, 1])),
        "ece": expected_calibration_error(y_true, p1),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def curves(y_true: np.ndarray, p1: np.ndarray, n_points: int = 101) -> dict:
    grid = np.linspace(0, 1, n_points)
    fpr, tpr, _ = roc_curve(y_true, p1)
    precision, recall, _ = precision_recall_curve(y_true, p1)
    return {
        "grid": grid.tolist(),
        "roc_tpr": np.interp(grid, fpr, tpr).tolist(),
        "pr_precision": np.interp(grid, recall[::-1], precision[::-1]).tolist(),
        "reliability": reliability_curve(y_true, p1),
    }


def recording_level(y_true: np.ndarray, p1: np.ndarray, subject: np.ndarray):
    keys = sorted({(int(s), int(c)) for s, c in zip(subject, y_true)})
    y_rec = np.array([c for _, c in keys])
    p_rec = np.array([p1[(subject == s) & (y_true == c)].mean() for s, c in keys])
    s_rec = np.array([s for s, _ in keys])
    return y_rec, p_rec, s_rec


def per_subject(y_true: np.ndarray, p1: np.ndarray, subject: np.ndarray) -> dict[int, dict]:
    out = {}
    for sid in np.unique(subject):
        mask = subject == sid
        yt, pp = y_true[mask], p1[mask]
        pred = (pp >= 0.5).astype(int)
        recalls = [np.mean(pred[yt == c] == c) for c in np.unique(yt)]
        out[int(sid)] = {
            "n_windows": int(mask.sum()),
            "accuracy": float(np.mean(pred == yt)),
            "balanced_accuracy": float(np.mean(recalls)),
            "roc_auc": float(roc_auc_score(yt, pp)) if len(np.unique(yt)) == 2 else None,
        }
    return out


def _weighted_auc(y: np.ndarray, tie_group: np.ndarray, n_groups: int, weights: np.ndarray) -> float:
    """Weighted Mann-Whitney AUC; ``tie_group`` indexes the sorted unique
    scores, so ties count one half."""
    pos = np.bincount(tie_group, weights * (y == 1), minlength=n_groups)
    neg = np.bincount(tie_group, weights * (y == 0), minlength=n_groups)
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    neg_below = np.cumsum(neg) - neg
    return float((pos * (neg_below + 0.5 * neg)).sum() / (n_pos * n_neg))


def _metrics_from_counts(tp, fp, tn, fn) -> dict[str, np.ndarray]:
    n = tp + fp + tn + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        prec1, prec0 = tp / (tp + fp), tn / (tn + fn)
        f1_1 = np.nan_to_num(2 * prec1 * sens / (prec1 + sens))
        f1_0 = np.nan_to_num(2 * prec0 * spec / (prec0 + spec))
        acc = (tp + tn) / n
        pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
        kappa = (acc - pe) / (1 - pe)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"accuracy": acc, "balanced_accuracy": (sens + spec) / 2, "f1_macro": (f1_0 + f1_1) / 2,
            "cohen_kappa": kappa, "mcc": np.nan_to_num(mcc)}


def subject_bootstrap(y_true, p1, subject, seed: int = 0, n_boot: int = 2000,
                      metrics=BOOTSTRAP_METRICS) -> dict:
    """Percentile 95% CIs from resampling whole subjects with replacement
    (cluster bootstrap), so correlated windows are never treated as
    independent observations."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(p1) >= 0.5).astype(int)
    subjects, subject_index = np.unique(subject, return_inverse=True)
    counts = np.zeros((len(subjects), 4))
    np.add.at(counts, (subject_index, 0), (pred == 1) & (y_true == 1))
    np.add.at(counts, (subject_index, 1), (pred == 1) & (y_true == 0))
    np.add.at(counts, (subject_index, 2), (pred == 0) & (y_true == 0))
    np.add.at(counts, (subject_index, 3), (pred == 0) & (y_true == 1))
    multiplicity = rng.multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)), size=n_boot)
    boot_counts = multiplicity @ counts  # (n_boot, 4)
    values = _metrics_from_counts(*boot_counts.T)
    if "roc_auc" in metrics:
        unique_scores, tie_group = np.unique(np.asarray(p1, float), return_inverse=True)
        values["roc_auc"] = np.array([
            _weighted_auc(y_true, tie_group, len(unique_scores), m[subject_index].astype(float))
            for m in multiplicity
        ])
    out = {}
    for m in metrics:
        v = np.asarray(values[m], float)
        v = v[np.isfinite(v)]
        out[m] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
    return out


def evaluate(y_true: np.ndarray, p1: np.ndarray, subject: np.ndarray, seed: int = 0,
             n_boot: int = 2000) -> dict:
    y_true, p1, subject = np.asarray(y_true), np.asarray(p1), np.asarray(subject)
    window = binary_metrics(y_true, p1)
    window["ci95"] = subject_bootstrap(y_true, p1, subject, seed, n_boot) if n_boot else None
    y_rec, p_rec, s_rec = recording_level(y_true, p1, subject)
    recording = binary_metrics(y_rec, p_rec)
    recording["ci95"] = subject_bootstrap(y_rec, p_rec, s_rec, seed, n_boot) if n_boot else None
    return {
        "window": window,
        "recording": recording,
        "per_subject": per_subject(y_true, p1, subject),
        "curves": curves(y_true, p1) if len(np.unique(y_true)) == 2 else None,
    }
