"""Window and subject-condition evaluation for binary EEG classifiers."""

from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "sensitivity_class1": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity_class0": float(tn / (tn + fp)) if tn + fp else 0.0,
        "confusion_matrix": matrix.tolist(),
    }


def aggregate_subject_conditions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average predictions within each subject and true experimental condition."""
    aggregated_true, aggregated_pred, aggregated_groups = [], [], []
    for subject_id in np.unique(groups):
        for class_id in np.unique(y_true[groups == subject_id]):
            mask = (groups == subject_id) & (y_true == class_id)
            mean_probability = probabilities[mask].mean(axis=0)
            aggregated_true.append(int(class_id))
            aggregated_pred.append(int(mean_probability.argmax()))
            aggregated_groups.append(int(subject_id))
    return (
        np.asarray(aggregated_true, dtype=np.int64),
        np.asarray(aggregated_pred, dtype=np.int64),
        np.asarray(aggregated_groups, dtype=np.int64),
    )


def clustered_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_bootstrap: int = 1000,
) -> dict:
    """Bootstrap whole subjects so overlapping windows are never resampled independently."""
    rng = np.random.default_rng(seed)
    subject_ids = np.unique(groups)
    accuracy_values, f1_values = [], []
    for _ in range(n_bootstrap):
        sampled_subjects = rng.choice(subject_ids, size=len(subject_ids), replace=True)
        true_parts, pred_parts = [], []
        for subject_id in sampled_subjects:
            mask = groups == subject_id
            true_parts.append(y_true[mask])
            pred_parts.append(y_pred[mask])
        sampled_true = np.concatenate(true_parts)
        sampled_pred = np.concatenate(pred_parts)
        accuracy_values.append(accuracy_score(sampled_true, sampled_pred))
        f1_values.append(f1_score(sampled_true, sampled_pred, average="macro", zero_division=0))
    return {
        "n_bootstrap": n_bootstrap,
        "accuracy_95ci": [float(v) for v in np.percentile(accuracy_values, [2.5, 97.5])],
        "f1_macro_95ci": [float(v) for v in np.percentile(f1_values, [2.5, 97.5])],
    }


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> dict:
    y_pred = probabilities.argmax(axis=1)
    window = binary_metrics(y_true, y_pred)
    window["clustered_subject_bootstrap"] = clustered_bootstrap_ci(
        y_true, y_pred, groups, seed=seed
    )

    condition_true, condition_pred, condition_groups = aggregate_subject_conditions(
        y_true, probabilities, groups
    )
    subject_condition = binary_metrics(condition_true, condition_pred)
    subject_condition["n_subject_condition_units"] = int(len(condition_true))
    subject_condition["clustered_subject_bootstrap"] = clustered_bootstrap_ci(
        condition_true, condition_pred, condition_groups, seed=seed
    )
    return {"window_level": window, "subject_condition_level": subject_condition}


def paired_sign_flip_test(without_gan: list[float], with_gan: list[float]) -> dict:
    """Exact paired two-sided sign-flip test across matched folds."""
    differences = np.asarray(with_gan) - np.asarray(without_gan)
    observed = abs(float(differences.mean()))
    null_values = [
        abs(float(np.mean(differences * np.asarray(signs))))
        for signs in product((-1, 1), repeat=len(differences))
    ]
    p_value = sum(value >= observed - 1e-12 for value in null_values) / len(null_values)
    return {
        "mean_paired_delta": float(differences.mean()),
        "fold_deltas": differences.tolist(),
        "exact_two_sided_p_value": float(p_value),
        "n_pairs": int(len(differences)),
    }
