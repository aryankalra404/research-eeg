"""Window and subject-condition evaluation for binary EEG classifiers."""

from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _safe_probability_metric(metric, y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if probabilities is None or len(np.unique(y_true)) < 2:
        return None
    return float(metric(y_true, probabilities[:, 1]))


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": _safe_probability_metric(roc_auc_score, y_true, probabilities),
        "average_precision": _safe_probability_metric(
            average_precision_score, y_true, probabilities
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "sensitivity_class1": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity_class0": float(tn / (tn + fp)) if tn + fp else 0.0,
        "per_class": {
            str(class_id): {
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
            }
            for class_id in (0, 1)
        },
        "confusion_matrix": matrix.tolist(),
    }
    if probabilities is not None and len(np.unique(y_true)) == 2:
        positive_probability = probabilities[:, 1]
        fpr, tpr, _ = roc_curve(y_true, positive_probability)
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_true, positive_probability
        )
        grid = np.linspace(0.0, 1.0, 101)
        metrics["curves"] = {
            "grid": grid.tolist(),
            "roc_tpr": np.interp(grid, fpr, tpr).tolist(),
            "pr_precision": np.interp(
                grid, recall_curve[::-1], precision_curve[::-1]
            ).tolist(),
        }
    else:
        metrics["curves"] = None
    return metrics


def aggregate_subject_conditions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average predictions within each subject and true experimental condition."""
    aggregated_true, aggregated_pred, aggregated_groups, aggregated_probabilities = [], [], [], []
    for subject_id in np.unique(groups):
        for class_id in np.unique(y_true[groups == subject_id]):
            mask = (groups == subject_id) & (y_true == class_id)
            mean_probability = probabilities[mask].mean(axis=0)
            aggregated_true.append(int(class_id))
            aggregated_pred.append(int(mean_probability.argmax()))
            aggregated_groups.append(int(subject_id))
            aggregated_probabilities.append(mean_probability)
    return (
        np.asarray(aggregated_true, dtype=np.int64),
        np.asarray(aggregated_pred, dtype=np.int64),
        np.asarray(aggregated_groups, dtype=np.int64),
        np.asarray(aggregated_probabilities, dtype=np.float64),
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
    accuracy_values, balanced_accuracy_values, f1_values, mcc_values = [], [], [], []
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
        balanced_accuracy_values.append(balanced_accuracy_score(sampled_true, sampled_pred))
        f1_values.append(f1_score(sampled_true, sampled_pred, average="macro", zero_division=0))
        mcc_values.append(matthews_corrcoef(sampled_true, sampled_pred))
    return {
        "n_bootstrap": n_bootstrap,
        "accuracy_95ci": [float(v) for v in np.percentile(accuracy_values, [2.5, 97.5])],
        "balanced_accuracy_95ci": [
            float(v) for v in np.percentile(balanced_accuracy_values, [2.5, 97.5])
        ],
        "f1_macro_95ci": [float(v) for v in np.percentile(f1_values, [2.5, 97.5])],
        "mcc_95ci": [float(v) for v in np.percentile(mcc_values, [2.5, 97.5])],
    }


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> dict:
    y_pred = probabilities.argmax(axis=1)
    window = binary_metrics(y_true, y_pred, probabilities)
    window["clustered_subject_bootstrap"] = clustered_bootstrap_ci(
        y_true, y_pred, groups, seed=seed
    )

    condition_true, condition_pred, condition_groups, condition_probabilities = aggregate_subject_conditions(
        y_true, probabilities, groups
    )
    subject_condition = binary_metrics(
        condition_true, condition_pred, condition_probabilities
    )
    subject_condition["n_subject_condition_units"] = int(len(condition_true))
    subject_condition["units"] = {
        "y_true": condition_true.tolist(),
        "y_pred": condition_pred.tolist(),
        "groups": condition_groups.tolist(),
        "probability_class1": condition_probabilities[:, 1].tolist(),
    }
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
        "minimum_attainable_two_sided_p": float(2 / (2 ** len(differences))),
    }


def holm_adjust_p_values(p_values: dict[str, float]) -> dict[str, float]:
    """Holm family-wise error correction, returned under the original keys."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    n_tests = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (n_tests - rank) * float(p_value))
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return {name: adjusted[name] for name in p_values}


def paired_subject_cluster_test(
    baseline_folds: list[dict],
    augmented_folds: list[dict],
    *,
    metric: str,
    seed: int,
    n_resamples: int = 5000,
) -> dict:
    """Paired bootstrap CI and randomization test over held-out subjects."""
    if metric not in {"accuracy", "f1_macro", "balanced_accuracy", "mcc"}:
        raise ValueError(f"Unsupported clustered paired metric: {metric}")

    def collect(folds):
        units = [fold["subject_condition_level"]["units"] for fold in folds]
        records = []
        for unit in units:
            records.extend(zip(unit["groups"], unit["y_true"], unit["y_pred"]))
        return sorted(records, key=lambda record: (record[0], record[1]))

    baseline = collect(baseline_folds)
    augmented = collect(augmented_folds)
    baseline_keys = [(group, truth) for group, truth, _ in baseline]
    augmented_keys = [(group, truth) for group, truth, _ in augmented]
    if baseline_keys != augmented_keys:
        raise ValueError("Paired subject-condition units are not aligned between conditions.")

    groups = np.asarray([record[0] for record in baseline])
    y_true = np.asarray([record[1] for record in baseline])
    baseline_pred = np.asarray([record[2] for record in baseline])
    augmented_pred = np.asarray([record[2] for record in augmented])

    def score(truth, prediction):
        if metric == "accuracy":
            return float(accuracy_score(truth, prediction))
        if metric == "f1_macro":
            return float(f1_score(truth, prediction, average="macro", zero_division=0))
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(truth, prediction))
        return float(matthews_corrcoef(truth, prediction))

    observed = score(y_true, augmented_pred) - score(y_true, baseline_pred)
    rng = np.random.default_rng(seed)
    subjects = np.unique(groups)
    bootstrap_deltas, null_deltas = [], []
    for _ in range(n_resamples):
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        sampled_indices = np.concatenate([
            np.flatnonzero(groups == subject) for subject in sampled_subjects
        ])
        bootstrap_deltas.append(
            score(y_true[sampled_indices], augmented_pred[sampled_indices])
            - score(y_true[sampled_indices], baseline_pred[sampled_indices])
        )

        swap_subjects = subjects[rng.random(len(subjects)) < 0.5]
        swap_mask = np.isin(groups, swap_subjects)
        permuted_baseline = baseline_pred.copy()
        permuted_augmented = augmented_pred.copy()
        permuted_baseline[swap_mask] = augmented_pred[swap_mask]
        permuted_augmented[swap_mask] = baseline_pred[swap_mask]
        null_deltas.append(
            score(y_true, permuted_augmented) - score(y_true, permuted_baseline)
        )

    p_value = (1 + np.sum(np.abs(null_deltas) >= abs(observed) - 1e-12)) / (
        n_resamples + 1
    )
    return {
        "metric": metric,
        "paired_delta": float(observed),
        "subject_cluster_bootstrap_95ci": [
            float(value) for value in np.percentile(bootstrap_deltas, [2.5, 97.5])
        ],
        "subject_cluster_randomization_p_value": float(p_value),
        "n_subjects": int(len(subjects)),
        "n_subject_condition_units": int(len(y_true)),
        "n_resamples": int(n_resamples),
    }
