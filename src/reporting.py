"""Publication-oriented summaries for classifier and augmentation experiments."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from scipy.stats import t


REPORT_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "sensitivity_class1",
    "specificity_class0",
    "f1_macro",
    "mcc",
    "roc_auc",
    "average_precision",
)


def _finite_values(values: list[float | None]) -> np.ndarray:
    return np.asarray([value for value in values if value is not None], dtype=float)


def summarize_values(values: list[float | None]) -> dict:
    """Summarize matched folds; CI is explicitly a fold-mean t interval."""
    finite = _finite_values(values)
    if len(finite) == 0:
        return {"mean": None, "std": None, "fold_mean_95ci": None, "n_folds": 0}
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
    if len(finite) > 1:
        half_width = float(t.ppf(0.975, len(finite) - 1) * std / np.sqrt(len(finite)))
        confidence_interval = [mean - half_width, mean + half_width]
    else:
        confidence_interval = [mean, mean]
    return {
        "mean": mean,
        "std": std,
        "fold_mean_95ci": [float(value) for value in confidence_interval],
        "n_folds": int(len(finite)),
    }


def summarize_fold_metrics(fold_metrics: list[dict]) -> dict:
    summary = {
        metric: summarize_values([fold.get(metric) for fold in fold_metrics])
        for metric in REPORT_METRICS
    }
    summary["subject_condition_level"] = {
        metric: summarize_values(
            [fold["subject_condition_level"].get(metric) for fold in fold_metrics]
        )
        for metric in REPORT_METRICS
    }
    for metric in (
        "parameter_count",
        "trainable_parameter_count",
        "training_seconds",
        "inference_seconds_total",
        "inference_ms_per_window",
    ):
        summary[metric] = summarize_values([fold.get(metric) for fold in fold_metrics])
    return summary


def baseline_table_row(model_name: str, result: dict) -> dict:
    summary = result.get("research_summary") or summarize_fold_metrics(result["fold_metrics"])
    row = {
        "model": result.get("model", model_name),
        "seed": result["random_seed"],
        "folds": result["n_folds"],
        "parameters": summary["parameter_count"]["mean"],
        "training_seconds_mean": summary["training_seconds"]["mean"],
        "training_seconds_std": summary["training_seconds"]["std"],
        "inference_ms_per_window_mean": summary["inference_ms_per_window"]["mean"],
        "inference_ms_per_window_std": summary["inference_ms_per_window"]["std"],
    }
    for level, source in (
        ("window", summary),
        ("subject_condition", summary["subject_condition_level"]),
    ):
        for metric in REPORT_METRICS:
            metric_summary = source[metric]
            confidence_interval = metric_summary["fold_mean_95ci"]
            row[f"{level}_{metric}_mean"] = metric_summary["mean"]
            row[f"{level}_{metric}_std"] = metric_summary["std"]
            row[f"{level}_{metric}_ci_low"] = (
                confidence_interval[0] if confidence_interval else None
            )
            row[f"{level}_{metric}_ci_high"] = (
                confidence_interval[1] if confidence_interval else None
            )
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_baseline_tables(path: Path, results: dict) -> None:
    write_csv(path, [baseline_table_row(name, result) for name, result in results.items()])


def write_baseline_diagnostic_plots(output_dir: Path, results: dict) -> None:
    project_cache = Path(__file__).resolve().parent.parent / ".cache"
    os.environ.setdefault("XDG_CACHE_HOME", str(project_cache))
    os.environ.setdefault("MPLCONFIGDIR", str(project_cache / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, result in results.items():
        folds = result.get("fold_metrics", [])
        curves = [fold.get("curves") for fold in folds if fold.get("curves")]
        if not curves:
            continue
        matrices = np.asarray([fold["confusion_matrix"] for fold in folds], dtype=float)
        matrix = matrices.sum(axis=0)
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
        grid = np.asarray(curves[0]["grid"])
        mean_tpr = np.mean([curve["roc_tpr"] for curve in curves], axis=0)
        mean_precision = np.mean([curve["pr_precision"] for curve in curves], axis=0)

        figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        image = axes[0].imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axes[0].text(column, row, f"{normalized[row, column]:.2f}", ha="center", va="center")
        axes[0].set(title="Pooled normalized confusion", xlabel="Predicted", ylabel="True")
        figure.colorbar(image, ax=axes[0], fraction=0.046)

        axes[1].plot(grid, mean_tpr, label="Mean ROC")
        axes[1].plot([0, 1], [0, 1], linestyle="--", color="0.5")
        axes[1].set(title="Cross-fold ROC", xlabel="False-positive rate", ylabel="True-positive rate")
        axes[1].legend()

        axes[2].plot(grid, mean_precision, label="Mean PR")
        axes[2].set(title="Cross-fold precision-recall", xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
        axes[2].legend()
        figure.suptitle(model_name)
        figure.tight_layout()
        safe_name = model_name.replace("/", "_")
        figure.savefig(output_dir / f"{safe_name}_classification_diagnostics.png", dpi=200)
        plt.close(figure)


def write_comparison_plot(path: Path, results: dict) -> None:
    project_cache = Path(__file__).resolve().parent.parent / ".cache"
    os.environ.setdefault("XDG_CACHE_HOME", str(project_cache))
    os.environ.setdefault("MPLCONFIGDIR", str(project_cache / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = [
        condition
        for condition in ("without_gan", "with_gan", "with_simple_augmentation")
        if results.get(condition)
    ]
    labels = {
        "without_gan": "Real only",
        "with_gan": "Real + CWGAN-GP",
        "with_simple_augmentation": "Real + simple aug.",
    }
    per_subject = []
    f1_values = []
    for condition in conditions:
        subject_outcomes: dict[int, list[bool]] = {}
        for fold in results[condition]:
            units = fold["subject_condition_level"]["units"]
            for group, truth, prediction in zip(
                units["groups"], units["y_true"], units["y_pred"]
            ):
                subject_outcomes.setdefault(int(group), []).append(truth == prediction)
        per_subject.append([
            float(np.mean(outcomes)) for outcomes in subject_outcomes.values()
        ])
        f1_values.append([
            fold["subject_condition_level"]["f1_macro"]
            for fold in results[condition]
        ])

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot(per_subject, labels=[labels[name] for name in conditions])
    axes[0].set(title="Held-out per-subject accuracy", ylabel="Accuracy", ylim=(-0.02, 1.02))
    axes[1].boxplot(f1_values, labels=[labels[name] for name in conditions])
    axes[1].set(title="Subject-condition macro-F1 by fold", ylabel="Macro-F1", ylim=(-0.02, 1.02))
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def comparison_table_rows(results: dict, model_name: str) -> list[dict]:
    rows = []
    baseline = summarize_fold_metrics(results["without_gan"])
    for condition in ("with_gan", "with_simple_augmentation"):
        if condition not in results or not results[condition]:
            continue
        augmented = summarize_fold_metrics(results[condition])
        paired_tests = results.get("paired_tests", {}).get(condition, {})
        for level, metric in (
            ("window", "f1_macro"),
            ("window", "accuracy"),
            ("subject_condition", "f1_macro"),
            ("subject_condition", "accuracy"),
        ):
            baseline_source = baseline if level == "window" else baseline["subject_condition_level"]
            augmented_source = augmented if level == "window" else augmented["subject_condition_level"]
            test_name = f"{level}_{metric}"
            test = paired_tests.get(test_name, {})
            cluster_test = results.get("subject_cluster_tests", {}).get(
                condition, {}
            ).get(metric, {}) if level == "subject_condition" else {}
            confidence_interval = cluster_test.get("subject_cluster_bootstrap_95ci")
            rows.append({
                "model": model_name,
                "augmentation": condition,
                "synth_fraction": results["metadata"].get("synth_fraction_per_class"),
                "level": level,
                "metric": metric,
                "real_only_mean": baseline_source[metric]["mean"],
                "augmented_mean": augmented_source[metric]["mean"],
                "paired_delta": cluster_test.get(
                    "paired_delta", test.get("mean_paired_delta")
                ),
                "paired_delta_ci_low": confidence_interval[0] if confidence_interval else None,
                "paired_delta_ci_high": confidence_interval[1] if confidence_interval else None,
                "p_value": cluster_test.get(
                    "subject_cluster_randomization_p_value",
                    test.get("exact_two_sided_p_value"),
                ),
                "holm_adjusted_p": cluster_test.get(
                    "holm_adjusted_p_value", test.get("holm_adjusted_p_value")
                ),
            })
    return rows
