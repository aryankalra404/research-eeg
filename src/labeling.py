"""
Converts processed subject files into binary labels.

DREAMER uses arousal/valence self-report scores as a binary stress proxy:
    high arousal + low valence  ->  stress = 1
    everything else             ->  stress = 0

STEW currently stores direct window labels during preprocessing:
    lo/rest = 0
    hi/multitasking = 1

Threshold method is controlled by config.LABEL_THRESHOLD_METHOD:
    'median_split' -> per-subject median of that subject's 18 arousal/valence
                       scores (data-driven, accounts for subjects using the
                       1-5 scale differently)
    'fixed'        -> a fixed scale midpoint (config.FIXED_AROUSAL_THRESHOLD /
                       config.FIXED_VALENCE_THRESHOLD), applied globally

IMPORTANT: this label is a PROXY for stress, not a validated stress measure.
State this explicitly in the paper -- see README.md limitations section.

Usage:
    python -m src.labeling --dataset stew
"""

import numpy as np

from . import config
from .preprocessing import ProcessedSubject, load_processed


def compute_trial_labels(subject: ProcessedSubject) -> np.ndarray:
    """
    Returns an (18,) binary array: 1 = stress-proxy, 0 = non-stress, for
    each of the subject's 18 trials.
    """
    if subject.arousal is None or subject.valence is None:
        raise ValueError(
            f"Subject {subject.subject_id} does not have DREAMER arousal/valence fields. "
            f"Use direct labels for this dataset."
        )
    arousal = subject.arousal
    valence = subject.valence

    if config.LABEL_THRESHOLD_METHOD == "median_split":
        arousal_thresh = np.median(arousal)
        valence_thresh = np.median(valence)
    elif config.LABEL_THRESHOLD_METHOD == "fixed":
        arousal_thresh = config.FIXED_AROUSAL_THRESHOLD
        valence_thresh = config.FIXED_VALENCE_THRESHOLD
    else:
        raise ValueError(f"Unknown LABEL_THRESHOLD_METHOD: {config.LABEL_THRESHOLD_METHOD}")

    high_arousal = arousal >= arousal_thresh
    low_valence = valence <= valence_thresh
    stress_label = (high_arousal & low_valence).astype(np.int64)
    return stress_label


def label_windows(subject: ProcessedSubject) -> np.ndarray:
    """
    Broadcasts trial-level labels to every window belonging to that trial.
    Returns (N,) array matching subject.windows.shape[0].
    """
    if subject.labels is not None:
        return subject.labels.astype(np.int64)

    trial_labels = compute_trial_labels(subject)  # (18,)
    window_labels = trial_labels[subject.trial_idx]  # (N,) via fancy indexing
    return window_labels


def build_dataset(processed: list[ProcessedSubject]):
    """
    Assembles the full dataset across all subjects into flat arrays, ready
    for a PyTorch Dataset / sklearn split.

    Returns:
        X: (total_N, T, C) float32
        y: (total_N,) int64  -- binary stress label
        groups: (total_N,) int64  -- subject_id, for subject-independent CV
    """
    X_list, y_list, groups_list = [], [], []
    for subject in processed:
        labels = label_windows(subject)
        X_list.append(subject.windows)
        y_list.append(labels)
        groups_list.append(np.full(labels.shape[0], subject.subject_id, dtype=np.int64))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)
    return X, y, groups


def summarize_labels(processed: list[ProcessedSubject], dataset: str = config.DEFAULT_DATASET) -> None:
    X, y, groups = build_dataset(processed)

    n_total = len(y)
    n_class1 = int(y.sum())
    n_class0 = n_total - n_class1
    class0_name, class1_name = config.class_names(dataset)

    print(f"Total windows: {n_total}")
    print(f"  class 1 ({class1_name}): {n_class1}  ({100*n_class1/n_total:.1f}%)")
    print(f"  class 0 ({class0_name}): {n_class0}  ({100*n_class0/n_total:.1f}%)")
    print(f"X shape: {X.shape}   (windows, T, C)")

    print("\nPer-subject class-1 fraction (check for subjects that are all one class):")
    for subj_id in sorted(set(groups)):
        mask = groups == subj_id
        frac = y[mask].mean()
        flag = "  <-- check this one" if frac in (0.0, 1.0) else ""
        print(f"  subject {subj_id:2d}: {frac:.2f}{flag}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=config.DEFAULT_DATASET, choices=config.SUPPORTED_DATASETS)
    args = parser.parse_args()

    processed = load_processed(dataset=args.dataset)
    summarize_labels(processed, dataset=args.dataset)
