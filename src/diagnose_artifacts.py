"""Compare artifact-rejection thresholds on raw DREAMER or STEW windows."""

import argparse

import numpy as np

from . import config
from .data_loader import load_dreamer_mat, load_stew
from .preprocessing import (
    baseline_correct,
    filter_signal,
    reject_artifact_windows,
    sliding_window_epochs,
)


def dreamer_windows(max_subjects: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subjects = load_dreamer_mat()
    sampling_rate = config.sampling_rate_hz("dreamer")
    window_samples = config.window_samples("dreamer")
    if max_subjects:
        subjects = subjects[:max_subjects]
    windows, subject_ids = [], []
    for subject in subjects:
        for trial_i in range(config.N_VIDEOS):
            baseline = filter_signal(subject.eeg_baseline[trial_i], fs=sampling_rate)
            stimuli = filter_signal(subject.eeg_stimuli[trial_i], fs=sampling_rate)
            signal = baseline_correct(stimuli, baseline) if config.APPLY_BASELINE_CORRECTION else stimuli
            trial_windows = sliding_window_epochs(signal, window_samples=window_samples)
            windows.append(trial_windows)
            subject_ids.append(np.full(len(trial_windows), subject.subject_id, dtype=np.int64))
    combined = np.concatenate(windows)
    return combined, np.zeros(len(combined), dtype=np.int64), np.concatenate(subject_ids)


def stew_windows(max_subjects: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subjects = load_stew()
    sampling_rate = config.sampling_rate_hz("stew")
    window_samples = config.window_samples("stew")
    if max_subjects:
        subjects = subjects[:max_subjects]
    windows, labels, subject_ids = [], [], []
    for subject in subjects:
        for raw, label in ((subject.eeg_lo, 0), (subject.eeg_hi, 1)):
            filtered = filter_signal(raw, fs=sampling_rate)
            condition_windows = sliding_window_epochs(
                filtered, window_samples=window_samples
            )
            windows.append(condition_windows)
            labels.append(np.full(len(condition_windows), label, dtype=np.int64))
            subject_ids.append(
                np.full(len(condition_windows), subject.subject_id, dtype=np.int64)
            )
    return np.concatenate(windows), np.concatenate(labels), np.concatenate(subject_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("dreamer", "stew"), default=config.DEFAULT_DATASET)
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--multipliers", type=float, nargs="+", default=[20, 30, 40, 50, 60, 80])
    args = parser.parse_args()

    windows, labels, subject_ids = (
        stew_windows(args.max_subjects)
        if args.dataset == "stew"
        else dreamer_windows(args.max_subjects)
    )
    print(f"Dataset={args.dataset}, candidate windows={len(windows)}")
    for multiplier in args.multipliers:
        keep = np.ones(len(windows), dtype=bool)
        for subject_id in np.unique(subject_ids):
            subject_mask = subject_ids == subject_id
            keep[subject_mask] = reject_artifact_windows(
                windows[subject_mask], mad_multiplier=multiplier
            )
        line = f"MAD={multiplier:5.1f}: rejected={100 * (~keep).mean():5.2f}%"
        if args.dataset == "stew":
            lo_rate = 100 * (~keep[labels == 0]).mean()
            hi_rate = 100 * (~keep[labels == 1]).mean()
            line += f" (lo={lo_rate:5.2f}%, hi={hi_rate:5.2f}%)"
        print(line)

    configured = config.artifact_rejection_mad_multiplier(args.dataset)
    print(f"Configured multiplier for {args.dataset}: {configured}")


if __name__ == "__main__":
    main()
