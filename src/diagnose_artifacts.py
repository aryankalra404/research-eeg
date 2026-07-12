"""
Run this BEFORE re-running the full preprocessing pipeline, to check what
rejection rate different MAD multipliers actually produce on your real data.
Avoids guessing a threshold and discovering it's wrong after a full run.

Usage:
    python -m src.diagnose_artifacts
"""

from . import config
from .data_loader import load_dreamer_mat
from .preprocessing import filter_signal, baseline_correct, sliding_window_epochs, diagnose_rejection_thresholds
import numpy as np


if __name__ == "__main__":
    subjects = load_dreamer_mat()

    # Check a handful of subjects (not all 23, this is just a threshold-picking tool)
    check_subjects = subjects[:5]

    all_windows = []
    for subject in check_subjects:
        for trial_i in range(config.N_VIDEOS):
            baseline_filt = filter_signal(subject.eeg_baseline[trial_i])
            stimuli_filt = filter_signal(subject.eeg_stimuli[trial_i])
            corrected = baseline_correct(stimuli_filt, baseline_filt) if config.APPLY_BASELINE_CORRECTION else stimuli_filt
            windows = sliding_window_epochs(corrected)
            if windows.shape[0] > 0:
                all_windows.append(windows)

    windows_concat = np.concatenate(all_windows, axis=0)
    print(f"Checking thresholds on {len(check_subjects)} subjects, {windows_concat.shape[0]} total windows\n")
    diagnose_rejection_thresholds(windows_concat)

    print(f"\nCurrent config.ARTIFACT_REJECTION_MAD_MULTIPLIER = {config.ARTIFACT_REJECTION_MAD_MULTIPLIER}")
    print("Pick a multiplier that rejects roughly 2-10% of windows (typical for amplitude-")
    print("based EEG rejection), then update config.py before re-running the full pipeline.")
