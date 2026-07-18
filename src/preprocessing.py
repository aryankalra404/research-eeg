"""
Phase 1 preprocessing: bandpass/notch filtering -> baseline correction ->
sliding-window epoching -> per-subject z-score normalization.

Design notes:
  - DREAMER:
  - Baseline clips are a fixed 7808 samples (61s) every trial.
  - Stimuli clips vary 8576-50432 samples (67s-394s) -> variable window counts
    per trial, handled naturally by sliding-window epoching per clip.
  - STEW:
    - Each subject has one rest/low workload recording and one high workload
      multitasking recording, both sampled at 128Hz with 14 channels.
    - Labels are condition-based for now: lo=0, hi=1.

Usage:
    python -m src.preprocessing --dataset stew
"""

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from . import config
from .data_loader import STEWSubjectData, SubjectData, load_dreamer_mat, load_stew


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def _bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass. signal shape: (M, C)."""
    nyq = fs / 2.0
    low = config.BANDPASS_LOW_HZ / nyq
    high = config.BANDPASS_HIGH_HZ / nyq
    b, a = butter(config.FILTER_ORDER, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=0)


def _notch_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase notch at mains frequency (50Hz for DREAMER/UK data)."""
    freq = config.NOTCH_FREQ_HZ
    q = 30.0  # quality factor
    b, a = iirnotch(freq / (fs / 2.0), q)
    return filtfilt(b, a, signal, axis=0)


def filter_signal(signal: np.ndarray, fs: float = config.EEG_SAMPLING_RATE) -> np.ndarray:
    """Apply bandpass then notch filter. signal shape: (M, C).
    The notch filter only runs if NOTCH_FREQ_HZ actually falls within the
    bandpass passband -- otherwise the bandpass has already removed that
    frequency content and the notch would be a silent no-op (e.g. bandpass
    0.5-45Hz followed by a 50Hz notch does nothing useful)."""
    filtered = _bandpass_filter(signal, fs)
    if config.BANDPASS_LOW_HZ < config.NOTCH_FREQ_HZ < config.BANDPASS_HIGH_HZ:
        filtered = _notch_filter(filtered, fs)
    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# Baseline correction
# ---------------------------------------------------------------------------
def baseline_correct(stimuli: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """
    Subtracts the per-channel mean of the (filtered) baseline clip from the
    (filtered) stimuli clip. Both must already be filtered before calling this.

    stimuli:  (M_stim, C)
    baseline: (M_base, C)   -- fixed 7808 samples in DREAMER
    """
    baseline_mean = baseline.mean(axis=0, keepdims=True)  # (1, C)
    return stimuli - baseline_mean


# ---------------------------------------------------------------------------
# Artifact rejection
# ---------------------------------------------------------------------------
def reject_artifact_windows(
    windows: np.ndarray,
    mad_multiplier: float = config.ARTIFACT_REJECTION_MAD_MULTIPLIER,
) -> np.ndarray:
    """
    Flags windows containing amplitude artifacts (eye blinks, muscle noise,
    electrode pops) using a per-subject adaptive threshold: for each channel,
    compute the peak-to-peak amplitude of every window, then reject any window
    where any channel's peak-to-peak exceeds median + mad_multiplier * MAD
    (median absolute deviation) for that channel.

    This is a simple, fast alternative to full ICA-based artifact removal --
    it won't remove artifacts *within* a kept window, only discards windows
    that are badly contaminated. State this as a limitation vs. full ICA in
    the paper.

    windows: (N, T, C)
    Returns: boolean mask of shape (N,), True = keep.
    """
    if windows.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    # Peak-to-peak amplitude per window, per channel: (N, C)
    ptp = windows.max(axis=1) - windows.min(axis=1)

    keep_mask = np.ones(windows.shape[0], dtype=bool)
    for c in range(windows.shape[2]):
        channel_ptp = ptp[:, c]
        median = np.median(channel_ptp)
        mad = np.median(np.abs(channel_ptp - median)) + 1e-8
        threshold = median + mad_multiplier * mad
        keep_mask &= channel_ptp <= threshold

    return keep_mask


def diagnose_rejection_thresholds(windows: np.ndarray, multipliers=(3, 5, 8, 10, 15, 20, 30)) -> None:
    """
    Prints the rejection rate at several MAD multiplier values, so you can
    pick a threshold based on evidence rather than guessing. Run this once
    on a subject or two before committing to a value in config.py.
    """
    print(f"  Rejection rate by MAD multiplier (n_windows={windows.shape[0]}):")
    for m in multipliers:
        mask = reject_artifact_windows(windows, mad_multiplier=m)
        rejected_pct = 100 * (~mask).sum() / len(mask)
        print(f"    multiplier={m:5.1f}  ->  {rejected_pct:5.1f}% rejected")


# ---------------------------------------------------------------------------
# Epoching (sliding window)
# ---------------------------------------------------------------------------
def sliding_window_epochs(
    signal: np.ndarray,
    window_samples: int = config.WINDOW_SAMPLES,
    overlap: float = config.WINDOW_OVERLAP,
) -> np.ndarray:
    """
    Splits a (M, C) signal into overlapping fixed-length windows.
    Returns array of shape (n_windows, window_samples, C).
    Drops any trailing partial window shorter than window_samples.
    """
    step = int(window_samples * (1 - overlap))
    step = max(step, 1)
    n_samples = signal.shape[0]

    windows = []
    start = 0
    while start + window_samples <= n_samples:
        windows.append(signal[start : start + window_samples])
        start += step

    if len(windows) == 0:
        return np.empty((0, window_samples, signal.shape[1]), dtype=signal.dtype)
    return np.stack(windows, axis=0)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def zscore_normalize(windows: np.ndarray) -> np.ndarray:
    """
    Per-subject, per-channel z-score. windows shape: (n_windows, T, C).
    Stats computed across all windows+timepoints for that subject (i.e. over
    axes 0 and 1), applied per channel -- this is what "per_subject_zscore"
    means in config.py.
    """
    mean = windows.mean(axis=(0, 1), keepdims=True)  # (1, 1, C)
    std = windows.std(axis=(0, 1), keepdims=True) + 1e-8
    return (windows - mean) / std


def zscore_normalize_per_window(windows: np.ndarray) -> np.ndarray:
    """Normalize each window/channel independently without cross-window state."""
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True) + 1e-8
    return (windows - mean) / std


# ---------------------------------------------------------------------------
# Full per-subject pipeline
# ---------------------------------------------------------------------------
@dataclass
class ProcessedSubject:
    subject_id: int
    windows: np.ndarray  # (N, T, C) float32 -- all windows across all 18 trials
    trial_idx: np.ndarray  # (N,) which of the 18 trials each window came from (0-17)
    valence: np.ndarray | None = None  # DREAMER (18,) raw scores
    arousal: np.ndarray | None = None
    dominance: np.ndarray | None = None
    labels: np.ndarray | None = None  # Optional direct window labels, e.g. STEW lo/hi
    rating_lo: float | None = None
    rating_hi: float | None = None


def process_subject(subject: SubjectData) -> ProcessedSubject:
    all_windows = []
    all_trial_idx = []
    sampling_rate = config.sampling_rate_hz("dreamer")
    window_samples = config.window_samples("dreamer")

    for trial_i in range(config.N_VIDEOS):
        baseline_raw = subject.eeg_baseline[trial_i]
        stimuli_raw = subject.eeg_stimuli[trial_i]

        baseline_filt = filter_signal(baseline_raw, fs=sampling_rate)
        stimuli_filt = filter_signal(stimuli_raw, fs=sampling_rate)

        if config.APPLY_BASELINE_CORRECTION:
            stimuli_corrected = baseline_correct(stimuli_filt, baseline_filt)
        else:
            stimuli_corrected = stimuli_filt

        windows = sliding_window_epochs(stimuli_corrected, window_samples=window_samples)
        if windows.shape[0] == 0:
            # Clip too short for even one window at this window size -- flag it.
            print(
                f"  [warn] subject {subject.subject_id} trial {trial_i}: "
                f"clip too short ({stimuli_corrected.shape[0]} samples) for "
                f"window size {window_samples} -- skipped."
            )
            continue

        all_windows.append(windows)
        all_trial_idx.append(np.full(windows.shape[0], trial_i, dtype=np.int32))

    windows_concat = np.concatenate(all_windows, axis=0)  # (N, T, C)
    trial_idx_concat = np.concatenate(all_trial_idx, axis=0)  # (N,)

    n_before = windows_concat.shape[0]
    if config.APPLY_ARTIFACT_REJECTION:
        keep_mask = reject_artifact_windows(
            windows_concat,
            mad_multiplier=config.artifact_rejection_mad_multiplier("dreamer"),
        )
        windows_concat = windows_concat[keep_mask]
        trial_idx_concat = trial_idx_concat[keep_mask]
        n_after = windows_concat.shape[0]
        n_rejected = n_before - n_after
        print(f"  subject {subject.subject_id}: rejected {n_rejected}/{n_before} "
              f"windows ({100*n_rejected/n_before:.1f}%) for amplitude artifacts")
        if n_after < config.ARTIFACT_REJECTION_MIN_WINDOWS_WARN:
            print(f"  [warn] subject {subject.subject_id} has only {n_after} windows "
                  f"remaining after artifact rejection -- check this subject's data quality.")

    windows_norm = zscore_normalize(windows_concat)

    return ProcessedSubject(
        subject_id=subject.subject_id,
        windows=windows_norm,
        trial_idx=trial_idx_concat,
        valence=subject.valence,
        arousal=subject.arousal,
        dominance=subject.dominance,
    )


def process_stew_subject(subject: STEWSubjectData) -> ProcessedSubject:
    condition_windows = []
    condition_trial_idx = []
    condition_labels = []
    sampling_rate = config.sampling_rate_hz("stew")
    window_samples = config.window_samples("stew")

    for trial_i, (condition_name, raw, label) in enumerate(
        [("lo", subject.eeg_lo, 0), ("hi", subject.eeg_hi, 1)]
    ):
        filtered = filter_signal(raw, fs=sampling_rate)
        windows = sliding_window_epochs(filtered, window_samples=window_samples)
        if windows.shape[0] == 0:
            print(
                f"  [warn] STEW subject {subject.subject_id} {condition_name}: "
                f"recording too short ({filtered.shape[0]} samples) for "
                f"window size {window_samples} -- skipped."
            )
            continue

        condition_windows.append(windows)
        condition_trial_idx.append(np.full(windows.shape[0], trial_i, dtype=np.int32))
        condition_labels.append(np.full(windows.shape[0], label, dtype=np.int64))

    if not condition_windows:
        raise ValueError(f"STEW subject {subject.subject_id} produced no windows.")

    windows_concat = np.concatenate(condition_windows, axis=0)
    trial_idx_concat = np.concatenate(condition_trial_idx, axis=0)
    labels_concat = np.concatenate(condition_labels, axis=0)

    n_before = windows_concat.shape[0]
    if config.APPLY_ARTIFACT_REJECTION:
        keep_mask = reject_artifact_windows(
            windows_concat,
            mad_multiplier=config.artifact_rejection_mad_multiplier("stew"),
        )
        windows_concat = windows_concat[keep_mask]
        trial_idx_concat = trial_idx_concat[keep_mask]
        labels_concat = labels_concat[keep_mask]
        n_after = windows_concat.shape[0]
        n_rejected = n_before - n_after
        print(f"  STEW subject {subject.subject_id}: rejected {n_rejected}/{n_before} "
              f"windows ({100*n_rejected/n_before:.1f}%) for amplitude artifacts")
        if n_after < config.ARTIFACT_REJECTION_MIN_WINDOWS_WARN:
            print(f"  [warn] STEW subject {subject.subject_id} has only {n_after} windows "
                  f"remaining after artifact rejection -- check this subject's data quality.")

    if config.STEW_NORMALIZATION != "per_window_channel_zscore":
        raise ValueError(f"Unsupported STEW normalization: {config.STEW_NORMALIZATION}")
    windows_norm = zscore_normalize_per_window(windows_concat)

    return ProcessedSubject(
        subject_id=subject.subject_id,
        windows=windows_norm.astype(np.float32),
        trial_idx=trial_idx_concat,
        labels=labels_concat,
        rating_lo=subject.rating_lo,
        rating_hi=subject.rating_hi,
    )


def process_all_subjects(subjects: list[SubjectData]) -> list[ProcessedSubject]:
    processed = []
    for s in subjects:
        print(f"Processing subject {s.subject_id}/{config.N_SUBJECTS}...")
        processed.append(process_subject(s))
    return processed


def process_all_stew_subjects(subjects: list[STEWSubjectData]) -> list[ProcessedSubject]:
    processed = []
    for s in subjects:
        print(f"Processing STEW subject {s.subject_id}/{config.STEW_N_SUBJECTS}...")
        processed.append(process_stew_subject(s))
    return processed


# ---------------------------------------------------------------------------
# Save / load processed data
# ---------------------------------------------------------------------------
def save_processed(processed: list[ProcessedSubject], out_dir=None) -> None:
    out_dir = out_dir or config.processed_dir("dreamer")
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in processed:
        payload = {
            "windows": p.windows,
            "trial_idx": p.trial_idx,
        }
        if p.valence is not None:
            payload["valence"] = p.valence
        if p.arousal is not None:
            payload["arousal"] = p.arousal
        if p.dominance is not None:
            payload["dominance"] = p.dominance
        if p.labels is not None:
            payload["labels"] = p.labels
        if p.rating_lo is not None:
            payload["rating_lo"] = np.array(p.rating_lo, dtype=np.float32)
        if p.rating_hi is not None:
            payload["rating_hi"] = np.array(p.rating_hi, dtype=np.float32)
        np.savez_compressed(out_dir / f"subject_{p.subject_id:02d}.npz", **payload)
    print(f"Saved {len(processed)} subject files to {out_dir}")


def load_processed(in_dir=None, dataset: str = config.DEFAULT_DATASET) -> list[ProcessedSubject]:
    in_dir = in_dir or config.processed_dir(dataset)
    files = sorted(in_dir.glob("subject_*.npz"))
    if not files and config.normalize_dataset_name(dataset) == "dreamer":
        legacy_files = sorted(config.DATA_PROCESSED.glob("subject_*.npz"))
        if legacy_files:
            in_dir = config.DATA_PROCESSED
            files = legacy_files
    if not files:
        raise FileNotFoundError(
            f"No processed files found in {in_dir}. Run `python -m src.preprocessing --dataset {dataset}` first."
        )
    processed = []
    for f in files:
        data = np.load(f)
        subject_id = int(f.stem.split("_")[1])
        keys = set(data.files)
        processed.append(
            ProcessedSubject(
                subject_id=subject_id,
                windows=data["windows"],
                trial_idx=data["trial_idx"],
                valence=data["valence"] if "valence" in keys else None,
                arousal=data["arousal"] if "arousal" in keys else None,
                dominance=data["dominance"] if "dominance" in keys else None,
                labels=data["labels"] if "labels" in keys else None,
                rating_lo=float(data["rating_lo"]) if "rating_lo" in keys else None,
                rating_hi=float(data["rating_hi"]) if "rating_hi" in keys else None,
            )
        )
    return processed


def run_preprocessing(dataset: str = config.DEFAULT_DATASET):
    dataset = config.normalize_dataset_name(dataset)
    if dataset == "stew":
        print("Loading raw STEW data...")
        subjects = load_stew()

        print("\nRunning STEW preprocessing pipeline (filter -> epoch -> normalize)...")
        processed = process_all_stew_subjects(subjects)

        total_windows = sum(p.windows.shape[0] for p in processed)
        print(f"\nTotal windows across all STEW subjects: {total_windows}")
        print(f"Window shape: {processed[0].windows.shape[1:]} (T, C)")
        print(f"Windows per subject: min={min(p.windows.shape[0] for p in processed)}, "
              f"max={max(p.windows.shape[0] for p in processed)}, "
              f"mean={total_windows / len(processed):.1f}")

        save_processed(processed, out_dir=config.processed_dir(dataset))
        return processed
    if dataset != "dreamer":
        raise NotImplementedError(
            f"{dataset.upper()} preprocessing is not implemented yet. Add a dataset loader first, "
            f"then save standardized subject_*.npz files to {config.processed_dir(dataset)}."
        )

    print("Loading raw DREAMER data...")
    subjects = load_dreamer_mat()

    print("\nRunning preprocessing pipeline (filter -> baseline correct -> epoch -> normalize)...")
    processed = process_all_subjects(subjects)

    total_windows = sum(p.windows.shape[0] for p in processed)
    print(f"\nTotal windows across all subjects: {total_windows}")
    print(f"Window shape: {processed[0].windows.shape[1:]} (T, C)")
    print(f"Windows per subject: min={min(p.windows.shape[0] for p in processed)}, "
          f"max={max(p.windows.shape[0] for p in processed)}, "
          f"mean={total_windows / len(processed):.1f}")

    save_processed(processed, out_dir=config.processed_dir(dataset))
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=config.DEFAULT_DATASET, choices=config.SUPPORTED_DATASETS)
    args = parser.parse_args()
    run_preprocessing(args.dataset)
