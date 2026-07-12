"""
Phase 1 preprocessing: bandpass/notch filtering -> baseline correction ->
sliding-window epoching -> per-subject z-score normalization.

Design notes (based on actual DREAMER shapes confirmed via data_loader.py):
  - Baseline clips are a fixed 7808 samples (61s) every trial.
  - Stimuli clips vary 8576-50432 samples (67s-394s) -> variable window counts
    per trial, handled naturally by sliding-window epoching per clip.

Usage:
    python -m src.preprocessing   # runs full pipeline, saves to data/processed/
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from . import config
from .data_loader import SubjectData, load_dreamer_mat


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
    """Apply bandpass then notch filter. signal shape: (M, C)."""
    filtered = _bandpass_filter(signal, fs)
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


# ---------------------------------------------------------------------------
# Full per-subject pipeline
# ---------------------------------------------------------------------------
@dataclass
class ProcessedSubject:
    subject_id: int
    windows: np.ndarray  # (N, T, C) float32 -- all windows across all 18 trials
    trial_idx: np.ndarray  # (N,) which of the 18 trials each window came from (0-17)
    valence: np.ndarray  # (18,) raw scores, indexed by trial_idx to label windows later
    arousal: np.ndarray  # (18,)
    dominance: np.ndarray  # (18,)


def process_subject(subject: SubjectData) -> ProcessedSubject:
    all_windows = []
    all_trial_idx = []

    for trial_i in range(config.N_VIDEOS):
        baseline_raw = subject.eeg_baseline[trial_i]
        stimuli_raw = subject.eeg_stimuli[trial_i]

        baseline_filt = filter_signal(baseline_raw)
        stimuli_filt = filter_signal(stimuli_raw)

        if config.APPLY_BASELINE_CORRECTION:
            stimuli_corrected = baseline_correct(stimuli_filt, baseline_filt)
        else:
            stimuli_corrected = stimuli_filt

        windows = sliding_window_epochs(stimuli_corrected)
        if windows.shape[0] == 0:
            # Clip too short for even one window at this window size -- flag it.
            print(
                f"  [warn] subject {subject.subject_id} trial {trial_i}: "
                f"clip too short ({stimuli_corrected.shape[0]} samples) for "
                f"window size {config.WINDOW_SAMPLES} -- skipped."
            )
            continue

        all_windows.append(windows)
        all_trial_idx.append(np.full(windows.shape[0], trial_i, dtype=np.int32))

    windows_concat = np.concatenate(all_windows, axis=0)  # (N, T, C)
    trial_idx_concat = np.concatenate(all_trial_idx, axis=0)  # (N,)

    n_before = windows_concat.shape[0]
    if config.APPLY_ARTIFACT_REJECTION:
        keep_mask = reject_artifact_windows(windows_concat)
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


def process_all_subjects(subjects: list[SubjectData]) -> list[ProcessedSubject]:
    processed = []
    for s in subjects:
        print(f"Processing subject {s.subject_id}/{config.N_SUBJECTS}...")
        processed.append(process_subject(s))
    return processed


# ---------------------------------------------------------------------------
# Save / load processed data
# ---------------------------------------------------------------------------
def save_processed(processed: list[ProcessedSubject], out_dir=config.DATA_PROCESSED) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in processed:
        np.savez_compressed(
            out_dir / f"subject_{p.subject_id:02d}.npz",
            windows=p.windows,
            trial_idx=p.trial_idx,
            valence=p.valence,
            arousal=p.arousal,
            dominance=p.dominance,
        )
    print(f"Saved {len(processed)} subject files to {out_dir}")


def load_processed(in_dir=config.DATA_PROCESSED) -> list[ProcessedSubject]:
    files = sorted(in_dir.glob("subject_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No processed files found in {in_dir}. Run `python -m src.preprocessing` first."
        )
    processed = []
    for f in files:
        data = np.load(f)
        subject_id = int(f.stem.split("_")[1])
        processed.append(
            ProcessedSubject(
                subject_id=subject_id,
                windows=data["windows"],
                trial_idx=data["trial_idx"],
                valence=data["valence"],
                arousal=data["arousal"],
                dominance=data["dominance"],
            )
        )
    return processed


if __name__ == "__main__":
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

    save_processed(processed)
