"""Signal preprocessing: zero-phase band-pass -> trimming -> windowing ->
class-agnostic amplitude artifact rejection.

Nothing here uses labels or information from other subjects, so the
processed window cache can be shared by every cross-validation fold without
leakage. Normalization is applied later, per window, when model inputs are
built (see ``windows.model_inputs``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .. import constants as C
from ..settings import PreprocessConfig
from .raw import RawRecording, load_stew


def bandpass(data: np.ndarray, config: PreprocessConfig, sfreq: int = C.SFREQ) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis."""
    low, high = config.bandpass_hz
    sos = butter(config.filter_order, [low, high], btype="band", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def sliding_windows(data: np.ndarray, window: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    """(C, S) -> windows (N, C, window) and their start samples (N,)."""
    n_samples = data.shape[-1]
    starts = np.arange(0, n_samples - window + 1, step, dtype=np.int64)
    if len(starts) == 0:
        return np.empty((0, data.shape[0], window), dtype=data.dtype), starts
    windows = np.stack([data[:, start:start + window] for start in starts])
    return windows, starts


def artifact_mask(windows: np.ndarray, k: float | None) -> np.ndarray:
    """True = keep. Reject windows where any channel's peak-to-peak amplitude
    exceeds median + k * MAD computed over the given windows for that channel.
    """
    if k is None or len(windows) == 0:
        return np.ones(len(windows), dtype=bool)
    ptp = windows.max(axis=-1) - windows.min(axis=-1)  # (N, C)
    median = np.median(ptp, axis=0)
    mad = np.median(np.abs(ptp - median), axis=0) + 1e-8
    return np.all(ptp <= median + k * mad, axis=1)


@dataclass
class WindowSet:
    """Filtered (not normalized) EEG windows with full bookkeeping."""
    X: np.ndarray        # (N, C, T) float32, band-passed device units
    y: np.ndarray        # (N,) int64, 0 = rest, 1 = SIMKAP
    subject: np.ndarray  # (N,) int64, 1..48
    start: np.ndarray    # (N,) int64, start sample within the recording
    config: PreprocessConfig
    rejection: dict      # per-subject/per-condition rejection counts

    def __len__(self) -> int:
        return len(self.y)

    @property
    def subjects(self) -> np.ndarray:
        return np.unique(self.subject)

    def subset(self, mask_or_index) -> "WindowSet":
        return WindowSet(
            X=self.X[mask_or_index], y=self.y[mask_or_index],
            subject=self.subject[mask_or_index], start=self.start[mask_or_index],
            config=self.config, rejection=self.rejection,
        )


def preprocess_recordings(recordings: list[RawRecording], config: PreprocessConfig,
                          sfreq: int = C.SFREQ) -> WindowSet:
    window = int(round(config.window_seconds * sfreq))
    step = max(1, int(round(window * (1.0 - config.window_overlap))))
    trim = int(round(config.trim_seconds * sfreq))

    by_subject: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for recording in recordings:
        filtered = bandpass(recording.data, config, sfreq)
        for channel in config.exclude_channels:
            filtered[C.CHANNELS.index(channel)] = 0.0
        if trim:
            filtered = filtered[:, trim:filtered.shape[1] - trim]
        windows, starts = sliding_windows(filtered, window, step)
        by_subject.setdefault(recording.subject, []).append(
            (recording.condition, windows.astype(np.float32), starts + trim)
        )

    X_parts, y_parts, subject_parts, start_parts = [], [], [], []
    rejection = {"k": config.artifact_mad_k, "subjects": {}}
    for subject in sorted(by_subject):
        parts = by_subject[subject]
        all_windows = np.concatenate([w for _, w, _ in parts])
        labels = np.concatenate([np.full(len(w), c, dtype=np.int64) for c, w, _ in parts])
        starts = np.concatenate([s for _, _, s in parts])
        # Threshold from the subject's own windows of BOTH conditions so the
        # criterion is identical for rest and task (class-agnostic).
        keep = artifact_mask(all_windows, config.artifact_mad_k)
        rejection["subjects"][int(subject)] = {
            str(c): {"total": int((labels == c).sum()), "rejected": int(((labels == c) & ~keep).sum())}
            for c in (0, 1)
        }
        X_parts.append(all_windows[keep])
        y_parts.append(labels[keep])
        subject_parts.append(np.full(int(keep.sum()), subject, dtype=np.int64))
        start_parts.append(starts[keep])

    total = {c: sum(v[str(c)]["total"] for v in rejection["subjects"].values()) for c in (0, 1)}
    rejected = {c: sum(v[str(c)]["rejected"] for v in rejection["subjects"].values()) for c in (0, 1)}
    rejection["summary"] = {
        str(c): {"total": total[c], "rejected": rejected[c],
                 "rate": rejected[c] / max(1, total[c])} for c in (0, 1)
    }
    return WindowSet(
        X=np.concatenate(X_parts), y=np.concatenate(y_parts),
        subject=np.concatenate(subject_parts), start=np.concatenate(start_parts),
        config=config, rejection=rejection,
    )


def cache_path(config: PreprocessConfig, cache_dir=None):
    return Path(cache_dir or C.PROCESSED_DIR) / f"windows_{config.cache_key()}.npz"


def save_windows(windows: WindowSet, path=None):
    path = path or cache_path(windows.config)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, X=windows.X, y=windows.y, subject=windows.subject, start=windows.start,
        config_json=np.array(json.dumps(asdict(windows.config))),
        rejection_json=np.array(json.dumps(windows.rejection)),
    )
    return path


def load_windows(config: PreprocessConfig, raw_dir=None, cache_dir=None, strict: bool = True,
                 build_if_missing: bool = True) -> WindowSet:
    path = cache_path(config, cache_dir)
    if not path.exists():
        if not build_if_missing:
            raise FileNotFoundError(f"{path} not found; run `stewbench preprocess` first.")
        windows = preprocess_recordings(load_stew(raw_dir, strict=strict), config)
        save_windows(windows, path)
        return windows
    with np.load(path, allow_pickle=False) as data:
        stored = json.loads(str(data["config_json"]))
        expected = json.loads(json.dumps(asdict(config)))
        # Caches written before `exclude_channels` existed have no such key.
        if not expected.get("exclude_channels"):
            expected.pop("exclude_channels", None)
            stored.pop("exclude_channels", None)
        if stored != expected:
            raise ValueError(f"{path} was built with a different preprocessing config: {stored}")
        return WindowSet(
            X=data["X"], y=data["y"], subject=data["subject"], start=data["start"],
            config=config, rejection=json.loads(str(data["rejection_json"])),
        )
