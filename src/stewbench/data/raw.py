"""Loader for the raw STEW text files.

Expected layout (as distributed on IEEE DataPort)::

    data/raw/stew/sub01_lo.txt   rest / low workload, 19200 x 14
    data/raw/stew/sub01_hi.txt   SIMKAP multitasking / high workload
    ...
    data/raw/stew/ratings.txt    "subject, rest rating, test rating"

Files may also sit one directory deeper (e.g. ``data/raw/stew/STEW Dataset/``);
the loader searches recursively for ``sub01_lo.txt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import constants as C


@dataclass
class RawRecording:
    subject: int
    condition: int  # 0 = lo (rest), 1 = hi (SIMKAP)
    data: np.ndarray  # (n_channels, n_samples) float64, device units (~microvolts)


def find_raw_dir(root: Path | None = None) -> Path:
    root = Path(root or C.RAW_DIR)
    if (root / "sub01_lo.txt").exists():
        return root
    matches = sorted(root.rglob("sub01_lo.txt"))
    if not matches:
        raise FileNotFoundError(
            f"No STEW files found under {root}. Download STEW from IEEE DataPort "
            "(https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset) "
            f"and extract sub01_lo.txt ... sub48_hi.txt, ratings.txt into {root}."
        )
    return matches[0].parent


def _read_recording(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=np.float64)
    data = frame.to_numpy()
    if data.ndim != 2 or data.shape[1] != C.N_CHANNELS:
        raise ValueError(f"{path.name}: expected (samples, {C.N_CHANNELS}), got {data.shape}")
    return data.T.copy()


def load_ratings(raw_dir: Path) -> dict[int, tuple[float, float]]:
    path = raw_dir / "ratings.txt"
    if not path.exists():
        return {}
    table = np.atleast_2d(np.loadtxt(path, delimiter=",", dtype=float))
    return {int(row[0]): (float(row[1]), float(row[2])) for row in table}


def available_subjects(raw_dir: Path) -> list[int]:
    return sorted(int(p.name[3:5]) for p in raw_dir.glob("sub[0-9][0-9]_lo.txt"))


def load_stew(raw_dir: Path | None = None, subjects: list[int] | None = None,
              strict: bool = True) -> list[RawRecording]:
    """``strict`` (default) requires all 48 subjects with full-length
    recordings, i.e. the real dataset. Tests use ``strict=False``."""
    raw_dir = find_raw_dir(raw_dir)
    if subjects is None:
        subjects = available_subjects(raw_dir)
        if strict and subjects != list(range(1, C.N_SUBJECTS + 1)):
            raise ValueError(f"Expected STEW subjects 1..{C.N_SUBJECTS} in {raw_dir}, found {subjects}")
    recordings = []
    for subject in subjects:
        for condition, tag in ((0, "lo"), (1, "hi")):
            path = raw_dir / f"sub{subject:02d}_{tag}.txt"
            if not path.exists():
                raise FileNotFoundError(f"Missing STEW file {path}")
            data = _read_recording(path)
            if strict and data.shape[1] != C.SAMPLES_PER_RECORDING:
                raise ValueError(
                    f"{path.name}: expected {C.SAMPLES_PER_RECORDING} samples, got {data.shape[1]}"
                )
            recordings.append(RawRecording(subject=subject, condition=condition, data=data))
    return recordings
