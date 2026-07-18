"""
Loads raw EEG datasets into clean, subject-indexed Python structures.

DREAMER.mat structure (from the official readme):
    DREAMER.Data{i}                -> per-subject struct
        .Age, .Gender
        .EEG.baseline{18x1}        -> each cell: M x 14 matrix (neutral clip before stimulus)
        .EEG.stimuli{18x1}         -> each cell: M x 14 matrix (film clip)
        .ECG.baseline / .stimuli   -> M x 2 matrices (not used in this pipeline)
        .ScoreValence[18x1]
        .ScoreArousal[18x1]
        .ScoreDominance[18x1]

Usage:
    python -m src.data_loader   # sanity-check load + print summary
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


@dataclass
class SubjectData:
    subject_id: int
    age: str
    gender: str
    eeg_baseline: list  # 18 arrays, each (M, 14)
    eeg_stimuli: list  # 18 arrays, each (M, 14)
    valence: np.ndarray  # (18,)
    arousal: np.ndarray  # (18,)
    dominance: np.ndarray  # (18,)


@dataclass
class STEWSubjectData:
    subject_id: int
    eeg_lo: np.ndarray  # rest / low workload, shape (M, 14)
    eeg_hi: np.ndarray  # SIMKAP multitasking / high workload, shape (M, 14)
    rating_lo: float | None
    rating_hi: float | None


def load_dreamer_mat(path: Path | None = None) -> list[SubjectData]:
    """
    Loads DREAMER.mat and returns a list of SubjectData, one per subject.

    Requires scipy. DREAMER.mat is saved in an older MATLAB format that
    scipy.io.loadmat can read directly (no need for h5py/mat73 unless
    Anthropic/UWS ever re-releases it as v7.3 — check version if this fails).
    """
    from scipy.io import loadmat

    path = path or config.DREAMER_MAT_PATH
    legacy_path = config.DATA_RAW / "DREAMER.mat"
    if not path.exists() and legacy_path.exists():
        path = legacy_path

    if not path.exists():
        raise FileNotFoundError(
            f"DREAMER.mat not found at {path}\n"
            f"Download it from https://zenodo.org/records/546113 and place it there."
        )

    mat = loadmat(str(path), simplify_cells=True)
    dreamer = mat["DREAMER"]

    subjects = []
    for i, subj in enumerate(dreamer["Data"]):
        eeg_baseline = [np.asarray(clip, dtype=np.float32) for clip in subj["EEG"]["baseline"]]
        eeg_stimuli = [np.asarray(clip, dtype=np.float32) for clip in subj["EEG"]["stimuli"]]

        subjects.append(
            SubjectData(
                subject_id=i + 1,
                age=str(subj.get("Age", "unknown")),
                gender=str(subj.get("Gender", "unknown")),
                eeg_baseline=eeg_baseline,
                eeg_stimuli=eeg_stimuli,
                valence=np.asarray(subj["ScoreValence"], dtype=np.float32),
                arousal=np.asarray(subj["ScoreArousal"], dtype=np.float32),
                dominance=np.asarray(subj["ScoreDominance"], dtype=np.float32),
            )
        )

    assert len(subjects) == config.N_SUBJECTS, (
        f"Expected {config.N_SUBJECTS} subjects, got {len(subjects)}. "
        f"Check DREAMER.mat integrity."
    )
    return subjects


def load_stew(raw_dir: Path | None = None) -> list[STEWSubjectData]:
    """
    Loads STEW raw text files.

    STEW file convention:
        sub01_lo.txt -> subject 1 at rest / low workload
        sub01_hi.txt -> subject 1 during SIMKAP multitasking / high workload
        ratings.txt  -> subject_id, rest rating, test rating

    Rows are samples and columns are the 14 Emotiv EPOC channels in
    config.EEG_CHANNELS order.
    """
    raw_dir = raw_dir or config.STEW_RAW_DIR
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"STEW raw directory not found at {raw_dir}. "
            f"Expected files like sub01_lo.txt, sub01_hi.txt, and ratings.txt."
        )

    ratings_path = raw_dir / "ratings.txt"
    ratings: dict[int, tuple[float, float]] = {}
    if ratings_path.exists():
        ratings_arr = np.loadtxt(ratings_path, delimiter=",", dtype=float)
        ratings_arr = np.atleast_2d(ratings_arr)
        for row in ratings_arr:
            subj_id = int(row[0])
            ratings[subj_id] = (float(row[1]), float(row[2]))
        expected_rating_subjects = set(range(1, config.STEW_N_SUBJECTS + 1)) - set(
            config.STEW_MISSING_RATING_SUBJECTS
        )
        if set(ratings) != expected_rating_subjects:
            raise ValueError(
                "STEW ratings.txt subject IDs do not match the documented set. "
                f"Missing={sorted(expected_rating_subjects - set(ratings))}, "
                f"unexpected={sorted(set(ratings) - expected_rating_subjects)}."
            )

    subjects = []
    for subj_id in range(1, config.STEW_N_SUBJECTS + 1):
        lo_path = raw_dir / f"sub{subj_id:02d}_lo.txt"
        hi_path = raw_dir / f"sub{subj_id:02d}_hi.txt"
        if not lo_path.exists() or not hi_path.exists():
            raise FileNotFoundError(
                f"Missing STEW file(s) for subject {subj_id}: "
                f"{lo_path.name} exists={lo_path.exists()}, {hi_path.name} exists={hi_path.exists()}"
            )

        eeg_lo = np.loadtxt(lo_path, dtype=np.float32)
        eeg_hi = np.loadtxt(hi_path, dtype=np.float32)
        if eeg_lo.ndim != 2 or eeg_hi.ndim != 2:
            raise ValueError(f"STEW subject {subj_id} files must be 2D sample x channel arrays.")
        if eeg_lo.shape[1] != config.N_CHANNELS or eeg_hi.shape[1] != config.N_CHANNELS:
            raise ValueError(
                f"STEW subject {subj_id} expected {config.N_CHANNELS} channels, "
                f"got lo={eeg_lo.shape}, hi={eeg_hi.shape}."
            )
        expected_shape = (config.STEW_SAMPLES_PER_CONDITION, config.N_CHANNELS)
        if eeg_lo.shape != expected_shape or eeg_hi.shape != expected_shape:
            raise ValueError(
                f"STEW subject {subj_id} expected each condition to have shape "
                f"{expected_shape}, got lo={eeg_lo.shape}, hi={eeg_hi.shape}."
            )

        rating_lo, rating_hi = ratings.get(subj_id, (None, None))
        subjects.append(
            STEWSubjectData(
                subject_id=subj_id,
                eeg_lo=eeg_lo,
                eeg_hi=eeg_hi,
                rating_lo=rating_lo,
                rating_hi=rating_hi,
            )
        )

    return subjects


def summarize(subjects: list[SubjectData]) -> None:
    print(f"Loaded {len(subjects)} subjects.\n")
    lengths_baseline, lengths_stimuli = [], []
    for s in subjects:
        for clip in s.eeg_baseline:
            lengths_baseline.append(clip.shape[0])
        for clip in s.eeg_stimuli:
            lengths_stimuli.append(clip.shape[0])

    print(f"EEG channels expected: {config.N_CHANNELS} ({', '.join(config.EEG_CHANNELS)})")
    print(f"Baseline clip lengths (samples): min={min(lengths_baseline)}, "
          f"max={max(lengths_baseline)}, mean={np.mean(lengths_baseline):.1f}")
    print(f"Stimuli clip lengths (samples):  min={min(lengths_stimuli)}, "
          f"max={max(lengths_stimuli)}, mean={np.mean(lengths_stimuli):.1f}")

    s0 = subjects[0]
    print(f"\nSubject 1 sanity check:")
    print(f"  age={s0.age}, gender={s0.gender}")
    print(f"  eeg_stimuli[0].shape = {s0.eeg_stimuli[0].shape}  (expect (M, 14))")
    print(f"  arousal scores = {s0.arousal}")
    print(f"  valence scores = {s0.valence}")


if __name__ == "__main__":
    subjects = load_dreamer_mat()
    summarize(subjects)
