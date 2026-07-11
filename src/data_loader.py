"""
Loads DREAMER.mat and converts it into a clean, subject-indexed Python structure.

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

from dataclasses import dataclass, field
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


def load_dreamer_mat(path: Path = config.DREAMER_MAT_PATH) -> list[SubjectData]:
    """
    Loads DREAMER.mat and returns a list of SubjectData, one per subject.

    Requires scipy. DREAMER.mat is saved in an older MATLAB format that
    scipy.io.loadmat can read directly (no need for h5py/mat73 unless
    Anthropic/UWS ever re-releases it as v7.3 — check version if this fails).
    """
    from scipy.io import loadmat

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
