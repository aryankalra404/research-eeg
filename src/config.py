"""
Central configuration for the EEG stress-detection research pipeline.
Change values here rather than scattering magic numbers across scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = PROJECT_ROOT / "runs"

SUPPORTED_DATASETS = ("dreamer", "stew", "iub", "dasps")
DEFAULT_DATASET = "dreamer"


def normalize_dataset_name(dataset: str | None) -> str:
    name = (dataset or DEFAULT_DATASET).strip().lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: {', '.join(SUPPORTED_DATASETS)}")
    return name


def raw_dir(dataset: str | None = None) -> Path:
    return DATA_RAW / normalize_dataset_name(dataset)


def processed_dir(dataset: str | None = None) -> Path:
    return DATA_PROCESSED / normalize_dataset_name(dataset)


def model_dir(dataset: str | None = None, run_name: str | None = None) -> Path:
    path = MODELS_DIR / normalize_dataset_name(dataset)
    return path / run_name if run_name else path


def output_dir(dataset: str | None = None, run_name: str | None = None) -> Path:
    path = OUTPUTS_DIR / normalize_dataset_name(dataset)
    return path / run_name if run_name else path


def run_dir(dataset: str | None = None, run_name: str | None = None) -> Path:
    path = RUNS_DIR / normalize_dataset_name(dataset)
    return path / run_name if run_name else path


DREAMER_MAT_PATH = raw_dir("dreamer") / "DREAMER.mat"

# ---------------------------------------------------------------------------
# DREAMER dataset constants (from the official readme — do not change)
# ---------------------------------------------------------------------------
EEG_SAMPLING_RATE = 128  # Hz
ECG_SAMPLING_RATE = 256  # Hz
N_SUBJECTS = 23
N_VIDEOS = 18
EEG_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
    "P8", "T8", "FC6", "F4", "F8", "AF4",
]
N_CHANNELS = len(EEG_CHANNELS)  # 14

# ---------------------------------------------------------------------------
# Preprocessing decisions (Phase 1)
# ---------------------------------------------------------------------------
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 45.0
NOTCH_FREQ_HZ = 50.0  # DREAMER collected in UK (UWS) -> mains is 50Hz, not 60Hz
FILTER_ORDER = 4

# Epoching
WINDOW_SECONDS = 4.0  # fixed window length fed to all models + the GAN
WINDOW_OVERLAP = 0.5  # 50% overlap between consecutive windows (0.0 = no overlap)
WINDOW_SAMPLES = int(WINDOW_SECONDS * EEG_SAMPLING_RATE)  # 512 samples per window

# Baseline correction: subtract mean baseline signal (per-channel) from stimuli
APPLY_BASELINE_CORRECTION = True

# Artifact rejection (amplitude-based, applied per-window after epoching,
# before normalization). Uses a per-subject adaptive threshold rather than a
# fixed microvolt cutoff, since DREAMER's raw units/scale aren't documented
# in the readme -- median + k*MAD is robust to that ambiguity.
APPLY_ARTIFACT_REJECTION = True
ARTIFACT_REJECTION_MAD_MULTIPLIER = 30.0  # Chosen via src/diagnose_artifacts.py on real
                                          # DREAMER data: 3.0->50.0%, 5.0->41.7%, 8.0->31.1%,
                                          # 10.0->27.1%, 15.0->18.4%, 20.0->12.4%, 30.0->4.1%
                                          # rejected. 30.0 lands in the typical 2-10% range
                                          # for amplitude-based EEG artifact rejection.
                                          # The steep curve reflects checking 14 channels
                                          # independently (any-channel-fails = reject),
                                          # which compounds false positives fast at
                                          # aggressive thresholds.
ARTIFACT_REJECTION_MIN_WINDOWS_WARN = 20  # warn if a subject drops below this many
                                           # windows after rejection (too aggressive / bad data)

# Normalization
NORMALIZATION = "per_subject_zscore"  # applied per-channel, per-subject

# ---------------------------------------------------------------------------
# Labeling decisions (Phase 0)
# ---------------------------------------------------------------------------
# Stress proxy: high arousal + low valence (circumplex quadrant model)
# Threshold method: 'median_split' (data-driven, per-subject) or 'fixed' (scale midpoint)
LABEL_THRESHOLD_METHOD = "median_split"
FIXED_AROUSAL_THRESHOLD = 3  # only used if LABEL_THRESHOLD_METHOD == "fixed" (scale is 1-5)
FIXED_VALENCE_THRESHOLD = 3

LABEL_MODE = "binary"  # 'binary' (stress vs non-stress) for v1. 'multiclass' reserved for later.

# ---------------------------------------------------------------------------
# Train/test split protocol (Phase 0) — LOCKED for consistency across
# baseline (Phase 2) and augmented (Phase 5) experiments
# ---------------------------------------------------------------------------
SPLIT_PROTOCOL = "subject_independent"  # grouped k-fold by subject ID
N_FOLDS = 5
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Feature extraction (for CWGAN-GP Path A + classical baselines)
# ---------------------------------------------------------------------------
FREQ_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}
FEATURE_TYPE = "differential_entropy"  # 'differential_entropy' or 'band_power'

# ---------------------------------------------------------------------------
# Device (hardware-agnostic — works on Mac M4 / Windows+CUDA interchangeably)
# ---------------------------------------------------------------------------
def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
