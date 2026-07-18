"""Central configuration for the multi-dataset EEG research pipeline."""

from dataclasses import asdict, dataclass
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

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    implemented: bool
    task: str
    sampling_rate_hz: int | None
    channels: tuple[str, ...]
    n_subjects: int | None
    class_names: tuple[str, str] | None


EMOTIV_14_CHANNELS = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
    "P8", "T8", "FC6", "F4", "F8", "AF4",
)

DATASET_SPECS = {
    "dreamer": DatasetSpec(
        name="dreamer",
        implemented=True,
        task="emotion-derived stress proxy classification",
        sampling_rate_hz=128,
        channels=EMOTIV_14_CHANNELS,
        n_subjects=23,
        class_names=("non-stress proxy", "stress proxy"),
    ),
    "stew": DatasetSpec(
        name="stew",
        implemented=True,
        task="low versus high workload classification",
        sampling_rate_hz=128,
        channels=EMOTIV_14_CHANNELS,
        n_subjects=48,
        class_names=("low workload/rest", "high workload/multitasking"),
    ),
    "iub": DatasetSpec("iub", False, "not configured", None, (), None, None),
    "dasps": DatasetSpec("dasps", False, "not configured", None, (), None, None),
}

SUPPORTED_DATASETS = tuple(name for name, spec in DATASET_SPECS.items() if spec.implemented)
PLANNED_DATASETS = tuple(name for name, spec in DATASET_SPECS.items() if not spec.implemented)
DEFAULT_DATASET = "stew"


def normalize_dataset_name(dataset: str | None) -> str:
    name = (dataset or DEFAULT_DATASET).strip().lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: {', '.join(SUPPORTED_DATASETS)}")
    return name


def dataset_spec(dataset: str | None = None) -> DatasetSpec:
    return DATASET_SPECS[normalize_dataset_name(dataset)]


def dataset_spec_dict(dataset: str | None = None) -> dict:
    return asdict(dataset_spec(dataset))


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
STEW_RAW_DIR = raw_dir("stew") / "stew_dataset"

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
EEG_SAMPLING_RATE = dataset_spec(DEFAULT_DATASET).sampling_rate_hz  # legacy shared alias
ECG_SAMPLING_RATE = 256  # Hz
N_SUBJECTS = DATASET_SPECS["dreamer"].n_subjects
N_VIDEOS = 18
EEG_CHANNELS = list(EMOTIV_14_CHANNELS)
N_CHANNELS = len(EEG_CHANNELS)  # 14

STEW_N_SUBJECTS = DATASET_SPECS["stew"].n_subjects
STEW_SAMPLES_PER_CONDITION = 19_200
STEW_MISSING_RATING_SUBJECTS = (5, 24, 42)

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
ARTIFACT_REJECTION_MAD_MULTIPLIER = 30.0  # DREAMER setting, chosen via diagnostics on real
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

# STEW is configured separately because its raw scale and recording protocol
# differ from DREAMER. On the fixed training subjects only, a multiplier of 60
# rejected 4.73% of 5,624 candidate windows (lo=3.06%, hi=6.40%). Held-out test
# subjects were not used to select this preprocessing setting.
STEW_ARTIFACT_REJECTION_MAD_MULTIPLIER = 60.0
STEW_NORMALIZATION = "per_window_channel_zscore"


def artifact_rejection_mad_multiplier(dataset: str | None = None) -> float:
    return (
        STEW_ARTIFACT_REJECTION_MAD_MULTIPLIER
        if normalize_dataset_name(dataset) == "stew"
        else ARTIFACT_REJECTION_MAD_MULTIPLIER
    )


def class_names(dataset: str | None = None) -> tuple[str, str]:
    names = dataset_spec(dataset).class_names
    if names is None:
        raise ValueError(f"Dataset {dataset} does not define binary class names.")
    return names

# ---------------------------------------------------------------------------
# Labeling decisions (Phase 0)
# ---------------------------------------------------------------------------
# Stress proxy: high arousal + low valence (circumplex quadrant model)
# Threshold method: 'median_split' (data-driven, per-subject) or 'fixed' (scale midpoint)
LABEL_THRESHOLD_METHOD = "median_split"
FIXED_AROUSAL_THRESHOLD = 3  # only used if LABEL_THRESHOLD_METHOD == "fixed" (scale is 1-5)
FIXED_VALENCE_THRESHOLD = 3

LABEL_MODE = "binary"  # Dataset-specific binary class names come from class_names().

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
