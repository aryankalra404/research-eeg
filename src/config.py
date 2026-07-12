"""
Central configuration for the DREAMER stress-detection pipeline.
Change values here rather than scattering magic numbers across scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DREAMER_MAT_PATH = DATA_RAW / "DREAMER.mat"

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
# NOTE: NOTCH_FREQ_HZ (50) sits OUTSIDE (BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ) = (0.5, 45).
# The bandpass filter already removes everything at/above 45Hz, so mains noise
# at 50Hz is gone before the notch filter would even run -- see filter_signal()
# in preprocessing.py, which now skips the notch filter in this case instead
# of silently doing nothing. If you raise BANDPASS_HIGH_HZ above 50Hz for a
# future experiment (e.g. to keep more gamma-band content), the notch filter
# will automatically switch back on.
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