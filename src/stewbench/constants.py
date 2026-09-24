"""Fixed facts about the STEW dataset and repository layout.

STEW (Lim, Sourina & Wang, IEEE TNSRE 2018): 48 subjects, Emotiv EPOC,
14 channels, 128 Hz, 2.5 min at rest ("lo") and 2.5 min during the SIMKAP
multitasking test ("hi"). Self-rated workload (1-9) is missing for subjects
5, 24 and 42.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "stew"
PROCESSED_DIR = DATA_DIR / "processed" / "stew"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "stew"
CONFIGS_DIR = PROJECT_ROOT / "configs"

SFREQ = 128
N_SUBJECTS = 48
SAMPLES_PER_RECORDING = 19_200  # 150 s * 128 Hz
MISSING_RATING_SUBJECTS = (5, 24, 42)

CHANNELS = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)
N_CHANNELS = len(CHANNELS)

# Left/right homologous pairs (same index = mirror electrode). Used by
# TSception's hemispheric kernels and by asymmetry features.
LEFT_HEMISPHERE = ("AF3", "F7", "F3", "FC5", "T7", "P7", "O1")
RIGHT_HEMISPHERE = ("AF4", "F8", "F4", "FC6", "T8", "P8", "O2")

# MNE standard_1020 head coordinates (metres) in CHANNELS order.
CHANNEL_POSITIONS = (
    (-0.033701, 0.076837, 0.021227),
    (-0.070263, 0.042474, -0.011420),
    (-0.050244, 0.053111, 0.042192),
    (-0.077215, 0.018643, 0.024460),
    (-0.084161, -0.016019, -0.009346),
    (-0.072434, -0.073453, -0.002487),
    (-0.029413, -0.112449, 0.008839),
    (0.029843, -0.112156, 0.008800),
    (0.073056, -0.073068, -0.002540),
    (0.085080, -0.015020, -0.009490),
    (0.079534, 0.019936, 0.024438),
    (0.051836, 0.054305, 0.040814),
    (0.073043, 0.044422, -0.012000),
    (0.035712, 0.077726, 0.021956),
)

CLASS_NAMES = ("rest (low workload)", "SIMKAP multitasking (high workload)")

FREQ_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
