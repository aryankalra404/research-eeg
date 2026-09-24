"""Shared figure style for publication figures (static PDF + 300 dpi PNG).

Categorical colours follow a fixed, CVD-validated order and are bound to the
model *family* (never to rank), so a family keeps its colour in every figure.
Marker shapes provide a second, colour-independent encoding.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stewbench-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
FAMILY_ORDER = ("Feature-based", "Riemannian", "Recurrent", "Generic CNN",
                "EEG CNN", "EEG Transformer", "Vision Transformer", "Graph")
FAMILY_COLOR = dict(zip(FAMILY_ORDER, CATEGORICAL))
FAMILY_MARKER = dict(zip(FAMILY_ORDER, ("o", "s", "^", "v", "D", "P", "X", "h")))

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"
NEUTRAL_MID = "#f0efec"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f7fafe", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", NEUTRAL_MID, "#f3a5a4", "#e34948", "#a32322"])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": TEXT_SECONDARY, "axes.labelcolor": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "text.color": TEXT_PRIMARY,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 2.0, "lines.markersize": 5,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })


def save(fig, path: Path) -> list[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".pdf", ".png"):
        target = path.with_suffix(suffix)
        fig.savefig(target)
        written.append(target)
    plt.close(fig)
    return written
