"""Publication figures. Every function takes plain data and a path, and
saves PDF + PNG."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import constants as C
from .style import (
    DIVERGING, FAMILY_COLOR, FAMILY_MARKER, FAMILY_ORDER, GRID, SEQUENTIAL, TEXT_PRIMARY,
    TEXT_SECONDARY, apply_style, save,
)

apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

LINESTYLES = ("-", "--", ":", "-.")


def _family_legend(ax, families, loc="lower right", **kwargs):
    present = [f for f in FAMILY_ORDER if f in set(families)]
    handles = [Line2D([], [], marker=FAMILY_MARKER[f], color=FAMILY_COLOR[f], linestyle="none",
                      markersize=6, label=f) for f in present]
    ax.legend(handles=handles, loc=loc, **kwargs)


def _styled_lines(names: list[str], families: dict[str, str]) -> dict[str, dict]:
    """Colour by family; models sharing a family get distinct dash patterns."""
    seen: dict[str, int] = {}
    styles = {}
    for name in names:
        family = families[name]
        styles[name] = {"color": FAMILY_COLOR[family], "linestyle": LINESTYLES[seen.get(family, 0) % 4]}
        seen[family] = seen.get(family, 0) + 1
    return styles


# ---------------------------------------------------------------------------
# Benchmark figures
# ---------------------------------------------------------------------------
def cd_diagram(mean_ranks: dict[str, float], cd: float, path: Path, title: str = "") -> None:
    """Critical-difference diagram (Demsar, 2006). Models joined by a bar are
    not significantly different (Nemenyi, alpha = 0.05)."""
    items = sorted(mean_ranks.items(), key=lambda kv: kv[1])
    names, ranks = [n for n, _ in items], np.array([r for _, r in items])
    k = len(items)
    half = (k + 1) // 2
    height = 1.2 + 0.22 * half
    fig, ax = plt.subplots(figsize=(7.0, height))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-(half + 1.5) * 0.22 - 0.2, 0.55)
    ax.axis("off")
    ax.hlines(0, 1, k, color=TEXT_PRIMARY, linewidth=1)
    for r in range(1, k + 1):
        ax.vlines(r, 0, 0.06, color=TEXT_PRIMARY, linewidth=1)
        ax.text(r, 0.1, str(r), ha="center", va="bottom", fontsize=7)
    ax.hlines(0.42, 1, 1 + cd, color=TEXT_PRIMARY, linewidth=1.5)
    ax.text(1 + cd / 2, 0.46, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
    for i, (name, rank) in enumerate(items):
        left = i < half
        row = (i if left else k - 1 - i) + 1
        y = -row * 0.22 - 0.25
        x_text = 0.55 if left else k + 0.45
        ax.plot([rank, rank, x_text], [0, y, y], color=TEXT_SECONDARY, linewidth=0.7)
        ax.text(x_text + (-0.05 if left else 0.05), y, f"{name} ({rank:.2f})", fontsize=7,
                ha="right" if left else "left", va="center")
    # Cliques: maximal runs whose rank span is below CD.
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for level, (i, j) in enumerate(cliques):
        y = -0.08 - 0.07 * (level % 4)
        ax.hlines(y, ranks[i] - 0.05, ranks[j] + 0.05, color=TEXT_PRIMARY, linewidth=3)
    if title:
        ax.set_title(title, loc="left")
    save(fig, path)


def per_subject_box(scores: pd.DataFrame, families: dict[str, str], path: Path,
                    metric_label: str = "Per-subject balanced accuracy") -> None:
    order = scores.median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(order) + 1.0))
    for i, name in enumerate(order):
        values = scores[name].dropna().to_numpy()
        colour = FAMILY_COLOR[families[name]]
        ax.boxplot(values, positions=[i], vert=False, widths=0.6, showfliers=False,
                   patch_artist=True, boxprops={"facecolor": "none", "edgecolor": colour, "linewidth": 1.2},
                   medianprops={"color": colour, "linewidth": 2}, whiskerprops={"color": colour},
                   capprops={"color": colour})
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(values))
        ax.scatter(values, i + jitter, s=6, color=colour, alpha=0.55, linewidths=0)
    ax.axvline(0.5, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.text(0.505, -0.45, "chance", color=TEXT_SECONDARY, fontsize=7, va="top")
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel(metric_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, len(order) - 0.4)
    ax.grid(axis="y", visible=False)
    _family_legend(ax, [families[n] for n in order], loc="lower left", bbox_to_anchor=(1.01, 0))
    save(fig, path)


def roc_pr(curves: dict[str, dict], families: dict[str, str], aucs: dict[str, float], path: Path) -> None:
    names = list(curves)
    styles = _styled_lines(names, families)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for name in names:
        grid = np.asarray(curves[name]["grid"])
        ax1.plot(grid, curves[name]["roc_tpr"], label=f"{name} (AUC {aucs[name]:.3f})", **styles[name])
        ax2.plot(grid, curves[name]["pr_precision"], label=name, **styles[name])
    ax1.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, linestyle=":")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (pooled test windows)",
            xlim=(0, 1), ylim=(0, 1))
    ax2.axhline(0.5, color=TEXT_SECONDARY, linewidth=1, linestyle=":")
    ax2.set(xlabel="Recall (high workload)", ylabel="Precision", title="Precision-recall", xlim=(0, 1), ylim=(0, 1))
    ax1.legend(loc="lower right", fontsize=6)
    save(fig, path)


def reliability(curves: dict[str, dict], families: dict[str, str], eces: dict[str, float], path: Path) -> None:
    names = list(curves)
    styles = _styled_lines(names, families)
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, linestyle=":")
    for name in names:
        rel = curves[name]["reliability"]
        x = [v for v in rel["mean_predicted"] if v is not None]
        y = [v for v, m in zip(rel["fraction_positive"], rel["mean_predicted"]) if m is not None]
        ax.plot(x, y, marker="o", markersize=3, label=f"{name} (ECE {eces[name]:.3f})", **styles[name])
    ax.set(xlabel="Predicted P(high workload)", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1),
           title="Calibration")
    ax.legend(fontsize=6, loc="upper left")
    save(fig, path)


def confusion_grid(matrices: dict[str, np.ndarray], path: Path, ncols: int = 5) -> None:
    names = list(matrices)
    ncols = min(ncols, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.45 * ncols, 1.5 * nrows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        m = np.asarray(matrices[name], float)
        norm = m / m.sum(1, keepdims=True)
        ax.imshow(norm, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else TEXT_PRIMARY)
        ax.set_title(name, fontsize=7)
        ax.set_xticks([0, 1], ["rest", "task"], fontsize=6)
        ax.set_yticks([0, 1], ["rest", "task"], fontsize=6)
        ax.grid(False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.supxlabel("Predicted", fontsize=8)
    fig.supylabel("True", fontsize=8)
    fig.tight_layout()
    save(fig, path)


def cost_scatter(df: pd.DataFrame, x: str, y: str, yerr: str | None, families: dict[str, str],
                 path: Path, xlabel: str, ylabel: str, logx: bool = True) -> None:
    """Performance vs. cost with direct labels; colour + marker = family."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for _, row in df.iterrows():
        if pd.isna(row[x]):
            continue
        fam = families[row["model"]]
        ax.errorbar(row[x], row[y], yerr=row[yerr] if yerr else None, fmt=FAMILY_MARKER[fam],
                    color=FAMILY_COLOR[fam], markersize=6, markeredgecolor="white", markeredgewidth=0.8,
                    elinewidth=1, capsize=0)
        ax.annotate(row["model"], (row[x], row[y]), xytext=(4, 3), textcoords="offset points", fontsize=6,
                    color=TEXT_SECONDARY)
    if logx:
        from matplotlib.ticker import NullFormatter
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(xlabel=xlabel, ylabel=ylabel)
    _family_legend(ax, [families[m] for m in df["model"]], loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, path)


def subject_heatmap(scores: pd.DataFrame, path: Path) -> None:
    """Subjects x models; rows sorted by mean difficulty."""
    ordered = scores.loc[scores.mean(1).sort_values().index, scores.mean().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(0.32 * ordered.shape[1] + 1.8, 0.14 * ordered.shape[0] + 1.2))
    image = ax.imshow(ordered.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(ordered.shape[1]), ordered.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(ordered.shape[0]), [f"S{s:02d}" for s in ordered.index], fontsize=5)
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.03, label="Balanced accuracy")
    save(fig, path)


def learning_curves(histories: dict[str, list[dict]], path: Path, ncols: int = 4) -> None:
    names = list(histories)
    ncols = min(ncols, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 1.6 * nrows), squeeze=False, sharey=True)
    for ax, name in zip(axes.ravel(), names):
        for key, colour, text in (("train_loss", "#2a78d6", "train"), ("val_loss", "#eb6834", "inner-val")):
            curves = [h[key] for h in histories[name] if h and h.get(key)]
            if not curves:
                continue
            length = max(len(c) for c in curves)
            padded = np.full((len(curves), length), np.nan)
            for i, c in enumerate(curves):
                padded[i, :len(c)] = c
            epochs = np.arange(1, length + 1)
            ax.plot(epochs, np.nanmedian(padded, 0), color=colour, linewidth=1.5, label=text)
            ax.fill_between(epochs, np.nanpercentile(padded, 25, 0), np.nanpercentile(padded, 75, 0),
                            color=colour, alpha=0.15, linewidth=0)
        ax.set_title(name, fontsize=7)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6)
    fig.supxlabel("Epoch", fontsize=8)
    fig.supylabel("Cross-entropy (median, IQR over folds)", fontsize=8)
    fig.tight_layout()
    save(fig, path)


def importance_heatmap(matrix: pd.DataFrame, path: Path, xlabel: str) -> None:
    v = np.nanmax(np.abs(matrix.to_numpy())) or 1e-3
    fig, ax = plt.subplots(figsize=(0.4 * matrix.shape[1] + 2.0, 0.22 * matrix.shape[0] + 1.0))
    image = ax.imshow(matrix.to_numpy(), cmap=DIVERGING.reversed(), vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(matrix.shape[0]), matrix.index, fontsize=6)
    ax.set_xlabel(xlabel)
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.03, label="ROC-AUC drop when removed")
    save(fig, path)


def _emotiv_info():
    import mne
    info = mne.create_info(list(C.CHANNELS), C.SFREQ, "eeg")
    info.set_montage("standard_1020")
    return info


def topomaps(values: np.ndarray, titles: list[str], path: Path, label: str,
             mask: np.ndarray | None = None, symmetric: bool = True) -> None:
    """``values``: (n_maps, 14). Channel names drawn; masked sensors = significant."""
    import mne
    info = _emotiv_info()
    n = len(values)
    fig, axes = plt.subplots(1, n, figsize=(1.55 * n + 0.6, 1.9), squeeze=False)
    v = float(np.nanmax(np.abs(values))) or 1.0
    vlim = (-v, v) if symmetric else (float(np.nanmin(values)), float(np.nanmax(values)))
    cmap = DIVERGING.reversed() if symmetric else SEQUENTIAL
    image = None
    for i, ax in enumerate(axes[0]):
        image, _ = mne.viz.plot_topomap(
            values[i], info, axes=ax, show=False, cmap=cmap, vlim=vlim, contours=0,
            names=list(C.CHANNELS), mask=None if mask is None else mask[i],
            mask_params={"marker": "o", "markerfacecolor": TEXT_PRIMARY, "markeredgecolor": "white",
                         "markersize": 3.5, "linewidth": 0}, sensors=True, sphere=(0.0, -0.017, 0.0, 0.095))
        for text in ax.texts:  # lift names clear of the sensor / significance markers
            x, y = text.get_position()
            text.set_position((x, y + 0.011))
            text.set_fontsize(4.5)
            text.set_color(TEXT_SECONDARY)
        ax.set_title(titles[i], fontsize=8)
    fig.colorbar(image, ax=axes[0].tolist(), fraction=0.025, label=label)
    save(fig, path)


# ---------------------------------------------------------------------------
# Augmentation figures
# ---------------------------------------------------------------------------
def forest(df: pd.DataFrame, path: Path, title: str) -> None:
    """``df`` columns: model, augmentation, delta, ci_low, ci_high, p_holm."""
    models = df["model"].unique().tolist()
    augs = df["augmentation"].unique().tolist()
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models) + 1.2, 0.3 * len(augs) + 1.2),
                             sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        sub = df[df["model"] == model].set_index("augmentation").reindex(augs)
        y = np.arange(len(augs))
        significant = sub["p_holm"] < 0.05
        for yi, (_, row), sig in zip(y, sub.iterrows(), significant):
            ax.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=TEXT_SECONDARY, linewidth=1.2)
            colour = "#2a78d6" if row["delta"] >= 0 else "#e34948"
            ax.plot(row["delta"], yi, marker="D" if sig else "o", markersize=5, color=colour,
                    markerfacecolor=colour if sig else "white")
        ax.axvline(0, color=TEXT_PRIMARY, linewidth=0.8)
        ax.set_title(model, fontsize=8)
        ax.set_yticks(y, augs)
        ax.invert_yaxis()
        ax.set_xlabel("Δ balanced accuracy vs. none")
    fig.suptitle(title + "  (filled diamond: Holm p < 0.05; blue = gain, red = loss)", fontsize=8, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, path)


def data_efficiency(df: pd.DataFrame, path: Path) -> None:
    """``df`` columns: model, augmentation, n_train, mean, sd."""
    models = df["model"].unique().tolist()
    augs = df["augmentation"].unique().tolist()[:8]
    # "none" is the neutral reference; augmentations take categorical slots in order.
    colours = dict(zip(augs, (TEXT_SECONDARY, "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                              "#e87ba4", "#008300", "#4a3aa7")))
    markers = dict(zip(augs, ("o", "s", "^", "v", "D", "P", "X", "h")))
    fig, axes = plt.subplots(1, len(models), figsize=(2.8 * len(models) + 1.4, 2.8), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        for aug in augs:
            sub = df[(df["model"] == model) & (df["augmentation"] == aug)].sort_values("n_train")
            ax.errorbar(sub["n_train"], sub["mean"], yerr=sub["sd"], color=colours[aug], marker=markers[aug],
                        markersize=4, capsize=0, linewidth=1.5 if aug != "none" else 2.2,
                        linestyle="--" if aug == "none" else "-", label=aug)
        ax.set_title(model, fontsize=8)
        ax.set_xlabel("Training subjects per fold")
    axes[0, 0].set_ylabel("Balanced accuracy")
    axes[0, -1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, path)


def synthetic_psd(real: dict, synthetic: dict, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4), sharey=True)
    for c, ax in enumerate(axes):
        freqs = np.asarray(real["freqs"])
        ax.plot(freqs, real[str(c)], color="#2a78d6", label="real")
        ax.plot(synthetic["freqs"], synthetic[str(c)], color="#eb6834", linestyle="--", label="synthetic")
        ax.set_title(("Rest", "SIMKAP")[c], fontsize=8)
        ax.set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("log10 PSD (z-scored input)")
    axes[0].legend()
    fig.suptitle(title, fontsize=8, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, path)
