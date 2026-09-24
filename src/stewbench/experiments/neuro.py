"""Dataset validation: does STEW show the expected workload signatures?

For each subject, mean log band power per channel is computed separately for
rest and SIMKAP windows (artifact-rejected, band-passed). Paired t-tests over
the 48 subjects (task - rest), Benjamini-Hochberg FDR across channel x band,
and Cohen's d_z effect sizes. Literature expectation (Gevins & Smith 2003;
Klimesch 1999): frontal-midline theta increases and parietal/occipital
alpha decreases with workload.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

from .. import constants as C
from ..data.preprocess import load_windows
from ..features import band_powers
from ..settings import ExperimentConfig
from ..utils import write_json


def fdr_bh(p: np.ndarray) -> np.ndarray:
    flat = p.ravel()
    order = np.argsort(flat)
    ranked = flat[order] * len(flat) / (np.arange(len(flat)) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty_like(flat)
    out[order] = adjusted
    return out.reshape(p.shape)


def run_neuro(config: ExperimentConfig, out_dir: Path, raw_dir=None, fixture: bool = False) -> dict:
    windows = load_windows(config.preprocess, raw_dir=raw_dir, strict=not fixture,
                           cache_dir=Path(out_dir) / "cache" if fixture else None)
    powers, bands = band_powers(windows.X)
    log_power = np.log10(powers + 1e-12)  # (N, C, B)
    subjects = windows.subjects
    per_condition = np.zeros((len(subjects), 2, C.N_CHANNELS, len(bands)))
    for i, sid in enumerate(subjects):
        for c in (0, 1):
            per_condition[i, c] = log_power[(windows.subject == sid) & (windows.y == c)].mean(0)
    theta = bands.index("theta")
    alpha = bands.index("alpha")
    ratio = per_condition[..., theta] - per_condition[..., alpha]  # log10(theta/alpha)

    diff = per_condition[:, 1] - per_condition[:, 0]  # (S, C, B)
    t, p = stats.ttest_rel(per_condition[:, 1], per_condition[:, 0], axis=0)
    dz = diff.mean(0) / diff.std(0, ddof=1)
    q = fdr_bh(p)
    t_ratio, p_ratio = stats.ttest_rel(ratio[:, 1], ratio[:, 0], axis=0)
    result = {
        "n_subjects": int(len(subjects)), "channels": list(C.CHANNELS), "bands": bands,
        "t": t.tolist(), "p": p.tolist(), "q_fdr": q.tolist(), "cohens_dz": dz.tolist(),
        "mean_log10_power": {"rest": per_condition[:, 0].mean(0).tolist(),
                             "task": per_condition[:, 1].mean(0).tolist()},
        "theta_alpha_log_ratio": {"t": t_ratio.tolist(), "p": p_ratio.tolist(),
                                  "q_fdr": fdr_bh(p_ratio).tolist(),
                                  "rest_mean": ratio[:, 0].mean(0).tolist(),
                                  "task_mean": ratio[:, 1].mean(0).tolist()},
        "rejection": windows.rejection,
        "synthetic_fixture_not_real_data": fixture,
    }
    write_json(Path(out_dir) / "neurophysiology.json", result)
    return result
