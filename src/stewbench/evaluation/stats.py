"""Statistical comparison of classifiers with subjects as the unit of analysis.

* Friedman test + Iman-Davenport correction and Nemenyi critical difference
  over per-subject scores (Demsar, JMLR 2006).
* Pairwise Wilcoxon signed-rank tests with Holm correction and the
  matched-pairs rank-biserial correlation as effect size.
* Paired subject-bootstrap CI of the mean difference.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats


def holm(p_values: dict) -> dict:
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    adjusted, running, m = {}, 0.0, len(ordered)
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return {k: adjusted[k] for k in p_values}


def rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Kerby, 2014), in [-1, 1]."""
    d = differences[differences != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def paired_bootstrap_ci(differences: np.ndarray, seed: int = 0, n_boot: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(differences, size=(n_boot, len(differences)), replace=True).mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test of paired scores ``a - b``."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        p = 1.0
        statistic = 0.0
    else:
        result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        p, statistic = float(result.pvalue), float(result.statistic)
    return {"mean_difference": float(d.mean()), "median_difference": float(np.median(d)),
            "statistic": statistic, "p_value": p, "rank_biserial": rank_biserial(d),
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum())}


def friedman_nemenyi(scores: np.ndarray, names: list[str], alpha: float = 0.05) -> dict:
    """``scores``: (n_subjects, n_models), higher is better."""
    n, k = scores.shape
    ranks = np.vstack([stats.rankdata(-row) for row in scores])  # rank 1 = best
    mean_ranks = ranks.mean(0)
    result = {"n_subjects": int(n), "n_models": int(k),
              "mean_ranks": dict(zip(names, map(float, mean_ranks)))}
    if k >= 3:
        chi2, p = stats.friedmanchisquare(*scores.T)
        ff = (n - 1) * chi2 / (n * (k - 1) - chi2) if n * (k - 1) != chi2 else np.inf
        result.update({
            "friedman_chi2": float(chi2), "friedman_p": float(p),
            "iman_davenport_F": float(ff),
            "iman_davenport_p": float(stats.f.sf(ff, k - 1, (k - 1) * (n - 1))),
        })
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    result["nemenyi_q_alpha"] = float(q_alpha)
    result["critical_difference"] = float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * n)))
    return result


def pairwise_wilcoxon(scores: np.ndarray, names: list[str]) -> list[dict]:
    rows = []
    for i, j in itertools.combinations(range(len(names)), 2):
        row = {"model_a": names[i], "model_b": names[j], **wilcoxon(scores[:, i], scores[:, j])}
        rows.append(row)
    adjusted = holm({(r["model_a"], r["model_b"]): r["p_value"] for r in rows})
    for r in rows:
        r["p_holm"] = adjusted[(r["model_a"], r["model_b"])]
    return rows


def paired_comparison(treatment: np.ndarray, control: np.ndarray, seed: int = 0) -> dict:
    """Treatment-minus-control on per-subject scores."""
    d = np.asarray(treatment, float) - np.asarray(control, float)
    return {**wilcoxon(treatment, control), "mean_difference_ci95": paired_bootstrap_ci(d, seed),
            "n_subjects": int(len(d))}
