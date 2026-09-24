"""Build paper-ready tables, statistics and figures from finished runs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .. import constants as C
from ..evaluation import stats
from ..evaluation.metrics import HEADLINE_METRICS
from ..models import SPECS
from ..utils import read_json, write_json
from . import figures as fig
from .tables import fmt_ci, fmt_mean_sd, write_table

FIXTURE_WARNING = ("> **WARNING: these results were produced on the SYNTHETIC TEST FIXTURE, "
                   "not on STEW. Do not report them.**\n\n")


def _mean_sd(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _mean_ci(cis):
    cis = [c for c in cis if c]
    return [float(np.mean([c[0] for c in cis])), float(np.mean([c[1] for c in cis]))] if cis else None


def _per_subject(results: list[dict], metric: str = "balanced_accuracy") -> pd.Series:
    frame = pd.DataFrame([{int(s): v[metric] for s, v in r["metrics"]["per_subject"].items()} for r in results])
    return frame.mean(0)


def load_runs(bench_dir: Path) -> dict[str, list[dict]]:
    runs = {}
    for model in SPECS:
        files = sorted((bench_dir / "runs" / model).glob("seed_*.json"))
        if files:
            runs[model] = [read_json(f) for f in files]
    return runs


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark_report(bench_dir: Path, out_dir: Path | None = None) -> Path:
    bench_dir = Path(bench_dir)
    out = Path(out_dir or bench_dir / "report")
    manifest = read_json(bench_dir / "manifest.json")
    fixture = manifest.get("synthetic_fixture_not_real_data", False)
    runs = load_runs(bench_dir)
    if not runs:
        raise FileNotFoundError(f"No finished runs under {bench_dir / 'runs'}")
    models = list(runs)
    display = {m: SPECS[m].display for m in models}
    families = {display[m]: SPECS[m].family for m in models}

    # ---- aggregate metrics over seeds ------------------------------------
    rows = []
    for m in models:
        rs = runs[m]
        row = {"model": display[m], "key": m, "family": SPECS[m].family, "n_seeds": len(rs),
               "transductive": SPECS[m].transductive}
        for metric in HEADLINE_METRICS:
            row[f"{metric}_mean"], row[f"{metric}_sd"] = _mean_sd([r["metrics"]["window"][metric] for r in rs])
        for metric in ("accuracy", "balanced_accuracy", "roc_auc", "cohen_kappa"):
            row[f"recording_{metric}_mean"], row[f"recording_{metric}_sd"] = _mean_sd(
                [r["metrics"]["recording"][metric] for r in rs])
        for metric in ("balanced_accuracy", "roc_auc", "cohen_kappa"):
            row[f"{metric}_ci95"] = _mean_ci([r["metrics"]["window"]["ci95"].get(metric) for r in rs
                                             if r["metrics"]["window"].get("ci95")])
        folds = [f for r in rs for f in r["folds"]]
        row["train_seconds_per_fold"] = float(np.mean([f["train_seconds"] for f in folds]))
        best = [f["best_epoch"] for f in folds if f.get("best_epoch")]
        row["median_best_epoch"] = float(np.median(best)) if best else None
        complexity_path = bench_dir / "runs" / m / "complexity.json"
        if complexity_path.exists():
            cx = read_json(complexity_path)
            row.update({"parameters": cx["parameters"], "mflops": (cx["flops_per_window"] or np.nan) / 1e6,
                        "latency_cpu_ms": cx["latency_ms_cpu_batch1"],
                        "latency_gpu_ms": cx.get("latency_ms_gpu_batch1")})
        rows.append(row)
    summary = pd.DataFrame(rows)

    # ---- statistics over subjects ----------------------------------------
    scores = pd.DataFrame({display[m]: _per_subject(runs[m]) for m in models}).sort_index()
    complete = scores.dropna(axis=1)
    fn = stats.friedman_nemenyi(complete.to_numpy(), list(complete.columns)) if complete.shape[1] >= 2 else {}
    summary["mean_rank"] = summary["model"].map(fn.get("mean_ranks", {}))
    summary = summary.sort_values("balanced_accuracy_mean", ascending=False).reset_index(drop=True)
    pairwise = pd.DataFrame(stats.pairwise_wilcoxon(complete.to_numpy(), list(complete.columns))) \
        if complete.shape[1] >= 2 else pd.DataFrame()
    best = summary.iloc[0]["model"]

    # ---- tables ----------------------------------------------------------
    main = pd.DataFrame({
        "Model": summary["model"] + np.where(summary["transductive"], " †", ""),
        "Family": summary["family"],
        "Acc": [fmt_mean_sd(a, b) for a, b in zip(summary["accuracy_mean"], summary["accuracy_sd"])],
        "BAcc": [fmt_mean_sd(a, b) for a, b in zip(summary["balanced_accuracy_mean"], summary["balanced_accuracy_sd"])],
        "BAcc 95% CI": [fmt_ci(c) for c in summary["balanced_accuracy_ci95"]],
        "Macro-F1": [fmt_mean_sd(a, b) for a, b in zip(summary["f1_macro_mean"], summary["f1_macro_sd"])],
        "κ": [fmt_mean_sd(a, b) for a, b in zip(summary["cohen_kappa_mean"], summary["cohen_kappa_sd"])],
        "MCC": [fmt_mean_sd(a, b) for a, b in zip(summary["mcc_mean"], summary["mcc_sd"])],
        "ROC-AUC": [fmt_mean_sd(a, b) for a, b in zip(summary["roc_auc_mean"], summary["roc_auc_sd"])],
        "ECE": [fmt_mean_sd(a, b) for a, b in zip(summary["ece_mean"], summary["ece_sd"])],
        "Rec. Acc": [fmt_mean_sd(a, b) for a, b in zip(summary["recording_accuracy_mean"], summary["recording_accuracy_sd"])],
        "Mean rank": [f"{r:.2f}" if pd.notna(r) else "--" for r in summary["mean_rank"]],
    })
    n_seeds = int(summary["n_seeds"].max())
    scheme = manifest.get("folds") and f"{len(manifest['folds'])}-fold subject-independent CV"
    write_table(main, out / "tables" / "table1_main_results",
                caption=(f"Subject-independent low- vs. high-workload classification on STEW ({scheme}, "
                         f"mean ± SD over {n_seeds} seeds; CI = subject-cluster bootstrap). "
                         "Rec. = recording level. † uses unlabelled test-subject data (transductive)."),
                label="tab:main", df_numeric=summary)
    secondary = pd.DataFrame({
        "Model": summary["model"],
        "Sensitivity": [fmt_mean_sd(a, b) for a, b in zip(summary["sensitivity_mean"], summary["sensitivity_sd"])],
        "Specificity": [fmt_mean_sd(a, b) for a, b in zip(summary["specificity_mean"], summary["specificity_sd"])],
        "Precision": [fmt_mean_sd(a, b) for a, b in zip(summary["precision_macro_mean"], summary["precision_macro_sd"])],
        "PR-AUC": [fmt_mean_sd(a, b) for a, b in zip(summary["pr_auc_mean"], summary["pr_auc_sd"])],
        "Brier": [fmt_mean_sd(a, b) for a, b in zip(summary["brier_mean"], summary["brier_sd"])],
        "Log-loss": [fmt_mean_sd(a, b) for a, b in zip(summary["log_loss_mean"], summary["log_loss_sd"])],
        "Rec. AUC": [fmt_mean_sd(a, b) for a, b in zip(summary["recording_roc_auc_mean"], summary["recording_roc_auc_sd"])],
        "AUC 95% CI": [fmt_ci(c) for c in summary["roc_auc_ci95"]],
    })
    write_table(secondary, out / "tables" / "table2_secondary_metrics",
                caption="Secondary and calibration metrics (window level unless noted).", label="tab:secondary")
    if "parameters" in summary:
        deep = summary.dropna(subset=["parameters"])
        cost = pd.DataFrame({
            "Model": deep["model"],
            "Params": [f"{int(p):,}" for p in deep["parameters"]],
            "MFLOPs/window": [f"{v:.2f}" for v in deep["mflops"]],
            "CPU latency (ms)": [f"{v:.2f}" for v in deep["latency_cpu_ms"]],
            "GPU latency (ms)": [f"{v:.2f}" if pd.notna(v) else "--" for v in deep["latency_gpu_ms"]],
            "Train time/fold (s)": [f"{v:.1f}" for v in deep["train_seconds_per_fold"]],
            "Median best epoch": [f"{v:.0f}" if pd.notna(v) else "--" for v in deep["median_best_epoch"]],
        })
        write_table(cost, out / "tables" / "table3_complexity",
                    caption="Model complexity and cost (batch size 1 latency; FLOPs exclude FFT/STFT ops).",
                    label="tab:complexity")
    if not pairwise.empty:
        vs_best = pairwise[(pairwise["model_a"] == best) | (pairwise["model_b"] == best)].copy()
        vs_best["other"] = np.where(vs_best["model_a"] == best, vs_best["model_b"], vs_best["model_a"])
        sign = np.where(vs_best["model_a"] == best, 1, -1)
        table = pd.DataFrame({
            "Compared with": vs_best["other"],
            f"Δ BAcc ({best} − other)": [f"{v:+.3f}" for v in sign * vs_best["mean_difference"]],
            "W/T/L": [f"{w}/{t}/{l}" if s > 0 else f"{l}/{t}/{w}" for w, t, l, s in
                      zip(vs_best["wins"], vs_best["ties"], vs_best["losses"], sign)],
            "r_rb": [f"{v:+.2f}" for v in sign * vs_best["rank_biserial"]],
            "p": [f"{v:.2g}" for v in vs_best["p_value"]],
            "p (Holm)": [f"{v:.2g}" for v in vs_best["p_holm"]],
        })
        write_table(table, out / "tables" / "table4_pairwise_vs_best",
                    caption=(f"Paired Wilcoxon signed-rank tests on per-subject balanced accuracy (n = "
                             f"{complete.shape[0]} subjects), Holm-corrected over all pairs; r_rb = rank-biserial."),
                    label="tab:pairwise")
        pairwise.to_csv(out / "tables" / "pairwise_wilcoxon_all.csv", index=False)
    scores.to_csv(out / "tables" / "per_subject_balanced_accuracy.csv", index_label="subject")

    # ---- figures ---------------------------------------------------------
    figdir = out / "figures"
    if fn.get("critical_difference"):
        fig.cd_diagram(fn["mean_ranks"], fn["critical_difference"], figdir / "fig_cd_diagram",
                       title=f"Friedman p = {fn.get('friedman_p', float('nan')):.2g}")
    fig.per_subject_box(complete, families, figdir / "fig_per_subject_bacc")
    fig.subject_heatmap(complete, figdir / "fig_subject_heatmap")
    top = summary.sort_values("roc_auc_mean", ascending=False)["key"].tolist()[:8]
    first = {m: runs[m][0]["metrics"] for m in top if runs[m][0]["metrics"].get("curves")}
    if first:
        fig.roc_pr({display[m]: v["curves"] for m, v in first.items()}, families,
                   {display[m]: v["window"]["roc_auc"] for m, v in first.items()}, figdir / "fig_roc_pr")
        fig.reliability({display[m]: v["curves"] for m, v in first.items()}, families,
                        {display[m]: v["window"]["ece"] for m, v in first.items()}, figdir / "fig_calibration")
    fig.confusion_grid({display[m]: np.sum([r["metrics"]["window"]["confusion_matrix"] for r in runs[m]], 0)
                        for m in summary["key"]}, figdir / "fig_confusion_matrices")
    if "parameters" in summary and summary["parameters"].notna().any():
        deep = summary.dropna(subset=["parameters"])
        fig.cost_scatter(deep, "parameters", "balanced_accuracy_mean", "balanced_accuracy_sd", families,
                         figdir / "fig_bacc_vs_parameters", "Trainable parameters (log)", "Balanced accuracy")
        fig.cost_scatter(deep, "latency_cpu_ms", "balanced_accuracy_mean", "balanced_accuracy_sd", families,
                         figdir / "fig_bacc_vs_latency", "CPU latency per window, ms (log)", "Balanced accuracy")
    histories = {display[m]: [f.get("history") for f in runs[m][0]["folds"]] for m in models
                 if SPECS[m].kind == "deep"}
    if histories:
        fig.learning_curves(histories, figdir / "fig_learning_curves")
    importance = {display[m]: pd.DataFrame([r["importance_auc_drop"] for r in runs[m]
                                            if r.get("importance_auc_drop")]).mean() for m in models}
    importance = pd.DataFrame({k: v for k, v in importance.items() if len(v)}).T
    if not importance.empty:
        channels = importance[[f"channel:{c}" for c in C.CHANNELS]].rename(columns=lambda c: c.split(":")[1])
        bands = importance[[f"band:{b}" for b in C.FREQ_BANDS]].rename(columns=lambda c: c.split(":")[1])
        order = [m for m in summary["model"] if m in importance.index]
        fig.importance_heatmap(channels.loc[order], figdir / "fig_channel_importance", "Channel (permuted)")
        fig.importance_heatmap(bands.loc[order], figdir / "fig_band_importance", "Band (removed)")
        top3 = order[:3]
        fig.topomaps(channels.loc[top3].to_numpy(), top3, figdir / "fig_channel_importance_topomap",
                     "ROC-AUC drop", symmetric=True)
        importance.to_csv(out / "tables" / "importance_auc_drop.csv", index_label="model")

    write_json(out / "statistics.json", {"friedman_nemenyi": fn, "best_model": best,
                                         "n_subjects": int(complete.shape[0]),
                                         "synthetic_fixture_not_real_data": fixture})
    lines = [FIXTURE_WARNING if fixture else "", "# STEW benchmark report\n\n",
             (out / "tables" / "table1_main_results.md").read_text(), "\n",
             f"Friedman χ² = {fn.get('friedman_chi2', float('nan')):.2f}, p = {fn.get('friedman_p', float('nan')):.2g}; "
             f"Nemenyi CD = {fn.get('critical_difference', float('nan')):.2f} ranks.\n\n"]
    for extra in ("table2_secondary_metrics", "table3_complexity", "table4_pairwise_vs_best"):
        path = out / "tables" / f"{extra}.md"
        if path.exists():
            lines += [path.read_text(), "\n"]
    lines.append("Figures: " + ", ".join(sorted(p.name for p in figdir.glob("*.pdf"))) + "\n")
    (out / "REPORT.md").write_text("".join(lines))
    print(f"Benchmark report written to {out}")
    return out


# ---------------------------------------------------------------------------
# Augmentation study
# ---------------------------------------------------------------------------
def augmentation_report(aug_dir: Path, out_dir: Path | None = None) -> Path:
    aug_dir = Path(aug_dir)
    out = Path(out_dir or aug_dir / "report")
    manifest = read_json(aug_dir / "manifest.json")
    fixture = manifest.get("synthetic_fixture_not_real_data", False)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted((aug_dir / "runs").glob("budget_*/*/*/seed_*.json")):
        r = read_json(path)
        grouped[(path.parts[-4].removeprefix("budget_"), r["augmentation"], r["model"])].append(r)
    if not grouped:
        raise FileNotFoundError(f"No augmentation runs under {aug_dir / 'runs'}")
    aug_order = [a["label"] for a in manifest["augmentations"]]

    rows, deltas = [], []
    for (budget, aug, model), rs in grouped.items():
        mean, sd = _mean_sd([r["metrics"]["window"]["balanced_accuracy"] for r in rs])
        auc_mean, auc_sd = _mean_sd([r["metrics"]["window"]["roc_auc"] for r in rs])
        kappa_mean, kappa_sd = _mean_sd([r["metrics"]["window"]["cohen_kappa"] for r in rs])
        n_train = np.mean([len(f["train_subjects"]) for r in rs for f in r["folds"]])
        rows.append({"budget": budget, "augmentation": aug, "model": SPECS[model].display, "key": model,
                     "n_train": n_train, "mean": mean, "sd": sd, "auc_mean": auc_mean, "auc_sd": auc_sd,
                     "kappa_mean": kappa_mean, "kappa_sd": kappa_sd, "n_seeds": len(rs)})
        if aug != "none" and (budget, "none", model) in grouped:
            treated = _per_subject(rs)
            control = _per_subject(grouped[(budget, "none", model)])
            common = treated.index.intersection(control.index)
            comparison = stats.paired_comparison(treated[common].to_numpy(), control[common].to_numpy())
            deltas.append({"budget": budget, "augmentation": aug, "model": SPECS[model].display,
                           "delta": comparison["mean_difference"], "ci_low": comparison["mean_difference_ci95"][0],
                           "ci_high": comparison["mean_difference_ci95"][1], "p": comparison["p_value"],
                           "r_rb": comparison["rank_biserial"], "wins": comparison["wins"],
                           "losses": comparison["losses"]})
    table = pd.DataFrame(rows)
    delta = pd.DataFrame(deltas)
    if not delta.empty:
        delta["p_holm"] = np.nan
        for _, idx in delta.groupby(["budget", "model"]).groups.items():
            adjusted = stats.holm(dict(zip(idx, delta.loc[idx, "p"])))
            delta.loc[list(adjusted), "p_holm"] = list(adjusted.values())
    rank = {a: i for i, a in enumerate(aug_order)}
    table["order"] = table["augmentation"].map(rank)
    table = table.sort_values(["budget", "model", "order"])
    merged = table.merge(delta, on=["budget", "augmentation", "model"], how="left")
    out.mkdir(parents=True, exist_ok=True)
    merged.drop(columns=["order"]).to_csv(out / "augmentation_all.csv", index=False)

    full = merged[merged["budget"] == "all"] if (merged["budget"] == "all").any() else merged
    display = pd.DataFrame({
        "Classifier": full["model"], "Augmentation": full["augmentation"],
        "BAcc": [fmt_mean_sd(a, b) for a, b in zip(full["mean"], full["sd"])],
        "ROC-AUC": [fmt_mean_sd(a, b) for a, b in zip(full["auc_mean"], full["auc_sd"])],
        "κ": [fmt_mean_sd(a, b) for a, b in zip(full["kappa_mean"], full["kappa_sd"])],
        "Δ BAcc [95% CI]": ["--" if pd.isna(d) else f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                             for d, lo, hi in zip(full["delta"], full["ci_low"], full["ci_high"])],
        "p (Holm)": ["--" if pd.isna(p) else f"{p:.2g}" for p in full["p_holm"]],
    })
    write_table(display, out / "tables" / "table5_augmentation",
                caption=("Effect of data augmentation (all training subjects). Δ = paired per-subject difference "
                         "vs. no augmentation (subject bootstrap CI); Wilcoxon signed-rank, Holm-corrected "
                         "within classifier."), label="tab:augmentation")
    figdir = out / "figures"
    if not delta.empty:
        for budget, sub in delta.groupby("budget"):
            sub = sub.assign(order=sub["augmentation"].map(rank)).sort_values("order")
            fig.forest(sub, figdir / f"fig_augmentation_forest_budget_{budget}",
                       f"Training subjects: {budget}")
    if table["budget"].nunique() > 1:
        fig.data_efficiency(table, figdir / "fig_data_efficiency")

    quality_rows = []
    for path in sorted((aug_dir / "synthetic").glob("budget_*/*.quality.json")):
        q = read_json(path)
        method = path.name.split("_seed")[0]
        for c, entry in q["quality"]["classes"].items():
            quality_rows.append({"budget": path.parent.name.removeprefix("budget_"), "generator": method,
                                 "class": int(c), **{k: v for k, v in entry.items() if not isinstance(v, dict)},
                                 **q["quality"].get("utility", {}),
                                 "generator_train_seconds": q["history"].get("train_seconds")})
    if quality_rows:
        quality = pd.DataFrame(quality_rows)
        quality.to_csv(out / "synthetic_quality_all.csv", index=False)
        agg = quality.groupby(["budget", "generator"]).mean(numeric_only=True).reset_index()
        cols = ["log_spectral_distance_db", "channel_correlation_rel_error", "feature_mmd", "density",
                "coverage", "memorization_ratio", "trtr_balanced_accuracy", "tstr_balanced_accuracy"]
        cols = [c for c in cols if c in agg]
        shown = agg[["budget", "generator", *cols]].copy()
        for c in cols:
            shown[c] = shown[c].map(lambda v: f"{v:.3f}")
        shown.columns = ["Budget", "Generator", "LSD (dB)", "Corr. err.", "MMD", "Density", "Coverage",
                         "Memorization ratio", "TRTR BAcc", "TSTR BAcc"][: len(shown.columns)]
        write_table(shown, out / "tables" / "table6_synthetic_quality",
                    caption=("Synthetic-data quality averaged over folds, seeds and classes. LSD = log-spectral "
                             "distance; TSTR/TRTR on inner-validation subjects."), label="tab:synthetic")
        for method in quality["generator"].unique():
            first = sorted((aug_dir / "synthetic").glob(f"budget_all/{method}_seed*_fold0.quality.json"))
            if first:
                q = read_json(first[0])["quality"]
                fig.synthetic_psd(q["psd_real"], q["psd_synthetic"], figdir / f"fig_psd_real_vs_{method}",
                                  f"{method}: real vs. synthetic mean spectrum (fold 1)")
    report = [FIXTURE_WARNING if fixture else "", "# Augmentation study report\n\n",
              (out / "tables" / "table5_augmentation.md").read_text()]
    if (out / "tables" / "table6_synthetic_quality.md").exists():
        report += ["\n", (out / "tables" / "table6_synthetic_quality.md").read_text()]
    (out / "REPORT.md").write_text("".join(report))
    print(f"Augmentation report written to {out}")
    return out


# ---------------------------------------------------------------------------
# Neurophysiology
# ---------------------------------------------------------------------------
def neuro_report(neuro_json: Path, out_dir: Path) -> Path:
    result = read_json(neuro_json)
    out = Path(out_dir)
    t = np.asarray(result["t"])  # (C, B)
    q = np.asarray(result["q_fdr"])
    bands = result["bands"]
    fig.topomaps(t.T, [b.capitalize() for b in bands], out / "figures" / "fig_task_vs_rest_topomaps",
                 "t (SIMKAP − rest)", mask=(q < 0.05).T)
    ratio = result["theta_alpha_log_ratio"]
    fig.topomaps(np.asarray([ratio["t"]]), ["log θ/α"], out / "figures" / "fig_theta_alpha_topomap",
                 "t (SIMKAP − rest)", mask=np.asarray([ratio["q_fdr"]]) < 0.05)
    rows = []
    dz = np.asarray(result["cohens_dz"])
    for ci, ch in enumerate(result["channels"]):
        for bi, band in enumerate(bands):
            rows.append({"Channel": ch, "Band": band, "t": f"{t[ci, bi]:+.2f}", "d_z": f"{dz[ci, bi]:+.2f}",
                         "q (FDR)": f"{q[ci, bi]:.2g}", "_q": q[ci, bi]})
    frame = pd.DataFrame(rows).sort_values("_q")
    write_table(frame.drop(columns="_q").head(20), out / "tables" / "table0_neurophysiology",
                caption=(f"Largest task-vs-rest band-power effects (paired t-test over {result['n_subjects']} "
                         "subjects, Benjamini-Hochberg FDR over 14 channels x 5 bands)."),
                label="tab:neuro", df_numeric=frame.drop(columns="_q"))
    print(f"Neurophysiology report written to {out}")
    return out
