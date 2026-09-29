"""Model E: artifact-counterfactual consistency training (ACCT) vs. controls.

Conditions (same folds, seeds and recipe):
    erm              plain training
    channel_dropout  strongest simple augmentation from the augmentation study
    cf_aug           artifact counterfactuals, cross-entropy only
    acct             artifact counterfactuals + consistency loss (Model E)

Beyond clean accuracy it measures artifact reliance: accuracy with
F7/F8/T7/T8 zeroed at test time, and accuracy / prediction flips under
injected artifacts at several magnitudes.

``dev_mode`` evaluates on each fold's inner-validation subjects and carves a
new validation pool from the training subjects; test subjects are never
touched, so design decisions cannot leak test information.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml

from .. import constants as C
from ..artifacts import ArtifactCounterfactual, fit_artifact_bank
from ..augment.transforms import make_online
from ..data.inputs import model_inputs
from ..data.preprocess import load_windows
from ..evaluation.metrics import evaluate
from ..evaluation.stats import holm, paired_comparison
from ..provenance import provenance
from ..settings import ExperimentConfig
from ..splits import Fold, make_folds, masks
from ..training import predict_proba
from ..utils import get_device, read_json, write_json
from .common import prepare, run_fold

CONDITIONS = ("erm", "channel_dropout", "cf_aug", "acct", "acct_cd")


def parse_condition(condition: str, default_weight: float) -> tuple[str, float]:
    """``acct@3`` -> ("acct", 3.0). The weight suffix applies to acct variants."""
    base, _, weight = condition.partition("@")
    if base not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition}")
    return base, float(weight) if weight else default_weight


def ocular_proxy(X_filtered: np.ndarray, sfreq: int = C.SFREQ) -> np.ndarray:
    """Natural ocular-activity score per window: mean log 0.5-4 Hz power over
    the frontal pole/lateral frontal sites (AF3, AF4, F7, F8). Computed from
    the recorded data only; no synthetic artifacts involved."""
    from scipy.signal import welch
    idx = [C.CHANNELS.index(ch) for ch in ("AF3", "AF4", "F7", "F8")]
    freqs, psd = welch(X_filtered[:, idx], fs=sfreq, nperseg=min(256, X_filtered.shape[-1]), axis=-1)
    band = (freqs >= 0.5) & (freqs <= 4.0)
    return np.log(psd[..., band].mean(-1) + 1e-12).mean(-1)


def artifact_strata(y, subject, proxy, quantile: float = 0.25):
    """Masks of the most / least ocular-laden windows within every
    (subject, class) cell, so both strata keep both classes of every subject."""
    high = np.zeros(len(y), dtype=bool)
    low = np.zeros(len(y), dtype=bool)
    for sid in np.unique(subject):
        for c in (0, 1):
            cell = np.flatnonzero((subject == sid) & (y == c))
            if len(cell) < 4:
                continue
            values = proxy[cell]
            high[cell[values >= np.quantile(values, 1 - quantile)]] = True
            low[cell[values <= np.quantile(values, quantile)]] = True
    return high, low


def dev_fold(fold: Fold, fraction: float, seed: int) -> Fold:
    """Evaluate on the fold's validation subjects; new validation from training."""
    rng = np.random.default_rng(seed * 7919 + fold.index)
    train = rng.permutation(np.array(fold.train_subjects))
    n_val = max(1, int(round(fraction * len(train))))
    return Fold(index=fold.index, train_subjects=tuple(sorted(int(s) for s in train[n_val:])),
                val_subjects=tuple(sorted(int(s) for s in train[:n_val])),
                test_subjects=fold.val_subjects)


def _bacc(y, p1) -> float:
    pred = p1 >= 0.5
    return float(np.mean([np.mean(pred[y == c] == c) for c in (0, 1)]))


def run_robust(config: ExperimentConfig, out_dir: Path, device=None, raw_dir=None,
               fixture: bool = False, verbose: bool = False) -> Path:
    rc = config.robust
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(device)
    windows = load_windows(config.preprocess, raw_dir=raw_dir, strict=not fixture,
                           cache_dir=out_dir / "cache" if fixture else None)
    data = prepare(windows, config)
    folds = make_folds(windows.subject, config.protocol, config.protocol.fold_seed)
    if rc.dev_mode:
        folds = [dev_fold(f, config.protocol.inner_val_fraction, config.protocol.fold_seed) for f in folds]
    for condition in rc.conditions:
        parse_condition(condition, rc.consistency_weight)
    proxy = ocular_proxy(windows.X)
    ablate = [C.CHANNELS.index(ch) for ch in rc.ablate_channels]
    X_ablated_filtered = windows.X.copy()
    X_ablated_filtered[:, ablate] = 0.0
    X_ablated = model_inputs(X_ablated_filtered, windows.subject, config.input)

    with open(out_dir / "config.yaml", "w") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
    write_json(out_dir / "manifest.json", {"experiment": config.name, "dev_mode": rc.dev_mode,
                                           "provenance": provenance(raw_dir),
                                           "synthetic_fixture_not_real_data": fixture,
                                           "folds": [f.as_dict() for f in folds]})

    for model_name in config.models:
        for condition in rc.conditions:
            for seed in config.seeds:
                path = out_dir / "runs" / model_name / condition / f"seed_{seed}.json"
                if path.exists():
                    continue
                print(f"\n=== {model_name} / {condition} / seed {seed} ===")
                n = len(data.y)
                p_clean, p_ablated = np.full(n, np.nan), np.full(n, np.nan)
                p_stress = {m: np.full(n, np.nan) for m in rc.stress_magnitudes}
                fold_info = []
                for fold in folds:
                    train, _, test = masks(data.subject, fold)
                    bank = fit_artifact_bank(data.X_input[train], n_ocular=rc.n_ocular, seed=seed)
                    cf = ArtifactCounterfactual(bank, rc.p_add_ocular, rc.p_remove_ocular, rc.p_emg, rc.magnitude)
                    augment, train_kwargs = None, {}
                    base, weight = parse_condition(condition, rc.consistency_weight)
                    if base in ("channel_dropout", "acct_cd"):
                        augment = make_online("channel_dropout", 0.5, 0.5)
                    if base == "cf_aug":
                        train_kwargs = {"counterfactual": cf, "consistency_weight": 0.0}
                    elif base in ("acct", "acct_cd"):
                        train_kwargs = {"counterfactual": cf, "consistency_weight": weight}
                    out = run_fold(model_name, data, fold, seed, config, device, augment=augment,
                                   train_kwargs=train_kwargs, verbose=verbose)
                    amp = config.training_for(model_name).amp
                    p_clean[out.test_index] = out.p1
                    p_ablated[out.test_index] = predict_proba(out.model, X_ablated[test], device, amp=amp)[:, 1]
                    X_test = torch.as_tensor(data.X_input[test])
                    for magnitude in rc.stress_magnitudes:
                        # Stress: always add ocular + EMG artifacts (bank from TRAINING subjects).
                        stress = ArtifactCounterfactual(bank, 1.0, 0.0, 1.0, magnitude)
                        g = torch.Generator().manual_seed(10_000 + seed * 100 + fold.index)
                        X_stressed = stress(X_test, g, magnitude).numpy()
                        p_stress[magnitude][out.test_index] = predict_proba(out.model, X_stressed, device,
                                                                            amp=amp)[:, 1]
                    y_test = data.y[out.test_index]
                    fold_info.append({**fold.as_dict(), "best_epoch": out.best_epoch,
                                      "train_seconds": out.train_seconds,
                                      "bacc_clean": _bacc(y_test, out.p1),
                                      "bacc_ablated": _bacc(y_test, p_ablated[out.test_index]),
                                      "artifact_bank": bank.summary() if fold.index == 0 else None})
                    print(f"  fold {fold.index + 1}: clean={fold_info[-1]['bacc_clean']:.3f} "
                          f"ablated={fold_info[-1]['bacc_ablated']:.3f} ({out.train_seconds:.1f}s)")
                tested = ~np.isnan(p_clean)
                y, s = data.y[tested], data.subject[tested]
                result = {
                    "model": model_name, "condition": condition, "seed": seed, "dev_mode": rc.dev_mode,
                    "metrics": evaluate(y, p_clean[tested], s, seed=seed),
                    "ablated_metrics": evaluate(y, p_ablated[tested], s, seed=seed, n_boot=0),
                    "stress": {str(m): {"balanced_accuracy": _bacc(y, p_stress[m][tested]),
                                        "flip_rate": float(np.mean((p_stress[m][tested] >= 0.5)
                                                                   != (p_clean[tested] >= 0.5))),
                                        "mean_abs_shift": float(np.mean(np.abs(p_stress[m][tested]
                                                                               - p_clean[tested])))}
                               for m in rc.stress_magnitudes},
                    "folds": fold_info,
                    "predictions": {"index": np.flatnonzero(tested), "y": y, "subject": s,
                                    "ocular_proxy": proxy[tested],
                                    "p_clean": p_clean[tested], "p_ablated": p_ablated[tested]},
                }
                write_json(path, result)
                w, a = result["metrics"]["window"], result["ablated_metrics"]["window"]
                print(f"  => clean bacc={w['balanced_accuracy']:.3f} ablated={a['balanced_accuracy']:.3f} "
                      f"stress@{rc.stress_magnitudes[-1]}={result['stress'][str(rc.stress_magnitudes[-1])]['balanced_accuracy']:.3f}")
    robust_report(out_dir)
    return out_dir


def robust_report(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    manifest = read_json(out_dir / "manifest.json")
    results: dict[tuple[str, str], list[dict]] = {}
    for path in sorted((out_dir / "runs").glob("*/*/seed_*.json")):
        r = read_json(path)
        results.setdefault((r["model"], r["condition"]), []).append(r)
    if not results:
        raise FileNotFoundError("No Model E runs found")

    def per_subject(rs, key):
        values = {}
        for r in rs:
            p, y, s = (np.asarray(r["predictions"][k]) for k in (key, "y", "subject"))
            for sid in np.unique(s):
                m = s == sid
                values.setdefault(int(sid), []).append(_bacc(y[m], p[m]))
        return {sid: float(np.mean(v)) for sid, v in values.items()}

    rows, tests = [], []
    def order(kv):
        base, _, weight = kv[0][1].partition("@")
        return kv[0][0], CONDITIONS.index(base) if base in CONDITIONS else 99, weight
    for (model, condition), rs in sorted(results.items(), key=order):
        clean = [r["metrics"]["window"]["balanced_accuracy"] for r in rs]
        ablated = [r["ablated_metrics"]["window"]["balanced_accuracy"] for r in rs]
        auc = [r["metrics"]["window"]["roc_auc"] for r in rs]
        ece = [r["metrics"]["window"]["ece"] for r in rs]
        mags = sorted(rs[0]["stress"], key=float)
        gaps, bacc_high = [], []
        for r in rs:
            pred = r["predictions"]
            if "ocular_proxy" not in pred:
                continue
            y, sub, p1 = (np.asarray(pred[k]) for k in ("y", "subject", "p_clean"))
            high, low = artifact_strata(y, sub, np.asarray(pred["ocular_proxy"]))
            bacc_high.append(_bacc(y[high], p1[high]))
            gaps.append(_bacc(y[low], p1[low]) - bacc_high[-1])
        row = {"model": model, "condition": condition, "n_seeds": len(rs),
               "bacc_high_ocular": float(np.mean(bacc_high)) if bacc_high else float("nan"),
               "natural_artifact_gap": float(np.mean(gaps)) if gaps else float("nan"),
               "bacc_clean": np.mean(clean), "bacc_clean_sd": np.std(clean, ddof=1) if len(rs) > 1 else 0.0,
               "auc": np.mean(auc), "ece": np.mean(ece),
               "bacc_ablated": np.mean(ablated), "reliance_drop": np.mean(clean) - np.mean(ablated)}
        for m in mags:
            row[f"bacc_stress_{m}"] = np.mean([r["stress"][m]["balanced_accuracy"] for r in rs])
            row[f"flip_rate_{m}"] = np.mean([r["stress"][m]["flip_rate"] for r in rs])
        rows.append(row)
        if condition != "erm" and (model, "erm") in results:
            for key, label in (("p_clean", "clean"), ("p_ablated", "ablated")):
                a, b = per_subject(rs, key), per_subject(results[(model, "erm")], key)
                common = sorted(set(a) & set(b))
                comp = paired_comparison(np.array([a[i] for i in common]), np.array([b[i] for i in common]))
                tests.append({"model": model, "condition": condition, "evaluation": label,
                              "delta_vs_erm": comp["mean_difference"], "ci95": comp["mean_difference_ci95"],
                              "p": comp["p_value"]})
    adjusted = holm({i: t["p"] for i, t in enumerate(tests)})
    for i, t in enumerate(tests):
        t["p_holm"] = adjusted[i]

    mags = sorted({k.split("_")[-1] for r in rows for k in r if k.startswith("bacc_stress_")}, key=float)
    header = ["Model", "Condition", "Clean BAcc", "AUC", "ECE", "BAcc F7/F8/T7/T8 zeroed", "Reliance drop",
              "BAcc high-ocular windows", "Natural artifact gap (low−high)"] + \
             [f"Stress×{m} BAcc / flips" for m in mags]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        cells = [r["model"], r["condition"], f"{r['bacc_clean']:.3f} ± {r['bacc_clean_sd']:.3f}",
                 f"{r['auc']:.3f}", f"{r['ece']:.3f}", f"{r['bacc_ablated']:.3f}", f"{r['reliance_drop']:+.3f}",
                 f"{r['bacc_high_ocular']:.3f}", f"{r['natural_artifact_gap']:+.3f}"]
        cells += [f"{r[f'bacc_stress_{m}']:.3f} / {r[f'flip_rate_{m}']:.3f}" for m in mags]
        lines.append("| " + " | ".join(cells) + " |")
    test_lines = ["| Model | Condition | Evaluated on | Δ BAcc vs erm [95% CI] | p (Holm) |", "|---|---|---|---|---|"]
    for t in tests:
        test_lines.append(f"| {t['model']} | {t['condition']} | {t['evaluation']} | {t['delta_vs_erm']:+.3f} "
                          f"[{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] | {t['p_holm']:.2g} |")
    banner = ("> **DEV MODE: evaluated on inner-validation subjects only (test subjects untouched).**\n\n"
              if manifest.get("dev_mode") else "")
    if manifest.get("synthetic_fixture_not_real_data"):
        banner = "> **SYNTHETIC FIXTURE - not real data.**\n\n" + banner
    report = out_dir / "REPORT.md"
    report.write_text(banner + "# Model E: artifact-counterfactual consistency training\n\n" + "\n".join(lines)
                      + "\n\nPaired per-subject Wilcoxon vs. erm (Holm):\n\n" + "\n".join(test_lines) + "\n")
    write_json(out_dir / "summary.json", {"rows": rows, "tests": tests})
    print(f"Model E report written to {report}")
    return report
