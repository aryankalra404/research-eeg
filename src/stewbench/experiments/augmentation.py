"""Augmentation study: does synthetic / transformed data improve
cross-subject workload decoding, and at which training-set sizes?

Factors: classifier x augmentation method x training-subject budget x seed,
all on the identical subject-independent folds. Generative models are
trained inside each fold on the inner-training subjects only; their samples
are cached per (budget, seed, fold) and shared by every classifier.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml

from ..augment.quality import evaluate_quality, psd_summary
from ..augment.transforms import GENERATIVE_AUGMENTATIONS, ONLINE_AUGMENTATIONS, make_online
from ..data.inputs import window_zscore
from ..data.preprocess import load_windows
from ..evaluation.metrics import evaluate
from ..models import spec
from ..provenance import provenance
from ..settings import AugmentationConfig, ExperimentConfig
from ..splits import make_folds, masks
from ..utils import get_device, write_json
from .common import prepare, run_fold


def label(aug: AugmentationConfig) -> str:
    if aug.method == "none":
        return "none"
    if aug.method in GENERATIVE_AUGMENTATIONS:
        return f"{aug.method}_frac{aug.synthetic_fraction:g}"
    return f"{aug.method}_p{aug.probability:g}_m{aug.magnitude:g}"


def _generator(method: str, n_channels: int, n_times: int, device, seed: int):
    if method == "cwgan_gp":
        from ..augment.cwgan import CWGANGP
        return CWGANGP(n_channels, n_times, device, seed=seed)
    if method == "ddpm":
        from ..augment.ddpm import ConditionalDDPM
        return ConditionalDDPM(n_channels, n_times, device, seed=seed)
    raise KeyError(method)


def synthetic_for_fold(method, max_fraction, epochs, batch_size, X_train, y_train, X_val, y_val,
                       cache: Path, device, seed, normalize: bool, verbose=False):
    """Train (or load) a generator on inner-train windows and return synthetic
    windows: ``max_fraction`` x the real count of each class (nested prefix
    subsets give smaller fractions)."""
    if cache.exists():
        with np.load(cache) as stored:
            return stored["X"], stored["y"]
    counts = {c: int(round(max_fraction * np.sum(y_train == c))) for c in (0, 1)}
    model = _generator(method, X_train.shape[1], X_train.shape[2], device, seed)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    X_synth, y_synth = model.sample(counts, seed=seed)
    if normalize:
        X_synth = window_zscore(X_synth)
    order = np.random.default_rng(seed).permutation(len(y_synth))
    X_synth, y_synth = X_synth[order].astype(np.float32), y_synth[order]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X_synth, y=y_synth)
    quality = evaluate_quality(X_train, y_train, X_synth, y_synth, X_val, y_val, seed=seed)
    quality["psd_real"] = psd_summary(X_train, y_train)
    quality["psd_synthetic"] = psd_summary(X_synth, y_synth)
    write_json(cache.with_suffix(".quality.json"), {"quality": quality, "history": model.history})
    return X_synth, y_synth


def _take_fraction(X, y, y_train, fraction):
    keep = []
    for c in (0, 1):
        need = int(round(fraction * np.sum(y_train == c)))
        keep.extend(np.flatnonzero(y == c)[:need])
    keep = np.sort(np.asarray(keep, dtype=int))
    return X[keep], y[keep]


def run_augmentation(config: ExperimentConfig, out_dir: Path, device=None, raw_dir=None,
                     verbose: bool = False, fixture: bool = False) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(device)
    if config.num_threads:
        torch.set_num_threads(config.num_threads)
    for name in config.models:
        if spec(name).kind != "deep":
            raise ValueError(f"Augmentation study needs deep classifiers, got {name}")
    augs = list(config.augmentations)
    if not any(a.method == "none" for a in augs):
        augs.insert(0, AugmentationConfig(method="none"))
    for a in augs:
        if a.method not in ONLINE_AUGMENTATIONS and a.method not in GENERATIVE_AUGMENTATIONS and a.method != "none":
            raise ValueError(f"Unknown augmentation {a.method}")

    windows = load_windows(config.preprocess, raw_dir=raw_dir, strict=not fixture,
                           cache_dir=out_dir / "cache" if fixture else None)
    data = prepare(windows, config)
    normalize_synthetic = config.input.normalization == "window_zscore" and config.input.alignment == "none"
    with open(out_dir / "config.yaml", "w") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
    write_json(out_dir / "manifest.json", {"experiment": config.name, "provenance": provenance(raw_dir),
                                           "synthetic_fixture_not_real_data": fixture,
                                           "augmentations": [dataclasses.asdict(a) | {"label": label(a)} for a in augs]})

    for budget in config.train_subject_budgets:
        protocol = dataclasses.replace(config.protocol, max_train_subjects=budget)
        folds = make_folds(windows.subject, protocol, protocol.fold_seed)
        budget_tag = "all" if budget is None else f"{budget}"
        for seed in config.seeds:
            todo = [(a, m) for a in augs for m in config.models
                    if not (out_dir / "runs" / f"budget_{budget_tag}" / label(a) / m / f"seed_{seed}.json").exists()]
            if not todo:
                continue
            print(f"\n=== budget={budget_tag} train subjects, seed={seed}: {len(todo)} runs ===")
            p1 = {(label(a), m): np.full(len(data.y), np.nan) for a, m in todo}
            info = {(label(a), m): [] for a, m in todo}
            for fold in folds:
                train, val, _ = masks(data.subject, fold)
                synthetic = {}
                for method in {a.method for a, _ in todo if a.method in GENERATIVE_AUGMENTATIONS}:
                    relevant = [a for a in augs if a.method == method]
                    first = relevant[0]
                    cache = (out_dir / "synthetic" / f"budget_{budget_tag}" /
                             f"{method}_seed{seed}_fold{fold.index}.npz")
                    print(f"  fold {fold.index + 1}: {method} generator ({'cached' if cache.exists() else 'training'})")
                    synthetic[method] = synthetic_for_fold(
                        method, max(a.synthetic_fraction for a in relevant), first.generator_epochs,
                        first.generator_batch_size, data.X_input[train], data.y[train],
                        data.X_input[val], data.y[val], cache, device, seed, normalize_synthetic, verbose)
                for a, model_name in todo:
                    augment, X_extra, y_extra = None, None, None
                    if a.method in ONLINE_AUGMENTATIONS:
                        augment = make_online(a.method, a.probability, a.magnitude)
                    elif a.method in GENERATIVE_AUGMENTATIONS:
                        X_extra, y_extra = _take_fraction(*synthetic[a.method], data.y[train], a.synthetic_fraction)
                    out = run_fold(model_name, data, fold, seed, config, device, augment=augment,
                                   X_extra=X_extra, y_extra=y_extra, verbose=verbose)
                    p1[(label(a), model_name)][out.test_index] = out.p1
                    info[(label(a), model_name)].append({
                        **fold.as_dict(), "best_epoch": out.best_epoch, "epochs_run": out.epochs_run,
                        "train_seconds": out.train_seconds,
                        "n_synthetic": 0 if X_extra is None else int(len(X_extra))})
                    acc = np.mean((out.p1 >= 0.5) == data.y[out.test_index])
                    print(f"    fold {fold.index + 1} {label(a):28s} {model_name:14s} acc={acc:.3f}")
            for (aug_label, model_name), values in p1.items():
                tested = ~np.isnan(values)
                result = {
                    "budget": budget, "augmentation": aug_label, "model": model_name, "seed": seed,
                    "metrics": evaluate(data.y[tested], values[tested], data.subject[tested], seed=seed),
                    "folds": info[(aug_label, model_name)],
                    "predictions": {"index": np.flatnonzero(tested), "p1": values[tested],
                                    "y": data.y[tested], "subject": data.subject[tested]},
                }
                write_json(out_dir / "runs" / f"budget_{budget_tag}" / aug_label / model_name / f"seed_{seed}.json",
                           result)
    return out_dir
