"""Main benchmark: every model x every seed x identical subject-independent folds.

Each (model, seed) is written to ``runs/<model>/seed_<seed>.json`` as soon as
it finishes, so an interrupted benchmark resumes where it stopped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from .. import constants as C
from ..data.preprocess import load_windows
from ..evaluation.complexity import profile
from ..evaluation.metrics import evaluate
from ..models import build_deep, spec
from ..provenance import provenance
from ..settings import ExperimentConfig
from ..splits import make_folds
from ..utils import get_device, read_json, write_json
from .common import prepare, run_fold


def occlusion_importance(y, p1, occlusion_p1: dict[str, np.ndarray]) -> dict:
    from sklearn.metrics import roc_auc_score
    base = roc_auc_score(y, p1)
    return {name: float(base - roc_auc_score(y, p)) for name, p in occlusion_p1.items()}


def _balanced_accuracy(y, p1) -> float:
    pred = p1 >= 0.5
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def run_model_seed(model_name, data, folds, seed, config, device, verbose=False,
                   checkpoint_dir: Path | None = None) -> dict:
    n = len(data.y)
    p1 = np.full(n, np.nan)
    occlusion: dict[str, np.ndarray] = {}
    fold_info = []
    for fold in folds:
        out = run_fold(model_name, data, fold, seed, config, device,
                       interpret=config.interpretability, verbose=verbose)
        p1[out.test_index] = out.p1
        for name, values in out.occlusion.items():
            occlusion.setdefault(name, np.full(n, np.nan))[out.test_index] = values
        fold_info.append({**fold.as_dict(), "best_epoch": out.best_epoch, "epochs_run": out.epochs_run,
                          "train_seconds": out.train_seconds, "predict_seconds": out.predict_seconds,
                          "n_test_windows": int(len(out.test_index)), "best_params": out.best_params,
                          "history": out.history})
        if checkpoint_dir is not None and isinstance(out.model, torch.nn.Module):
            path = checkpoint_dir / model_name / f"seed{seed}_fold{fold.index}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(out.model.state_dict(), path)
        print(f"    fold {fold.index + 1}/{len(folds)} test={list(fold.test_subjects)} "
              f"bacc={_balanced_accuracy(data.y[out.test_index], out.p1):.3f} ({out.train_seconds:.1f}s)")
    tested = ~np.isnan(p1)
    metrics = evaluate(data.y[tested], p1[tested], data.subject[tested], seed=seed)
    result = {
        "model": model_name, "seed": seed, "metrics": metrics, "folds": fold_info,
        "predictions": {"index": np.flatnonzero(tested), "p1": p1[tested],
                        "y": data.y[tested], "subject": data.subject[tested]},
    }
    if occlusion:
        result["importance_auc_drop"] = occlusion_importance(data.y[tested], p1[tested],
                                                             {k: v[tested] for k, v in occlusion.items()})
    return result


def run_benchmark(config: ExperimentConfig, out_dir: Path, device=None, raw_dir=None,
                  models: list[str] | None = None, verbose: bool = False, fixture: bool = False) -> Path:
    """``fixture=True`` runs on synthetic test data (non-strict loading, cache
    kept inside ``out_dir``) and marks every artifact as NOT REAL DATA."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(device)
    if config.num_threads:
        torch.set_num_threads(config.num_threads)
    models = models or config.models
    for name in models:
        spec(name)  # fail fast on typos

    windows = load_windows(config.preprocess, raw_dir=raw_dir, strict=not fixture,
                           cache_dir=out_dir / "cache" if fixture else None)
    data = prepare(windows, config)
    folds = make_folds(windows.subject, config.protocol, config.protocol.fold_seed)

    with open(out_dir / "config.yaml", "w") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
    manifest_path = out_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"runs": {}}
    manifest.update({
        "experiment": config.name, "device": str(device), "provenance": provenance(raw_dir),
        "synthetic_fixture_not_real_data": fixture,
        "data": {"n_windows": int(len(windows)), "n_subjects": int(len(windows.subjects)),
                 "class_counts": np.bincount(windows.y, minlength=2).tolist(),
                 "window_shape": list(windows.X.shape[1:]), "rejection": windows.rejection},
        "folds": [f.as_dict() for f in folds],
    })
    write_json(manifest_path, manifest)

    for model_name in models:
        model_dir = out_dir / "runs" / model_name
        print(f"\n=== {model_name} ({spec(model_name).display}) ===")
        if spec(model_name).kind == "deep" and not (model_dir / "complexity.json").exists():
            write_json(model_dir / "complexity.json",
                       profile(build_deep(model_name, n_times=windows.X.shape[-1]), windows.X.shape[1:], device))
        for seed in config.seeds:
            path = model_dir / f"seed_{seed}.json"
            if path.exists():
                print(f"  seed {seed}: done, skipping ({path.name})")
                continue
            print(f"  seed {seed}")
            result = run_model_seed(model_name, data, folds, seed, config, device, verbose,
                                    checkpoint_dir=out_dir / "checkpoints" if config.save_checkpoints else None)
            write_json(path, result)
            w = result["metrics"]["window"]
            print(f"  seed {seed}: acc={w['accuracy']:.3f} bacc={w['balanced_accuracy']:.3f} "
                  f"auc={w['roc_auc']:.3f} kappa={w['cohen_kappa']:.3f}")
            manifest["runs"].setdefault(model_name, []).append(seed)
            write_json(manifest_path, manifest)
    return out_dir
