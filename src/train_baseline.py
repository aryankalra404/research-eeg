"""
Phase 2: trains and evaluates all six EEG baseline models on REAL data only
(no GAN augmentation yet -- this is the paper's control condition).

Uses subject-independent GroupKFold cross-validation, as locked in
config.SPLIT_PROTOCOL / config.N_FOLDS, so results here are directly
comparable to the Phase 5 augmented-training run later (same folds must
be reused -- see note at bottom).

Usage:
    python -m src.train_baseline --dataset stew
    python -m src.train_baseline --dataset stew --model eegnet_adapted
    python -m src.train_baseline --dataset stew --epochs 5
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from . import config
from .datasets import EEGWindowDataset
from .evaluation import evaluate_predictions
from .labeling import build_dataset
from .models import MODEL_REGISTRY, get_model
from .preprocessing import load_processed
from .provenance import build_provenance, write_json
from .reproducibility import set_seed
from .split import inner_group_split_indices


def split_inner_validation(X, y, groups, inner_val_fraction: float = 0.15,
                            seed: int = config.RANDOM_SEED):
    """
    Carves an inner validation set out of TRAINING data only, grouped by
    subject so no subject straddles inner-train/inner-val. This inner-val
    set is used ONLY for checkpoint selection (picking the best epoch) --
    it must never be the same set used for final reported metrics, or you
    get optimistic bias from implicitly tuning on your test/held-out data.

    Returns: X_inner_train, y_inner_train, X_inner_val, y_inner_val
    """
    inner_train_idx, inner_val_idx = inner_group_split_indices(
        groups,
        inner_val_fraction=inner_val_fraction,
        seed=seed,
    )
    if inner_val_idx is None:
        print(f"  [warn] only {len(np.unique(groups))} training subjects available -- "
              f"cannot carve a proper inner validation split. Using last-epoch "
              f"checkpoint instead of best-epoch selection for this run.")
        return X, y, None, None
    return X[inner_train_idx], y[inner_train_idx], X[inner_val_idx], y[inner_val_idx]


def train_one_fold(model_name: str, X_inner_train, y_inner_train, X_inner_val, y_inner_val,
                    X_holdout, y_holdout, groups_holdout, device, epochs: int,
                    batch_size: int = 64, lr: float = 1e-3,
                    seed: int = config.RANDOM_SEED):
    """
    X_inner_train/y_inner_train: used for gradient updates.
    X_inner_val/y_inner_val: used ONLY to pick the best-epoch checkpoint
        (can be None if too few subjects to split -- falls back to last epoch).
    X_holdout/y_holdout: the real fold-validation or test set. Touched exactly
        ONCE, after checkpoint selection is already decided, to compute the
        final reported metrics. This is what prevents checkpoint-selection
        leakage into the reported numbers.
    """
    set_seed(seed)
    n_timepoints = X_inner_train.shape[1]
    n_channels = X_inner_train.shape[2]

    model = get_model(model_name, n_channels=n_channels, n_timepoints=n_timepoints, n_classes=2)
    model.to(device)

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        EEGWindowDataset(X_inner_train, y_inner_train),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    has_inner_val = X_inner_val is not None
    if has_inner_val:
        inner_val_loader = DataLoader(EEGWindowDataset(X_inner_val, y_inner_val),
                                        batch_size=batch_size, shuffle=False)
    holdout_loader = DataLoader(EEGWindowDataset(X_holdout, y_holdout),
                                  batch_size=batch_size, shuffle=False)

    # Class-weighted loss, computed from inner-train only (never from
    # inner-val or holdout -- those must stay untouched by anything that
    # influences training).
    class_counts = np.bincount(y_inner_train, minlength=2)
    class_weights = torch.tensor(
        [len(y_inner_train) / (2 * c) if c > 0 else 0.0 for c in class_counts], dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = epochs
    training_history = {"train_loss": [], "inner_val_f1_macro": []}

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        training_history["train_loss"].append(float(np.mean(epoch_losses)))

        if has_inner_val:
            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for xb, yb in inner_val_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(yb.numpy())
            val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
            training_history["inner_val_f1_macro"].append(float(val_f1))
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            training_history["inner_val_f1_macro"].append(None)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics computed on the holdout set EXACTLY ONCE, after checkpoint
    # selection is already locked in.
    model.eval()
    holdout_probabilities, holdout_true = [], []
    with torch.no_grad():
        for xb, yb in holdout_loader:
            xb = xb.to(device)
            logits = model(xb)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            holdout_probabilities.extend(probabilities)
            holdout_true.extend(yb.numpy())

    evaluation = evaluate_predictions(
        np.asarray(holdout_true),
        np.asarray(holdout_probabilities),
        np.asarray(groups_holdout),
        seed=seed,
    )
    metrics = dict(evaluation["window_level"])
    metrics["subject_condition_level"] = evaluation["subject_condition_level"]
    metrics["selected_epoch"] = int(best_epoch)
    metrics["training_history"] = training_history
    return metrics, model


def run_all(model_names, epochs: int, n_folds: int = config.N_FOLDS, batch_size: int = 64,
            dataset: str = config.DEFAULT_DATASET, seed: int = config.RANDOM_SEED,
            run_name: str | None = None, overwrite: bool = False):
    dataset = config.normalize_dataset_name(dataset)
    set_seed(seed)
    device = config.get_device()
    print(f"Using device: {device}")
    print(f"Dataset: {dataset}")

    print("Loading processed data...")
    processed = load_processed(dataset=dataset)
    X, y, groups = build_dataset(processed)
    print(f"X: {X.shape}, y: {y.shape}, groups: {len(set(groups))} unique subjects")

    gkf = GroupKFold(n_splits=n_folds)
    # Fixed fold assignment -- IMPORTANT: reuse this exact split in Phase 5
    # (augmented training) so baseline vs augmented comparisons are apples-to-apples.
    folds = list(gkf.split(X, y, groups))
    split_path = config.processed_dir(dataset) / "split.json"
    provenance = build_provenance(dataset, split_path if split_path.exists() else None)

    all_results = {}
    for model_name in model_names:
        model_run_name = (
            f"{run_name}_{model_name}"
            if run_name
            else f"baseline_{model_name}_seed{seed}_epochs{epochs}_folds{n_folds}"
        )
        model_run_dir = config.run_dir(dataset, model_run_name)
        model_checkpoint_dir = config.model_dir(dataset, model_run_name)
        existing_paths = [
            path for path in (model_run_dir, model_checkpoint_dir) if path.exists()
        ]
        if existing_paths and not overwrite:
            raise FileExistsError(
                "Baseline artifacts already exist; choose another --run_name or pass "
                "--overwrite: " + ", ".join(str(path) for path in existing_paths)
            )
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
        fold_metrics = []
        fold_protocol = []
        t0 = time.time()

        for fold_i, (train_idx, val_idx) in enumerate(folds):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            groups_train = groups[train_idx]
            val_subjects = sorted(set(groups[val_idx].tolist()))
            fold_seed = seed + fold_i

            print(f"  Fold {fold_i+1}/{n_folds} (held-out subjects: {val_subjects}) "
                  f"train={len(train_idx)} val={len(val_idx)}...")

            inner_train_idx, inner_val_idx = inner_group_split_indices(groups_train, seed=seed)
            if inner_val_idx is None:
                raise ValueError("Strict baseline evaluation requires at least three training subjects.")
            X_it, y_it = X_train[inner_train_idx], y_train[inner_train_idx]
            X_iv, y_iv = X_train[inner_val_idx], y_train[inner_val_idx]
            set_seed(fold_seed)

            metrics, model = train_one_fold(
                model_name, X_it, y_it, X_iv, y_iv, X_val, y_val,
                groups[val_idx], device=device, epochs=epochs, batch_size=batch_size,
                seed=fold_seed,
            )
            print(f"    acc={metrics['accuracy']:.3f} f1_macro={metrics['f1_macro']:.3f}")
            fold_metrics.append(metrics)
            checkpoint = model_checkpoint_dir / f"fold_{fold_i + 1}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint)
            fold_protocol.append({
                "fold": fold_i + 1,
                "seed": fold_seed,
                "inner_train_subjects": sorted(np.unique(groups_train[inner_train_idx]).astype(int).tolist()),
                "inner_val_subjects": sorted(np.unique(groups_train[inner_val_idx]).astype(int).tolist()),
                "holdout_subjects": val_subjects,
                "checkpoint": str(checkpoint.relative_to(config.PROJECT_ROOT)),
            })

        elapsed = time.time() - t0
        acc_mean = np.mean([m["accuracy"] for m in fold_metrics])
        acc_std = np.std([m["accuracy"] for m in fold_metrics])
        f1_mean = np.mean([m["f1_macro"] for m in fold_metrics])
        f1_std = np.std([m["f1_macro"] for m in fold_metrics])
        subject_acc_mean = np.mean([
            m["subject_condition_level"]["accuracy"] for m in fold_metrics
        ])
        subject_acc_std = np.std([
            m["subject_condition_level"]["accuracy"] for m in fold_metrics
        ])
        subject_f1_mean = np.mean([
            m["subject_condition_level"]["f1_macro"] for m in fold_metrics
        ])
        subject_f1_std = np.std([
            m["subject_condition_level"]["f1_macro"] for m in fold_metrics
        ])

        print(f"\n  {model_name} SUMMARY ({elapsed:.1f}s):")
        print(f"    accuracy = {acc_mean:.3f} +/- {acc_std:.3f}")
        print(f"    f1_macro = {f1_mean:.3f} +/- {f1_std:.3f}")
        print(f"    subject-condition accuracy = {subject_acc_mean:.3f} +/- {subject_acc_std:.3f}")
        print(f"    subject-condition f1 = {subject_f1_mean:.3f} +/- {subject_f1_std:.3f}")

        all_results[model_name] = {
            "fold_metrics": fold_metrics,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "f1_macro_mean": f1_mean,
            "f1_macro_std": f1_std,
            "subject_condition_accuracy_mean": subject_acc_mean,
            "subject_condition_accuracy_std": subject_acc_std,
            "subject_condition_f1_macro_mean": subject_f1_mean,
            "subject_condition_f1_macro_std": subject_f1_std,
            "elapsed_seconds": elapsed,
            "random_seed": seed,
            "epochs": epochs,
            "n_folds": n_folds,
            "run_name": model_run_name,
        }
        manifest = {
            "run_type": "subject_independent_baseline_cv",
            "run_name": model_run_name,
            "dataset": dataset,
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "n_folds": n_folds,
            "base_seed": seed,
            "fold_protocol": fold_protocol,
            "results": all_results[model_name],
            "provenance": provenance,
        }
        write_json(
            model_run_dir / "manifest.json",
            manifest,
        )

    return all_results


def save_results(results: dict, out_path=None, dataset: str = config.DEFAULT_DATASET):
    """
    Merges new results into the existing results file rather than overwriting
    it, so running one model at a time (e.g. across multiple sessions) doesn't
    destroy previously saved results for other models.
    """
    if out_path is None:
        out_path = config.output_dir(dataset) / "baseline_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if out_path.exists():
        with open(out_path, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                print(f"  [warn] existing {out_path} was invalid JSON -- overwriting it.")
                existing = {}

    seeded_results = {
        (
            f"{name}_seed{result.get('random_seed', config.RANDOM_SEED)}"
            f"_epochs{result.get('epochs', 'unknown')}"
            f"_folds{result.get('n_folds', 'unknown')}"
        ): result
        for name, result in results.items()
    }
    existing.update(seeded_results)

    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved/merged results to {out_path} (now contains: {list(existing.keys())})")


def print_summary_table(results: dict):
    print(f"\n{'='*108}")
    print(f"{'Model':<42} {'Window accuracy':<18} {'Window F1':<18} {'Subject-cond acc':<18}")
    print(f"{'-'*108}")
    for name, r in results.items():
        acc = f"{r['accuracy_mean']:.3f} +/- {r['accuracy_std']:.3f}"
        f1 = f"{r['f1_macro_mean']:.3f} +/- {r['f1_macro_std']:.3f}"
        subject_acc = (
            f"{r.get('subject_condition_accuracy_mean', float('nan')):.3f} +/- "
            f"{r.get('subject_condition_accuracy_std', float('nan')):.3f}"
        )
        print(f"{name:<42} {acc:<18} {f1:<18} {subject_acc:<18}")
    print(f"{'='*108}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=config.DEFAULT_DATASET, choices=config.SUPPORTED_DATASETS)
    parser.add_argument("--model", type=str, default=None,
                         help="Run one model (e.g. eegnet_adapted). Default: all 6.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=config.N_FOLDS)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_names = [args.model] if args.model else list(MODEL_REGISTRY.keys())

    results = run_all(model_names, epochs=args.epochs, n_folds=args.folds,
                      batch_size=args.batch_size, dataset=args.dataset, seed=args.seed,
                      run_name=args.run_name, overwrite=args.overwrite)
    save_results(results, dataset=args.dataset)

    # Reload the merged file so the printed summary shows every model run so far,
    # not just the ones from this invocation.
    results_path = config.output_dir(args.dataset) / "baseline_results.json"
    with open(results_path, "r") as f:
        all_results_so_far = json.load(f)
    print_summary_table(all_results_so_far)
