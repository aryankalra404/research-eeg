"""
Phase 2: trains and evaluates all six baseline models on REAL data only
(no GAN augmentation yet -- this is the paper's control condition).

Uses subject-independent GroupKFold cross-validation, as locked in
config.SPLIT_PROTOCOL / config.N_FOLDS, so results here are directly
comparable to the Phase 5 augmented-training run later (same folds must
be reused -- see note at bottom).

Usage:
    python -m src.train_baseline --dataset dreamer
    python -m src.train_baseline --dataset dreamer --model eegnet
    python -m src.train_baseline --dataset dreamer --epochs 5
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader

from . import config
from .datasets import EEGWindowDataset
from .labeling import build_dataset
from .models import MODEL_REGISTRY, get_model
from .preprocessing import load_processed


def set_seed(seed: int = config.RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    n_unique_groups = len(set(groups.tolist()))
    if n_unique_groups < 3:
        print(f"  [warn] only {n_unique_groups} training subjects available -- "
              f"cannot carve a proper inner validation split. Using last-epoch "
              f"checkpoint instead of best-epoch selection for this run.")
        return X, y, None, None

    gss = GroupShuffleSplit(n_splits=1, test_size=inner_val_fraction, random_state=seed)
    it_idx, iv_idx = next(gss.split(X, y, groups))
    return X[it_idx], y[it_idx], X[iv_idx], y[iv_idx]


def train_one_fold(model_name: str, X_inner_train, y_inner_train, X_inner_val, y_inner_val,
                    X_holdout, y_holdout, device, epochs: int, batch_size: int = 64, lr: float = 1e-3):
    """
    X_inner_train/y_inner_train: used for gradient updates.
    X_inner_val/y_inner_val: used ONLY to pick the best-epoch checkpoint
        (can be None if too few subjects to split -- falls back to last epoch).
    X_holdout/y_holdout: the real fold-validation or test set. Touched exactly
        ONCE, after checkpoint selection is already decided, to compute the
        final reported metrics. This is what prevents checkpoint-selection
        leakage into the reported numbers.
    """
    n_timepoints = X_inner_train.shape[1]
    n_channels = X_inner_train.shape[2]

    model = get_model(model_name, n_channels=n_channels, n_timepoints=n_timepoints, n_classes=2)
    model.to(device)

    train_loader = DataLoader(EEGWindowDataset(X_inner_train, y_inner_train),
                                batch_size=batch_size, shuffle=True)
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

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

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
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics computed on the holdout set EXACTLY ONCE, after checkpoint
    # selection is already locked in.
    model.eval()
    holdout_preds, holdout_true = [], []
    with torch.no_grad():
        for xb, yb in holdout_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            holdout_preds.extend(preds)
            holdout_true.extend(yb.numpy())

    metrics = {
        "accuracy": accuracy_score(holdout_true, holdout_preds),
        "f1_macro": f1_score(holdout_true, holdout_preds, average="macro", zero_division=0),
        "precision_macro": precision_score(holdout_true, holdout_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(holdout_true, holdout_preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(holdout_true, holdout_preds).tolist(),
    }
    return metrics, model


def run_all(model_names, epochs: int, n_folds: int = config.N_FOLDS, batch_size: int = 64,
            dataset: str = config.DEFAULT_DATASET):
    dataset = config.normalize_dataset_name(dataset)
    set_seed()
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

    all_results = {}
    for model_name in model_names:
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
        fold_metrics = []
        t0 = time.time()

        for fold_i, (train_idx, val_idx) in enumerate(folds):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            groups_train = groups[train_idx]
            val_subjects = sorted(set(groups[val_idx].tolist()))

            print(f"  Fold {fold_i+1}/{n_folds} (held-out subjects: {val_subjects}) "
                  f"train={len(train_idx)} val={len(val_idx)}...")

            X_it, y_it, X_iv, y_iv = split_inner_validation(X_train, y_train, groups_train)

            metrics, _ = train_one_fold(
                model_name, X_it, y_it, X_iv, y_iv, X_val, y_val,
                device=device, epochs=epochs, batch_size=batch_size,
            )
            print(f"    acc={metrics['accuracy']:.3f} f1_macro={metrics['f1_macro']:.3f}")
            fold_metrics.append(metrics)

        elapsed = time.time() - t0
        acc_mean = np.mean([m["accuracy"] for m in fold_metrics])
        acc_std = np.std([m["accuracy"] for m in fold_metrics])
        f1_mean = np.mean([m["f1_macro"] for m in fold_metrics])
        f1_std = np.std([m["f1_macro"] for m in fold_metrics])

        print(f"\n  {model_name} SUMMARY ({elapsed:.1f}s):")
        print(f"    accuracy = {acc_mean:.3f} +/- {acc_std:.3f}")
        print(f"    f1_macro = {f1_mean:.3f} +/- {f1_std:.3f}")

        all_results[model_name] = {
            "fold_metrics": fold_metrics,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "f1_macro_mean": f1_mean,
            "f1_macro_std": f1_std,
            "elapsed_seconds": elapsed,
        }

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

    existing.update(results)  # new results for a model overwrite that model's old entry only

    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved/merged results to {out_path} (now contains: {list(existing.keys())})")


def print_summary_table(results: dict):
    print(f"\n{'='*70}")
    print(f"{'Model':<16} {'Accuracy':<18} {'F1 (macro)':<18}")
    print(f"{'-'*70}")
    for name, r in results.items():
        acc = f"{r['accuracy_mean']:.3f} +/- {r['accuracy_std']:.3f}"
        f1 = f"{r['f1_macro_mean']:.3f} +/- {r['f1_macro_std']:.3f}"
        print(f"{name:<16} {acc:<18} {f1:<18}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=config.DEFAULT_DATASET, choices=config.SUPPORTED_DATASETS)
    parser.add_argument("--model", type=str, default=None,
                         help="Run just one model (e.g. eegnet). Default: all 6.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=config.N_FOLDS)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    model_names = [args.model] if args.model else list(MODEL_REGISTRY.keys())

    results = run_all(model_names, epochs=args.epochs, n_folds=args.folds,
                      batch_size=args.batch_size, dataset=args.dataset)
    save_results(results, dataset=args.dataset)

    # Reload the merged file so the printed summary shows every model run so far,
    # not just the ones from this invocation.
    results_path = config.output_dir(args.dataset) / "baseline_results.json"
    with open(results_path, "r") as f:
        all_results_so_far = json.load(f)
    print_summary_table(all_results_so_far)
