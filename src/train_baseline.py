"""
Phase 2: trains and evaluates all six baseline models on REAL data only
(no GAN augmentation yet -- this is the paper's control condition).

Uses subject-independent GroupKFold cross-validation, as locked in
config.SPLIT_PROTOCOL / config.N_FOLDS, so results here are directly
comparable to the Phase 5 augmented-training run later (same folds must
be reused -- see note at bottom).

Usage:
    python -m src.train_baseline                  # all 6 models, all folds
    python -m src.train_baseline --model eegnet    # just one model, for a quick check
    python -m src.train_baseline --epochs 5        # override epoch count for a smoke test
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


def split_train_val_by_subject(X, y, groups, val_fraction: float = 0.15,
                                seed: int = config.RANDOM_SEED):
    """
    Carves an INNER validation set out of a training pool, split by subject
    (never by window), so checkpoint selection never touches the real test
    set. Returns (train_idx, val_idx) into the arrays passed in.

    Falls back to using all data for both if there are too few subjects to
    split cleanly (only happens on tiny debug runs, not the real dataset).
    """
    n_unique_subjects = len(set(groups.tolist()))
    if n_unique_subjects < 3:
        idx = np.arange(len(y))
        return idx, idx

    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(gss.split(X, y, groups))
    return train_idx, val_idx


def train_and_evaluate(model_name: str, X_tr, y_tr, X_val, y_val, X_test, y_test,
                        device, epochs: int, batch_size: int = 64, lr: float = 1e-3):
    """
    X_tr/y_tr     -- used for gradient updates only
    X_val/y_val   -- used ONLY to pick the best-epoch checkpoint (never the test set)
    X_test/y_test -- touched exactly once, after the best checkpoint is loaded,
                      to produce the metrics that actually get reported
    """
    n_timepoints = X_tr.shape[1]
    n_channels = X_tr.shape[2]

    model = get_model(model_name, n_channels=n_channels, n_timepoints=n_timepoints, n_classes=2)
    model.to(device)

    train_loader = DataLoader(EEGWindowDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EEGWindowDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(EEGWindowDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # Class-weighted loss to counter the ~38.5/61.5 imbalance we measured in Phase 1
    class_counts = np.bincount(y_tr, minlength=2)
    class_weights = torch.tensor(
        [len(y_tr) / (2 * c) if c > 0 else 0.0 for c in class_counts], dtype=torch.float32
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

        # Checkpoint selection uses ONLY the inner validation set, never X_test.
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(yb.numpy())
        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Reload best checkpoint (chosen on val), then evaluate on the TEST set --
    # this is the ONLY time X_test/y_test are touched.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(yb.numpy())

    metrics = {
        "accuracy": accuracy_score(test_true, test_preds),
        "f1_macro": f1_score(test_true, test_preds, average="macro", zero_division=0),
        "precision_macro": precision_score(test_true, test_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(test_true, test_preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(test_true, test_preds).tolist(),
        "best_val_f1_at_selection": best_val_f1,  # kept for transparency, not a test-set metric
    }
    return metrics, model


# Backwards-compatible alias: some callers (train_baseline_single.py,
# compare_gan_augmentation.py) still import `train_one_fold`. It now REQUIRES
# a proper inner validation split rather than reusing the test set for
# checkpoint selection.
def train_one_fold(model_name: str, X_train, y_train, X_test, y_test,
                    device, epochs: int, batch_size: int = 64, lr: float = 1e-3,
                    groups_train=None, val_fraction: float = 0.15):
    if groups_train is None:
        raise ValueError(
            "train_one_fold now requires groups_train (subject IDs for X_train) "
            "so it can carve out an inner validation split by subject instead of "
            "reusing X_test for checkpoint selection. Pass groups_train=... ."
        )
    inner_train_idx, inner_val_idx = split_train_val_by_subject(
        X_train, y_train, groups_train, val_fraction=val_fraction
    )
    X_tr, y_tr = X_train[inner_train_idx], y_train[inner_train_idx]
    X_val, y_val = X_train[inner_val_idx], y_train[inner_val_idx]
    return train_and_evaluate(
        model_name, X_tr, y_tr, X_val, y_val, X_test, y_test,
        device=device, epochs=epochs, batch_size=batch_size, lr=lr,
    )


def run_all(model_names, epochs: int, n_folds: int = config.N_FOLDS, batch_size: int = 64):
    set_seed()
    device = config.get_device()
    print(f"Using device: {device}")

    print("Loading processed data...")
    processed = load_processed()
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

        for fold_i, (train_idx, test_idx) in enumerate(folds):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            groups_train = groups[train_idx]
            test_subjects = sorted(set(groups[test_idx].tolist()))

            print(f"  Fold {fold_i+1}/{n_folds} (held-out TEST subjects: {test_subjects}) "
                  f"train_pool={len(train_idx)} test={len(test_idx)}...")

            metrics, _ = train_one_fold(
                model_name, X_train, y_train, X_test, y_test,
                device=device, epochs=epochs, batch_size=batch_size,
                groups_train=groups_train,
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


def save_results(results: dict, out_path=None):
    """
    Merges new results into the existing results file rather than overwriting
    it, so running one model at a time (e.g. across multiple sessions) doesn't
    destroy previously saved results for other models.
    """
    if out_path is None:
        out_path = config.OUTPUTS_DIR / "baseline_results.json"
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
    parser.add_argument("--model", type=str, default=None,
                         help="Run just one model (e.g. eegnet). Default: all 6.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=config.N_FOLDS)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    model_names = [args.model] if args.model else list(MODEL_REGISTRY.keys())

    results = run_all(model_names, epochs=args.epochs, n_folds=args.folds, batch_size=args.batch_size)
    save_results(results)

    # Reload the merged file so the printed summary shows every model run so far,
    # not just the ones from this invocation.
    results_path = config.OUTPUTS_DIR / "baseline_results.json"
    with open(results_path, "r") as f:
        all_results_so_far = json.load(f)
    print_summary_table(all_results_so_far)
    