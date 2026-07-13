"""
The research-grade with-GAN vs without-GAN comparison, for the paper.

Trains a SEPARATE CWGAN-GP inside EACH fold's training split only (never
sees that fold's held-out validation subjects -- avoids leakage), generates
synthetic data to augment that fold's training set, then trains the given
classifier on real-only vs real+synthetic. Uses the IDENTICAL GroupKFold
folds as train_baseline.py (same seed/order) so results are directly
comparable to your Phase 2 baseline numbers.

Within each fold, an inner validation split (carved from that fold's
training subjects only) is used for checkpoint selection -- the fold's real
held-out subjects are touched exactly once, for the final reported metric,
never during training or checkpoint selection. Synthetic data is added only
to the inner-training partition, never to inner-validation or the held-out
fold.

Usage:
    python -m src.compare_gan_augmentation --model 1dcnn --gan_epochs 200 --clf_epochs 30
"""

import argparse
import json

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from . import config
from .gan import LATENT_DIM
from .labeling import build_dataset
from .preprocessing import load_processed
from .train_baseline import train_one_fold, set_seed, split_inner_validation
from .train_gan import train_gan, generate_synthetic


def run_comparison(model_name: str, gan_epochs: int, clf_epochs: int,
                    n_folds: int = config.N_FOLDS, batch_size: int = 64,
                    synth_fraction: float = 1.0):
    """
    synth_fraction: amount of synthetic data to add, relative to real training
    set size per class (1.0 = double the minority class up to majority-class
    count; adjust if you want a different augmentation ratio).
    """
    set_seed()
    device = config.get_device()
    print(f"Using device: {device}")

    processed = load_processed()
    X, y, groups = build_dataset(processed)
    print(f"Full dataset: X={X.shape}, y={y.shape}, subjects={len(set(groups))}")

    gkf = GroupKFold(n_splits=n_folds)
    folds = list(gkf.split(X, y, groups))  # SAME split logic as train_baseline.py (same seed/order)

    results = {"without_gan": [], "with_gan": []}

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        groups_train = groups[train_idx]
        val_subjects = sorted(set(groups[val_idx].tolist()))
        print(f"\n{'='*60}\nFold {fold_i+1}/{n_folds} (held-out subjects: {val_subjects})\n{'='*60}")

        # Carve inner validation from this fold's TRAINING subjects only.
        # X_val/y_val (this fold's real held-out subjects) is never used for
        # checkpoint selection -- only for the final reported metric, once,
        # for both conditions below.
        X_it, y_it, X_iv, y_iv = split_inner_validation(X_train, y_train, groups_train)

        # --- WITHOUT GAN: baseline on real inner-training data only ---
        print("  [without GAN] training...")
        metrics_no_gan, _ = train_one_fold(
            model_name, X_it, y_it, X_iv, y_iv, X_val, y_val,
            device=device, epochs=clf_epochs, batch_size=batch_size,
        )
        print(f"    acc={metrics_no_gan['accuracy']:.3f} f1_macro={metrics_no_gan['f1_macro']:.3f}")
        results["without_gan"].append(metrics_no_gan)

        # --- WITH GAN: train CWGAN-GP on this fold's FULL training data
        # (X_train, not just X_it) -- more real data for the GAN to learn
        # from is preferable, and the GAN itself has no checkpoint-selection
        # step that could leak from X_val. ---
        print("  [with GAN] training CWGAN-GP on this fold's training subjects...")
        gen, crit, history = train_gan(X_train, y_train, device=device,
                                         epochs=gan_epochs, batch_size=batch_size)

        n_class0 = int((y_train == 0).sum())
        n_class1 = int((y_train == 1).sum())
        n_minority = min(n_class0, n_class1)
        n_majority = max(n_class0, n_class1)
        n_to_generate = int((n_majority - n_minority) * synth_fraction)
        n_to_generate = max(n_to_generate, 100)  # always generate at least something
        minority_class = 0 if n_class0 < n_class1 else 1

        print(f"  Generating {n_to_generate} synthetic samples for minority class={minority_class} "
              f"(real class balance: {n_class0} vs {n_class1})...")
        X_synth, y_synth = generate_synthetic(
            gen, {minority_class: n_to_generate}, device,
            n_timepoints=X_train.shape[1], n_channels=X_train.shape[2],
        )

        # IMPORTANT: synthetic data augments ONLY the inner-training
        # partition (X_it/y_it), never inner-val (X_iv/y_iv) and never the
        # real held-out fold (X_val/y_val). Checkpoint selection must be
        # judged on real data only.
        X_it_aug = np.concatenate([X_it, X_synth], axis=0)
        y_it_aug = np.concatenate([y_it, y_synth], axis=0)
        print(f"  Augmented inner-train set: {X_it.shape[0]} real + {X_synth.shape[0]} synthetic "
              f"= {X_it_aug.shape[0]} total")

        print("  [with GAN] training classifier on real+synthetic...")
        metrics_gan, _ = train_one_fold(
            model_name, X_it_aug, y_it_aug, X_iv, y_iv, X_val, y_val,
            device=device, epochs=clf_epochs, batch_size=batch_size,
        )
        print(f"    acc={metrics_gan['accuracy']:.3f} f1_macro={metrics_gan['f1_macro']:.3f}")
        results["with_gan"].append(metrics_gan)

    return results


def summarize_comparison(results: dict, model_name: str):
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY: {model_name}")
    print(f"{'='*70}")
    for condition in ("without_gan", "with_gan"):
        accs = [m["accuracy"] for m in results[condition]]
        f1s = [m["f1_macro"] for m in results[condition]]
        print(f"  {condition:14s}  accuracy = {np.mean(accs):.3f} +/- {np.std(accs):.3f}   "
              f"f1_macro = {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
    print(f"{'='*70}")


def save_comparison(results: dict, model_name: str):
    out_path = config.OUTPUTS_DIR / f"gan_comparison_{model_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved comparison results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="1dcnn")
    parser.add_argument("--gan_epochs", type=int, default=200)
    parser.add_argument("--clf_epochs", type=int, default=30)
    parser.add_argument("--folds", type=int, default=config.N_FOLDS)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    results = run_comparison(
        args.model, gan_epochs=args.gan_epochs, clf_epochs=args.clf_epochs,
        n_folds=args.folds, batch_size=args.batch_size,
    )
    summarize_comparison(results, args.model)
    save_comparison(results, args.model)
