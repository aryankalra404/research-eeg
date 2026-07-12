"""
The actual with-GAN vs without-GAN comparison your teacher asked for.

For fairness, this trains a SEPARATE CWGAN-GP inside EACH fold's training
split only (never sees that fold's held-out validation subjects -- avoids
any leakage), generates synthetic data to augment that fold's training set,
then trains 1D-CNN on real-only vs real+synthetic, using the IDENTICAL folds
as your original Phase 2 baseline run (same GroupKFold call, same seed) so
the comparison is apples-to-apples.

This is slower than training one GAN on everything (a GAN gets trained per
fold), but it's the methodologically correct way to avoid the augmented
run having an unfair advantage from synthetic data derived from subjects
the model will later be tested against indirectly.

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
from .train_baseline import train_and_evaluate, split_train_val_by_subject, set_seed
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

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_train_pool, y_train_pool = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        groups_train_pool = groups[train_idx]
        test_subjects = sorted(set(groups[test_idx].tolist()))
        print(f"\n{'='*60}\nFold {fold_i+1}/{n_folds} (held-out TEST subjects: {test_subjects})\n{'='*60}")

        # Carve an inner train/val split (by subject) out of this fold's
        # training pool. The val split is used ONLY for checkpoint selection
        # and is never touched by the GAN or by synthetic data.
        inner_train_idx, inner_val_idx = split_train_val_by_subject(
            X_train_pool, y_train_pool, groups_train_pool
        )
        X_tr, y_tr = X_train_pool[inner_train_idx], y_train_pool[inner_train_idx]
        X_val, y_val = X_train_pool[inner_val_idx], y_train_pool[inner_val_idx]

        # --- WITHOUT GAN: baseline on real training data only ---
        print("  [without GAN] training...")
        metrics_no_gan, _ = train_and_evaluate(
            model_name, X_tr, y_tr, X_val, y_val, X_test, y_test,
            device=device, epochs=clf_epochs, batch_size=batch_size,
        )
        print(f"    acc={metrics_no_gan['accuracy']:.3f} f1_macro={metrics_no_gan['f1_macro']:.3f}")
        results["without_gan"].append(metrics_no_gan)

        # --- WITH GAN: train CWGAN-GP on this fold's INNER TRAINING data only
        # (never the inner-val subjects, and never the outer test subjects) ---
        print("  [with GAN] training CWGAN-GP on this fold's inner-train subjects...")
        gen, crit, history = train_gan(X_tr, y_tr, device=device,
                                         epochs=gan_epochs, batch_size=batch_size)

        n_class0 = int((y_tr == 0).sum())
        n_class1 = int((y_tr == 1).sum())
        n_minority = min(n_class0, n_class1)
        n_majority = max(n_class0, n_class1)
        n_to_generate_per_class = int((n_majority - n_minority) * synth_fraction)
        n_to_generate_per_class = max(n_to_generate_per_class, 100)  # always generate at least something

        print(f"  Generating {n_to_generate_per_class} synthetic samples per class "
              f"(real class balance: {n_class0} vs {n_class1})...")
        X_synth, y_synth = generate_synthetic(
            gen, {0: n_to_generate_per_class, 1: n_to_generate_per_class}, device,
            n_timepoints=X_tr.shape[1], n_channels=X_tr.shape[2],
        )

        # Synthetic data augments ONLY the inner-train set -- X_val (checkpoint
        # selection) and X_test (final report) stay 100% real.
        X_tr_aug = np.concatenate([X_tr, X_synth], axis=0)
        y_tr_aug = np.concatenate([y_tr, y_synth], axis=0)
        print(f"  Augmented inner-train set: {X_tr.shape[0]} real + {X_synth.shape[0]} synthetic "
              f"= {X_tr_aug.shape[0]} total")

        print("  [with GAN] training classifier on real+synthetic...")
        metrics_gan, _ = train_and_evaluate(
            model_name, X_tr_aug, y_tr_aug, X_val, y_val, X_test, y_test,
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