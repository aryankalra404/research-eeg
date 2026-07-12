"""
The actual with-GAN vs without-GAN comparison, using:
    - ONE fixed subject-independent train/test split (src/split.py)
    - The GAN trained ONCE on the train split, synthetic data SAVED to disk
      (src/train_gan.py -> data/processed/synthetic_train.npz)

This is simpler and more inspectable than retraining a GAN per fold: you can
open synthetic_train.npz, look at it, reuse it across multiple classifier
runs, and the GAN training/validation plots are a one-time artifact instead
of being regenerated 5+ times.

Usage:
    python -m src.train_baseline_single --model 1dcnn                  # without GAN
    python -m src.train_baseline_single --model 1dcnn --use_gan        # with GAN
    python -m src.train_baseline_single --model 1dcnn --use_gan --epochs 50
"""

import argparse
import json

import numpy as np

from . import config
from .labeling import build_dataset
from .preprocessing import load_processed
from .split import load_split, apply_split
from .train_baseline import train_one_fold, set_seed


def run_single(model_name: str, use_gan: bool, epochs: int, batch_size: int = 64):
    set_seed()
    device = config.get_device()
    print(f"Using device: {device}")

    processed = load_processed()
    X, y, groups = build_dataset(processed)

    split_info = load_split()
    X_train, y_train, X_test, y_test = apply_split(X, y, groups, split_info)
    print(f"Train: {X_train.shape[0]} windows ({len(split_info['train_subjects'])} subjects)")
    print(f"Test:  {X_test.shape[0]} windows ({len(split_info['test_subjects'])} subjects)")
    print(f"Train class balance before augmentation: "
          f"non-stress={int((y_train==0).sum())}, stress={int((y_train==1).sum())}")

    condition = "WITHOUT GAN"
    if use_gan:
        synth_path = config.DATA_PROCESSED / "synthetic_train.npz"
        if not synth_path.exists():
            raise FileNotFoundError(
                f"No synthetic data found at {synth_path}. "
                f"Run `python -m src.train_gan` first to train the GAN and save synthetic data."
            )
        synth_data = np.load(synth_path)
        X_synth, y_synth = synth_data["X_synth"], synth_data["y_synth"]
        print(f"Loaded {len(y_synth)} synthetic windows from {synth_path} "
              f"(non-stress={int((y_synth==0).sum())}, stress={int((y_synth==1).sum())})")

        X_train = np.concatenate([X_train, X_synth], axis=0)
        y_train = np.concatenate([y_train, y_synth], axis=0)
        condition = "WITH GAN"
        print(f"Augmented train set: {X_train.shape[0]} windows "
              f"(non-stress={int((y_train==0).sum())}, stress={int((y_train==1).sum())})")

    print(f"\n{'='*60}\n{condition} -- training {model_name}\n{'='*60}")
    metrics, model = train_one_fold(
        model_name, X_train, y_train, X_test, y_test,
        device=device, epochs=epochs, batch_size=batch_size,
    )

    print(f"\nRESULTS ({condition}):")
    print(f"  accuracy       = {metrics['accuracy']:.4f}")
    print(f"  f1_macro       = {metrics['f1_macro']:.4f}")
    print(f"  precision_macro = {metrics['precision_macro']:.4f}")
    print(f"  recall_macro    = {metrics['recall_macro']:.4f}")
    print(f"  confusion_matrix = {metrics['confusion_matrix']}")

    return metrics


def save_result(model_name: str, use_gan: bool, metrics: dict):
    out_path = config.OUTPUTS_DIR / "single_split_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if out_path.exists():
        with open(out_path, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}

    key = f"{model_name}_{'with_gan' if use_gan else 'without_gan'}"
    existing[key] = metrics

    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved result under key '{key}' to {out_path}")

    if f"{model_name}_without_gan" in existing and f"{model_name}_with_gan" in existing:
        wo = existing[f"{model_name}_without_gan"]
        w = existing[f"{model_name}_with_gan"]
        print(f"\n{'='*60}\nCOMPARISON for {model_name}\n{'='*60}")
        print(f"  without GAN: accuracy={wo['accuracy']:.4f}  f1_macro={wo['f1_macro']:.4f}")
        print(f"  with GAN:    accuracy={w['accuracy']:.4f}  f1_macro={w['f1_macro']:.4f}")
        print(f"  delta:       accuracy={w['accuracy']-wo['accuracy']:+.4f}  "
              f"f1_macro={w['f1_macro']-wo['f1_macro']:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="1dcnn")
    parser.add_argument("--use_gan", action="store_true", help="Augment training data with saved synthetic data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    metrics = run_single(args.model, use_gan=args.use_gan, epochs=args.epochs, batch_size=args.batch_size)
    save_result(args.model, args.use_gan, metrics)
