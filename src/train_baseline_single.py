"""
The actual with-GAN vs without-GAN comparison, using:
    - ONE fixed subject-independent train/test split (src/split.py)
    - The GAN trained ONCE on inner-training subjects, synthetic data SAVED to disk
      (src/train_gan.py -> data/processed/synthetic_train.npz)

This is simpler and more inspectable than retraining a GAN per fold: you can
open synthetic_train.npz, look at it, reuse it across multiple classifier
runs, and the GAN training/validation plots are a one-time artifact instead
of being regenerated 5+ times.

Usage:
    python -m src.train_baseline_single --dataset stew --model eegnet_adapted
    python -m src.train_baseline_single --dataset stew --model eegnet_adapted --use_gan --gan_run gan_400epoch
"""

import argparse
import json

import numpy as np

from . import config
from .labeling import build_dataset
from .preprocessing import load_processed
from .reproducibility import set_seed
from .split import apply_split_with_groups, inner_group_split_indices, load_split
from .train_baseline import train_one_fold


def run_single(model_name: str, use_gan: bool, epochs: int, batch_size: int = 64,
               dataset: str = config.DEFAULT_DATASET, gan_run: str = "gan_400epoch",
               seed: int = config.RANDOM_SEED):
    dataset = config.normalize_dataset_name(dataset)
    set_seed(seed)
    device = config.get_device()
    print(f"Using device: {device}")
    print(f"Dataset: {dataset}")

    processed = load_processed(dataset=dataset)
    X, y, groups = build_dataset(processed)

    split_info = load_split(dataset=dataset)
    X_train, y_train, groups_train, X_test, y_test, groups_test = apply_split_with_groups(
        X, y, groups, split_info
    )
    print(f"Train: {X_train.shape[0]} windows ({len(split_info['train_subjects'])} subjects)")
    print(f"Test:  {X_test.shape[0]} windows ({len(split_info['test_subjects'])} subjects)")
    class0_name, class1_name = config.class_names(dataset)
    print(f"Train class balance before augmentation: "
          f"class0={int((y_train==0).sum())} ({class0_name}), "
          f"class1={int((y_train==1).sum())} ({class1_name})")

    # Carve inner validation from TRAINING subjects only, BEFORE any synthetic
    # augmentation. X_test is never touched here -- it's used exactly once,
    # at the end, for the reported metric.
    inner_train_idx, inner_val_idx = inner_group_split_indices(groups_train, seed=seed)
    if inner_val_idx is None:
        raise ValueError("At least three training subjects are required for strict validation isolation.")
    X_it, y_it = X_train[inner_train_idx], y_train[inner_train_idx]
    X_iv, y_iv = X_train[inner_val_idx], y_train[inner_val_idx]
    expected_gan_subjects = sorted(np.unique(groups_train[inner_train_idx]).astype(int).tolist())
    expected_inner_val_subjects = sorted(np.unique(groups_train[inner_val_idx]).astype(int).tolist())

    condition = "WITHOUT GAN"
    if use_gan:
        synth_path = config.processed_dir(dataset) / f"synthetic_train_{gan_run}.npz"
        legacy_synth_path = config.processed_dir(dataset) / "synthetic_train.npz"
        if not synth_path.exists() and legacy_synth_path.exists():
            synth_path = legacy_synth_path
        if not synth_path.exists():
            raise FileNotFoundError(
                f"No synthetic data found at {synth_path}. "
                f"Run `python -m src.train_gan --dataset {dataset} --run_name {gan_run}` first."
            )
        synth_data = np.load(synth_path)
        required_metadata = {
            "dataset", "random_seed", "gan_train_subjects", "inner_val_subjects", "test_subjects"
        }
        missing_metadata = required_metadata - set(synth_data.files)
        if missing_metadata:
            raise ValueError(
                f"Synthetic data {synth_path} predates strict split metadata "
                f"(missing {sorted(missing_metadata)}). Retrain this GAN run."
            )
        saved_dataset = str(synth_data["dataset"].item())
        saved_seed = int(synth_data["random_seed"].item())
        saved_gan_subjects = sorted(synth_data["gan_train_subjects"].astype(int).tolist())
        saved_inner_val_subjects = sorted(synth_data["inner_val_subjects"].astype(int).tolist())
        saved_test_subjects = sorted(synth_data["test_subjects"].astype(int).tolist())
        expected_test_subjects = sorted(split_info["test_subjects"])
        if (
            saved_dataset != dataset
            or saved_seed != seed
            or saved_gan_subjects != expected_gan_subjects
            or saved_inner_val_subjects != expected_inner_val_subjects
            or saved_test_subjects != expected_test_subjects
        ):
            raise ValueError(
                "Synthetic data protocol metadata does not match this classifier run. "
                "Use the same dataset/split seed or retrain the GAN."
            )
        X_synth, y_synth = synth_data["X_synth"], synth_data["y_synth"]
        print(f"Loaded {len(y_synth)} synthetic windows from {synth_path} "
              f"(class0={int((y_synth==0).sum())}, class1={int((y_synth==1).sum())})")

        # IMPORTANT: synthetic data is added ONLY to the inner-training
        # partition, never to inner-val or the real test set. Checkpoint
        # selection must be judged on real data only.
        X_it = np.concatenate([X_it, X_synth], axis=0)
        y_it = np.concatenate([y_it, y_synth], axis=0)
        condition = "WITH GAN"
        print(f"Augmented inner-train set: {X_it.shape[0]} windows "
              f"(class0={int((y_it==0).sum())}, class1={int((y_it==1).sum())})")

    print(f"\n{'='*60}\n{condition} -- training {model_name}\n{'='*60}")
    metrics, model = train_one_fold(
        model_name, X_it, y_it, X_iv, y_iv, X_test, y_test,
        groups_test, device=device, epochs=epochs, batch_size=batch_size, seed=seed,
    )
    metrics["dataset"] = dataset
    metrics["random_seed"] = seed
    metrics["gan_run"] = gan_run if use_gan else None

    print(f"\nRESULTS ({condition}):")
    print(f"  accuracy       = {metrics['accuracy']:.4f}")
    print(f"  f1_macro       = {metrics['f1_macro']:.4f}")
    print(f"  precision_macro = {metrics['precision_macro']:.4f}")
    print(f"  recall_macro    = {metrics['recall_macro']:.4f}")
    print(f"  confusion_matrix = {metrics['confusion_matrix']}")
    subject_metrics = metrics["subject_condition_level"]
    print(f"  subject-condition accuracy = {subject_metrics['accuracy']:.4f}")
    print(f"  subject-condition f1_macro = {subject_metrics['f1_macro']:.4f}")

    return metrics


def save_result(model_name: str, use_gan: bool, metrics: dict, dataset: str = config.DEFAULT_DATASET):
    out_path = config.output_dir(dataset) / "single_split_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if out_path.exists():
        with open(out_path, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}

    seed = int(metrics.get("random_seed", config.RANDOM_SEED))
    key = f"{model_name}_{'with_gan' if use_gan else 'without_gan'}_seed{seed}"
    existing[key] = metrics

    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved result under key '{key}' to {out_path}")

    without_key = f"{model_name}_without_gan_seed{seed}"
    with_key = f"{model_name}_with_gan_seed{seed}"
    if without_key in existing and with_key in existing:
        wo = existing[without_key]
        w = existing[with_key]
        print(f"\n{'='*60}\nCOMPARISON for {model_name}\n{'='*60}")
        print(f"  without GAN: accuracy={wo['accuracy']:.4f}  f1_macro={wo['f1_macro']:.4f}")
        print(f"  with GAN:    accuracy={w['accuracy']:.4f}  f1_macro={w['f1_macro']:.4f}")
        print(f"  delta:       accuracy={w['accuracy']-wo['accuracy']:+.4f}  "
              f"f1_macro={w['f1_macro']-wo['f1_macro']:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=config.DEFAULT_DATASET, choices=config.SUPPORTED_DATASETS)
    parser.add_argument("--model", type=str, default="1dcnn")
    parser.add_argument("--use_gan", action="store_true", help="Augment training data with saved synthetic data")
    parser.add_argument("--gan_run", type=str, default="gan_400epoch")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    metrics = run_single(args.model, use_gan=args.use_gan, epochs=args.epochs,
                         batch_size=args.batch_size, dataset=args.dataset,
                         gan_run=args.gan_run, seed=args.seed)
    save_result(args.model, args.use_gan, metrics, dataset=args.dataset)
