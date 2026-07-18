"""
Creates ONE fixed subject-independent train/test split and saves it to disk,
so every script (GAN training, baseline training, augmented training) uses
the exact same split -- required for a fair with-GAN vs without-GAN comparison.

Usage:
    python -m src.split --dataset dreamer
"""

import argparse
import json

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from . import config
from .labeling import build_dataset
from .preprocessing import load_processed


TEST_FRACTION = 0.2  # ~20% of subjects held out as the fixed test set


def create_split(dataset: str = config.DEFAULT_DATASET):
    dataset = config.normalize_dataset_name(dataset)
    processed_dir = config.processed_dir(dataset)
    processed = load_processed(dataset=dataset)
    X, y, groups = build_dataset(processed)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=config.RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))

    train_subjects = sorted(set(groups[train_idx].tolist()))
    test_subjects = sorted(set(groups[test_idx].tolist()))

    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
    print(f"Train windows: {len(train_idx)}, Test windows: {len(test_idx)}")

    split_info = {
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "test_fraction": TEST_FRACTION,
        "random_seed": config.RANDOM_SEED,
    }
    out_path = processed_dir / "split.json"
    with open(out_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved split to {out_path}")
    return split_info


def load_split(dataset: str = config.DEFAULT_DATASET):
    dataset = config.normalize_dataset_name(dataset)
    split_path = config.processed_dir(dataset) / "split.json"
    if not split_path.exists() and dataset == "dreamer":
        legacy_split_path = config.DATA_PROCESSED / "split.json"
        if legacy_split_path.exists():
            split_path = legacy_split_path
    if not split_path.exists():
        raise FileNotFoundError(f"No split found at {split_path}. Run `python -m src.split --dataset {dataset}` first.")
    with open(split_path, "r") as f:
        return json.load(f)


def apply_split(X, y, groups, split_info):
    """Returns X_train, y_train, X_test, y_test using the saved subject lists."""
    train_subjects = set(split_info["train_subjects"])
    test_subjects = set(split_info["test_subjects"])

    train_mask = np.isin(groups, list(train_subjects))
    test_mask = np.isin(groups, list(test_subjects))

    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


def apply_split_with_groups(X, y, groups, split_info):
    """
    Same as apply_split, but also returns groups_train -- needed for carving
    an inner validation split (see train_baseline.split_inner_validation)
    without leaking the real test set into checkpoint selection.
    """
    train_subjects = set(split_info["train_subjects"])
    test_subjects = set(split_info["test_subjects"])

    train_mask = np.isin(groups, list(train_subjects))
    test_mask = np.isin(groups, list(test_subjects))

    return X[train_mask], y[train_mask], groups[train_mask], X[test_mask], y[test_mask]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=config.DEFAULT_DATASET, choices=config.SUPPORTED_DATASETS)
    args = parser.parse_args()
    create_split(args.dataset)
