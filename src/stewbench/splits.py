"""Subject-independent cross-validation with nested, subject-disjoint
inner validation.

Three disjoint subject pools exist inside every fold:

    inner-train  -> gradient updates, feature scaling, generative models
    inner-val    -> early stopping / checkpoint selection / model tuning
    test         -> touched once, after the model is frozen

Fold assignment depends only on the subject IDs and ``fold_seed``; it is
identical for every model so per-subject results are paired across models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .settings import ProtocolConfig


@dataclass(frozen=True)
class Fold:
    index: int
    train_subjects: tuple[int, ...]
    val_subjects: tuple[int, ...]
    test_subjects: tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "fold": self.index,
            "train_subjects": list(self.train_subjects),
            "val_subjects": list(self.val_subjects),
            "test_subjects": list(self.test_subjects),
        }


def outer_test_groups(subjects: np.ndarray, protocol: ProtocolConfig, fold_seed: int) -> list[np.ndarray]:
    subjects = np.sort(np.unique(subjects))
    if protocol.scheme == "loso":
        return [np.array([s]) for s in subjects]
    n_folds = protocol.n_folds
    if not 2 <= n_folds <= len(subjects):
        raise ValueError(f"n_folds={n_folds} invalid for {len(subjects)} subjects")
    permuted = np.random.default_rng(fold_seed).permutation(subjects)
    return [np.sort(chunk) for chunk in np.array_split(permuted, n_folds)]


def make_folds(subjects: np.ndarray, protocol: ProtocolConfig, fold_seed: int = 2024) -> list[Fold]:
    """Folds are fixed by ``fold_seed`` (NOT the model seed), so every model
    and every training seed sees exactly the same subject partition."""
    subjects = np.sort(np.unique(subjects))
    folds = []
    for index, test in enumerate(outer_test_groups(subjects, protocol, fold_seed)):
        remaining = np.setdiff1d(subjects, test)
        rng = np.random.default_rng(fold_seed * 1000 + index)
        remaining = rng.permutation(remaining)
        # The validation pool is fixed first, so capping the number of training
        # subjects (data-efficiency curves) never changes validation or test.
        n_val = max(1, int(round(protocol.inner_val_fraction * len(remaining))))
        val, train = remaining[:n_val], remaining[n_val:]
        if protocol.max_train_subjects is not None:
            train = train[: protocol.max_train_subjects]
        if len(train) < 1:
            raise ValueError("Not enough subjects for a train/val/test partition")
        folds.append(Fold(
            index=index,
            train_subjects=tuple(int(s) for s in np.sort(train)),
            val_subjects=tuple(int(s) for s in np.sort(val)),
            test_subjects=tuple(int(s) for s in test),
        ))
    check_folds(folds, subjects)
    return folds


def check_folds(folds: list[Fold], subjects: np.ndarray) -> None:
    tested = []
    for fold in folds:
        pools = [set(fold.train_subjects), set(fold.val_subjects), set(fold.test_subjects)]
        if pools[0] & pools[1] or pools[0] & pools[2] or pools[1] & pools[2]:
            raise AssertionError(f"Subject overlap inside fold {fold.index}")
        tested.extend(fold.test_subjects)
    if sorted(tested) != sorted(np.unique(subjects).tolist()):
        raise AssertionError("Every subject must be tested exactly once")


def masks(subject: np.ndarray, fold: Fold) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.isin(subject, fold.train_subjects),
        np.isin(subject, fold.val_subjects),
        np.isin(subject, fold.test_subjects),
    )
