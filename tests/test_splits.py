import numpy as np
import pytest

from stewbench.settings import ProtocolConfig
from stewbench.splits import make_folds

SUBJECTS = np.repeat(np.arange(1, 49), 10)


@pytest.mark.parametrize("scheme,n_folds", [("group_kfold", 10), ("group_kfold", 5), ("loso", 0)])
def test_pools_disjoint_and_every_subject_tested_once(scheme, n_folds):
    folds = make_folds(SUBJECTS, ProtocolConfig(scheme=scheme, n_folds=n_folds or 10))
    tested = []
    for fold in folds:
        train, val, test = map(set, (fold.train_subjects, fold.val_subjects, fold.test_subjects))
        assert not (train & val) and not (train & test) and not (val & test)
        assert train and val and test
        tested += fold.test_subjects
    assert sorted(tested) == list(range(1, 49))
    if scheme == "loso":
        assert len(folds) == 48


def test_folds_depend_only_on_fold_seed():
    a = make_folds(SUBJECTS, ProtocolConfig(), fold_seed=7)
    b = make_folds(SUBJECTS, ProtocolConfig(), fold_seed=7)
    c = make_folds(SUBJECTS, ProtocolConfig(), fold_seed=8)
    assert [f.as_dict() for f in a] == [f.as_dict() for f in b]
    assert [f.as_dict() for f in a] != [f.as_dict() for f in c]


def test_training_budgets_are_nested_and_keep_val_and_test_fixed():
    full = make_folds(SUBJECTS, ProtocolConfig())
    for budget in (6, 12, 24):
        capped = make_folds(SUBJECTS, ProtocolConfig(max_train_subjects=budget))
        for f_full, f_cap in zip(full, capped):
            assert len(f_cap.train_subjects) == budget
            assert set(f_cap.train_subjects) <= set(f_full.train_subjects)
            assert f_cap.val_subjects == f_full.val_subjects
            assert f_cap.test_subjects == f_full.test_subjects
    small = make_folds(SUBJECTS, ProtocolConfig(max_train_subjects=6))
    mid = make_folds(SUBJECTS, ProtocolConfig(max_train_subjects=12))
    for a, b in zip(small, mid):
        assert set(a.train_subjects) <= set(b.train_subjects)
