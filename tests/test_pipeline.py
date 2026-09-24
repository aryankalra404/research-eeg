"""End-to-end run on the synthetic fixture: benchmark -> report, plus the
protocol invariants that make the numbers trustworthy."""

import json

import numpy as np

from stewbench.experiments.benchmark import run_benchmark
from stewbench.reporting.report import benchmark_report
from stewbench.settings import ExperimentConfig, ProtocolConfig, TrainingConfig


def test_benchmark_end_to_end_and_no_subject_leakage(fixture_raw, tmp_path):
    config = ExperimentConfig(
        name="t", seeds=[0], models=["bp_lr", "riemann_mdm", "eegnet"],
        protocol=ProtocolConfig(n_folds=4), training=TrainingConfig(max_epochs=2, patience=2, batch_size=32),
    )
    out = run_benchmark(config, tmp_path / "bench", raw_dir=fixture_raw, fixture=True, device="cpu")
    for model in config.models:
        result = json.loads((out / "runs" / model / "seed_0.json").read_text())
        tested = []
        for fold in result["folds"]:
            train, val, test = (set(fold[k]) for k in ("train_subjects", "val_subjects", "test_subjects"))
            assert not (train & test) and not (val & test) and not (train & val)
            tested += fold["test_subjects"]
        assert sorted(tested) == list(range(1, 9))
        assert len(result["predictions"]["p1"]) == result["metrics"]["window"]["n"]
        assert set(result["importance_auc_drop"]) >= {"channel:AF3", "band:alpha"}
    # Resuming skips finished runs.
    mtime = (out / "runs" / "bp_lr" / "seed_0.json").stat().st_mtime
    run_benchmark(config, out, raw_dir=fixture_raw, fixture=True, device="cpu")
    assert (out / "runs" / "bp_lr" / "seed_0.json").stat().st_mtime == mtime

    report = benchmark_report(out)
    text = (report / "REPORT.md").read_text()
    assert "SYNTHETIC TEST FIXTURE" in text
    for name in ("table1_main_results", "table2_secondary_metrics", "table3_complexity"):
        for ext in (".csv", ".tex", ".md"):
            assert (report / "tables" / f"{name}{ext}").exists()
    assert (report / "figures" / "fig_cd_diagram.pdf").exists()


def test_same_seed_is_reproducible(fixture_raw, tmp_path):
    config = ExperimentConfig(name="r", seeds=[0], models=["eegnet"], protocol=ProtocolConfig(n_folds=4),
                              training=TrainingConfig(max_epochs=2, patience=2, batch_size=32),
                              interpretability=False)
    a = run_benchmark(config, tmp_path / "a", raw_dir=fixture_raw, fixture=True, device="cpu")
    b = run_benchmark(config, tmp_path / "b", raw_dir=fixture_raw, fixture=True, device="cpu")
    pa = json.loads((a / "runs/eegnet/seed_0.json").read_text())["predictions"]["p1"]
    pb = json.loads((b / "runs/eegnet/seed_0.json").read_text())["predictions"]["p1"]
    np.testing.assert_allclose(pa, pb, atol=1e-6)
