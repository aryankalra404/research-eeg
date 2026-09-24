"""Command-line entry point: ``python -m stewbench <command>``.

    check        verify raw data, environment and GPU
    preprocess   build the cached window file
    neuro        task-vs-rest band-power statistics + topomaps
    benchmark    run the model benchmark (resumable)
    augment      run the augmentation study (resumable)
    report       build tables/figures for a finished experiment
    smoke        whole pipeline on a tiny synthetic fixture (sanity check)
    models       list registered models
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

from . import constants as C

warnings.filterwarnings("ignore", category=UserWarning)


def _config(args):
    from .settings import load_config
    return load_config(args.config)


def _out(args, config) -> Path:
    return Path(args.out) if args.out else C.OUTPUTS_DIR / config.name


def cmd_check(args):
    import torch
    from .data.raw import _read_recording, available_subjects, find_raw_dir, load_ratings
    from .provenance import environment
    env = environment()
    for key, value in env.items():
        print(f"{key:>14}: {value}")
    print(f"{'cuda available':>14}: {torch.cuda.is_available()}")
    try:
        raw = find_raw_dir(args.raw_dir)
    except FileNotFoundError as error:
        print(f"\n[MISSING] {error}")
        return 1
    subjects = available_subjects(raw)
    ok = subjects == list(range(1, C.N_SUBJECTS + 1))
    print(f"\nraw dir: {raw}")
    print(f"[{'OK' if ok else 'PROBLEM'}] subjects found: {len(subjects)} (expected {C.N_SUBJECTS})")
    ratings = load_ratings(raw)
    print(f"[{'OK' if ratings else 'MISSING'}] ratings.txt entries: {len(ratings)} "
          f"(subjects {list(C.MISSING_RATING_SUBJECTS)} have no rating by design)")
    lengths = {tag: _read_recording(raw / f"sub01_{tag}.txt").shape for tag in ("lo", "hi")}
    print(f"[{'OK' if all(s == (14, C.SAMPLES_PER_RECORDING) for s in lengths.values()) else 'PROBLEM'}] "
          f"sub01 shapes (channels, samples): {lengths}")
    return 0 if ok else 1


def cmd_preprocess(args):
    import numpy as np
    from .data.preprocess import cache_path, load_windows
    config = _config(args)
    windows = load_windows(config.preprocess, raw_dir=args.raw_dir)
    print(f"windows: {windows.X.shape}  class counts: {np.bincount(windows.y).tolist()}")
    for c, entry in windows.rejection["summary"].items():
        print(f"  class {c}: rejected {entry['rejected']}/{entry['total']} ({100 * entry['rate']:.2f}%)")
    print(f"cache: {cache_path(config.preprocess)}")
    return 0


def cmd_neuro(args):
    from .experiments.neuro import run_neuro
    from .reporting.report import neuro_report
    config = _config(args)
    out = Path(args.out) if args.out else C.OUTPUTS_DIR / "neurophysiology"
    run_neuro(config, out, raw_dir=args.raw_dir)
    neuro_report(out / "neurophysiology.json", out)
    return 0


def cmd_benchmark(args):
    from .experiments.benchmark import run_benchmark
    from .reporting.report import benchmark_report
    config = _config(args)
    out = _out(args, config)
    run_benchmark(config, out, device=args.device, raw_dir=args.raw_dir, models=args.models, verbose=args.verbose)
    benchmark_report(out)
    return 0


def cmd_augment(args):
    from .experiments.augmentation import run_augmentation
    from .reporting.report import augmentation_report
    config = _config(args)
    out = _out(args, config)
    run_augmentation(config, out, device=args.device, raw_dir=args.raw_dir, verbose=args.verbose)
    augmentation_report(out)
    return 0


def cmd_report(args):
    from .reporting.report import augmentation_report, benchmark_report, neuro_report
    target = Path(args.dir)
    if args.kind == "benchmark":
        benchmark_report(target)
    elif args.kind == "augment":
        augmentation_report(target)
    else:
        neuro_report(target / "neurophysiology.json", target)
    return 0


def cmd_smoke(args):
    """Exercise every stage on a tiny synthetic fixture (NOT real data)."""
    from .data.fixture import write_fixture
    from .experiments.augmentation import run_augmentation
    from .experiments.benchmark import run_benchmark
    from .experiments.neuro import run_neuro
    from .models import ALL_MODELS
    from .reporting.report import augmentation_report, benchmark_report, neuro_report
    from .settings import AugmentationConfig, ExperimentConfig, ProtocolConfig, TrainingConfig
    root = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="stewbench_smoke_"))
    raw = write_fixture(root / "fixture_raw", n_subjects=8, seconds=24)
    config = ExperimentConfig(
        name="smoke", seeds=[0], models=list(args.models or ALL_MODELS),
        protocol=ProtocolConfig(scheme="group_kfold", n_folds=4),
        training=TrainingConfig(max_epochs=2, patience=2, batch_size=32), tune_classical=False,
        augmentations=[AugmentationConfig("none"), AugmentationConfig("ft_surrogate"),
                       AugmentationConfig("cwgan_gp", synthetic_fraction=0.5, generator_epochs=1),
                       AugmentationConfig("ddpm", synthetic_fraction=0.5, generator_epochs=1)],
        train_subject_budgets=[None, 3],
    )
    run_benchmark(config, root / "benchmark", device=args.device, raw_dir=raw, fixture=True)
    benchmark_report(root / "benchmark")
    config.models = ["eegnet"]
    run_augmentation(config, root / "augment", device=args.device, raw_dir=raw, fixture=True)
    augmentation_report(root / "augment")
    run_neuro(config, root / "neuro", raw_dir=raw, fixture=True)
    neuro_report(root / "neuro" / "neurophysiology.json", root / "neuro")
    print(f"\nSMOKE TEST PASSED. Artifacts (synthetic, not real data): {root}")
    return 0


def cmd_models(args):
    from .models import SPECS
    for s in SPECS.values():
        flag = " [transductive]" if s.transductive else ""
        print(f"{s.name:18s} {s.kind:9s} {s.family:18s} {s.display:28s} {s.reference}{flag}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="stewbench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, fn, config=True, out=True):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if config:
            p.add_argument("--config", default=str(C.CONFIGS_DIR / "benchmark.yaml"))
        if out:
            p.add_argument("--out", default=None, help="output directory (default outputs/stew/<name>)")
        p.add_argument("--raw-dir", dest="raw_dir", default=None, type=Path)
        p.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto)")
        p.add_argument("--verbose", action="store_true")
        return p

    add("check", cmd_check, config=False, out=False)
    add("preprocess", cmd_preprocess, out=False)
    add("neuro", cmd_neuro)
    add("benchmark", cmd_benchmark).add_argument("--models", nargs="+", default=None,
                                                 help="subset of models (default: config)")
    add("augment", cmd_augment)
    rep = sub.add_parser("report")
    rep.set_defaults(fn=cmd_report)
    rep.add_argument("kind", choices=("benchmark", "augment", "neuro"))
    rep.add_argument("dir")
    add("smoke", cmd_smoke, config=False).add_argument("--models", nargs="+", default=None)
    sub.add_parser("models").set_defaults(fn=cmd_models)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
