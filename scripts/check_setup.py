"""
Audits the research project layout for one dataset.

Usage:
    python scripts/check_setup.py --dataset dreamer
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


def status(label: str, ok: bool, detail: str) -> None:
    mark = "OK" if ok else "MISSING"
    print(f"[{mark:<7}] {label:<24} {detail}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=config.DEFAULT_DATASET, choices=config.SUPPORTED_DATASETS)
    args = parser.parse_args()

    dataset = config.normalize_dataset_name(args.dataset)
    raw_dir = config.raw_dir(dataset)
    processed_dir = config.processed_dir(dataset)
    model_dir = config.model_dir(dataset)
    output_dir = config.output_dir(dataset)
    run_dir = config.run_dir(dataset)

    print(f"Dataset setup audit: {dataset}")
    print(f"Project root: {PROJECT_ROOT}\n")

    raw_files = [p for p in raw_dir.glob("*") if p.name != ".gitkeep"]
    processed_files = sorted(processed_dir.glob("subject_*.npz"))
    legacy_processed_files = sorted(config.DATA_PROCESSED.glob("subject_*.npz")) if dataset == "dreamer" else []
    split_file = processed_dir / "split.json"
    legacy_split_file = config.DATA_PROCESSED / "split.json"
    synth_files = sorted(processed_dir.glob("synthetic_train*.npz"))
    legacy_synth_files = sorted(config.DATA_PROCESSED.glob("synthetic_train*.npz")) if dataset == "dreamer" else []
    checkpoint_files = sorted(model_dir.glob("**/*.pt"))
    output_files = [p for p in output_dir.glob("**/*") if p.is_file() and p.name != ".gitkeep"]

    status("raw directory", raw_dir.exists(), str(raw_dir))
    status("raw files", bool(raw_files), f"{len(raw_files)} file(s)")
    status("processed subjects", bool(processed_files or legacy_processed_files),
           f"{len(processed_files)} dataset-folder file(s), {len(legacy_processed_files)} legacy file(s)")
    status("split", split_file.exists() or (dataset == "dreamer" and legacy_split_file.exists()),
           str(split_file if split_file.exists() else legacy_split_file))
    status("synthetic GAN data", bool(synth_files or legacy_synth_files),
           f"{len(synth_files)} dataset-folder file(s), {len(legacy_synth_files)} legacy file(s)")
    status("model checkpoints", bool(checkpoint_files), f"{len(checkpoint_files)} .pt file(s) under {model_dir}")
    status("outputs", bool(output_files), f"{len(output_files)} file(s) under {output_dir}")
    status("runs directory", run_dir.exists(), str(run_dir))

    print("\nRecommended commands:")
    print(f"  python -m src.preprocessing --dataset {dataset}")
    print(f"  python -m src.split --dataset {dataset}")
    print(f"  python -m src.train_gan --dataset {dataset} --run_name gan_400epoch --epochs 400")
    print(f"  python -m src.train_baseline_single --dataset {dataset} --model eegnet --epochs 30")
    print(f"  python -m src.train_baseline_single --dataset {dataset} --model eegnet --use_gan --gan_run gan_400epoch --epochs 30")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
