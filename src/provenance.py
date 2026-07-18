"""Reproducibility metadata and stable checksums for research runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch

from . import config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    """Hash relative paths and file contents in stable lexical order."""
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.name != ".gitkeep")
    for file_path in files:
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(file_sha256(file_path).encode("ascii"))
    return digest.hexdigest()


def _git(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *command],
            cwd=config.PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_provenance() -> dict:
    commit = _git(["rev-parse", "HEAD"])
    status = _git(["status", "--porcelain"])
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "dirty_file_count": len(status.splitlines()) if status else 0,
    }


def runtime_versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
    }


def preprocessing_snapshot(dataset: str) -> dict:
    spec = config.dataset_spec(dataset)
    return {
        "sampling_rate_hz": spec.sampling_rate_hz,
        "channels": list(spec.channels),
        "bandpass_hz": [config.BANDPASS_LOW_HZ, config.BANDPASS_HIGH_HZ],
        "notch_hz": config.NOTCH_FREQ_HZ,
        "filter_order": config.FILTER_ORDER,
        "window_seconds": config.WINDOW_SECONDS,
        "window_overlap": config.WINDOW_OVERLAP,
        "window_samples": config.WINDOW_SAMPLES,
        "artifact_rejection": config.APPLY_ARTIFACT_REJECTION,
        "artifact_mad_multiplier": config.artifact_rejection_mad_multiplier(dataset),
        "normalization": config.STEW_NORMALIZATION if dataset == "stew" else config.NORMALIZATION,
    }


def build_provenance(dataset: str, split_path: Path | None = None) -> dict:
    raw_path = config.raw_dir(dataset)
    return {
        "git": git_provenance(),
        "runtime": runtime_versions(),
        "dataset_spec": config.dataset_spec_dict(dataset),
        "preprocessing": preprocessing_snapshot(dataset),
        "raw_data_sha256": directory_sha256(raw_path),
        "split_sha256": file_sha256(split_path) if split_path and split_path.exists() else None,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
