"""Run provenance: code revision, environment, hardware and data checksums."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import constants as C


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=C.PROJECT_ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_data_sha256(raw_dir: Path | None = None) -> str | None:
    raw_dir = Path(raw_dir or C.RAW_DIR)
    files = sorted(p for p in raw_dir.rglob("*.txt"))
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(file_sha256(path).encode())
    return digest.hexdigest()


def environment() -> dict:
    import numpy
    import scipy
    import sklearn
    import torch
    info = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": numpy.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import pyriemann
        info["pyriemann"] = pyriemann.__version__
    except ImportError:
        pass
    return info


def provenance(raw_dir: Path | None = None) -> dict:
    status = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "environment": environment(),
        "raw_data_sha256": raw_data_sha256(raw_dir),
    }
