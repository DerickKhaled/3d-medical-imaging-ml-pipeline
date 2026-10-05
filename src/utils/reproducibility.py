"""Seeds, device selection and code version: the inputs that make a run repeatable."""

from __future__ import annotations

import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every random number generator the pipeline uses."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Prefer deterministic kernels; warn (do not crash) where none exists.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine.")
    return torch.device(requested)


def code_version(repo_dir: Path | None = None) -> dict[str, object]:
    """Git commit of the code that produced an artifact, and whether the tree was dirty.

    A dirty tree is recorded, not refused: in research that is normal, but a
    release reviewer must be able to see it.
    """
    cwd = repo_dir or Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"git_commit": commit, "git_dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
