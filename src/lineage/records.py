"""Where lineage records live and how they are written.

Every stage writes a plain JSON record. Records reference each other by ID and
by content hash, which is what ``src.lineage.trace`` walks:

    inference record ──model_version──▶ registry entry ──experiment_id──▶ experiment record
            │                                 │                                  │
       input_hash                     checkpoint_hash                 dataset_version / preprocessing_version
            │                                                                    │
            └──────────────── found in ──▶ dataset manifest ◀────────────────────┘

There is no database: the artifacts directory *is* the store. It can be
archived, diffed and reviewed with ordinary tools.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactPaths:
    """The single definition of the artifacts directory layout."""

    root: Path

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    def manifest(self, dataset_version: str) -> Path:
        return self.manifests / f"{dataset_version}.json"

    def processed(self, dataset_version: str, preprocessing_version: str) -> Path:
        return self.root / "processed" / dataset_version / preprocessing_version

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    def experiment(self, experiment_id: str) -> Path:
        return self.experiments / experiment_id

    @property
    def evaluations(self) -> Path:
        return self.root / "evaluations"

    @property
    def registry(self) -> Path:
        return self.root / "registry"

    @property
    def inference(self) -> Path:
        return self.root / "inference"

    def inference_run(self, inference_id: str) -> Path:
        return self.inference / inference_id


def write_json(path: Path, record: dict[str, Any]) -> Path:
    """Write atomically (temp file + rename) so a crash never leaves half a record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=False, default=str)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"record not found: {path}")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def next_experiment_id(paths: ArtifactPaths) -> str:
    """Sequential, human-friendly IDs (EXP-001, EXP-002, ...) for discussion in reviews."""
    existing = [p.name for p in paths.experiments.glob("EXP-*") if p.is_dir()]
    numbers = [int(name.split("-")[1]) for name in existing if name.split("-")[1].isdigit()]
    return f"EXP-{max(numbers, default=0) + 1:03d}"


def new_run_id(prefix: str) -> str:
    """Unique ID for high-volume records (inference, evaluation): date + random suffix."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"
