"""Save the preprocessed scans so training doesn't redo the work every epoch.

    artifacts/processed/<dataset_version>/<preprocessing_version>/
        <sample_id>.npz   image and label
        index.json        which source scan became which processed file

The folder name contains both versions, so you always know which data and
which settings produced a file. Existing files are never overwritten.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import DataConfig, PreprocessingConfig
from src.data.loaders import load_nifti
from src.lineage.records import ArtifactPaths, read_json, write_json
from src.preprocessing.transforms import preprocess_image, preprocess_label
from src.utils.hashing import sha256_array
from src.utils.logging import get_logger
from src.utils.reproducibility import utc_now

log = get_logger(__name__)


def ensure_processed(
    manifest: dict[str, Any],
    data_config: DataConfig,
    preprocessing: PreprocessingConfig,
    paths: ArtifactPaths,
) -> dict[str, Any]:
    out_dir = paths.processed(manifest["dataset_version"], preprocessing.version)
    index_path = out_dir / "index.json"
    if index_path.exists():
        return read_json(index_path)

    log.info("preprocessing %d samples -> %s", len(manifest["records"]), out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for record in manifest["records"]:
        image = load_nifti(data_config.root / record["source_path"])
        label = load_nifti(data_config.root / record["label_path"])
        if image.source_hash != record["source_hash"]:
            raise ValueError(
                f"{record['source_path']} changed on disk since the manifest was built"
            )
        image_array, _ = preprocess_image(image.image, preprocessing)
        label_array = preprocess_label(label.image, preprocessing)
        np.savez_compressed(
            out_dir / f"{record['sample_id']}.npz", image=image_array, label=label_array
        )
        samples.append(
            {
                "sample_id": record["sample_id"],
                "split": record["split"],
                "source_hash": record["source_hash"],
                "processed_hash": sha256_array(image_array),
                "label_processed_hash": sha256_array(label_array),
            }
        )

    index = {
        "dataset_version": manifest["dataset_version"],
        "preprocessing_version": preprocessing.version,
        "preprocessing_config": preprocessing.model_dump(mode="json"),
        "created_at": utc_now(),
        "samples": samples,
    }
    write_json(index_path, index)
    return index


def sample_file(paths: ArtifactPaths, index: dict[str, Any], sample_id: str) -> Path:
    return (
        paths.processed(index["dataset_version"], index["preprocessing_version"])
        / f"{sample_id}.npz"
    )
