"""Run the complete pipeline once on synthetic data; shared by the end-to-end tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.config import MeshConfig, ReleaseConfig, load_config
from src.data.manifest import load_manifest
from src.data.synthetic import write_dataset
from src.evaluation.evaluate import evaluate
from src.inference.predict import run_inference
from src.lineage.records import ArtifactPaths
from src.reconstruction.mesh import build_meshes
from src.registry.model_registry import ModelRegistry
from src.training.train import train

from .conftest import REPO_ROOT, SMOKE_CONFIGS


@dataclass
class PipelineRun:
    root: Path
    paths: ArtifactPaths
    data_root: Path
    experiment: dict[str, Any]
    evaluation: dict[str, Any]
    inference: dict[str, Any]
    meshes: dict[str, Any]
    test_scan: Path


def write_smoke_experiment(root: Path, data_root: Path) -> Path:
    data = yaml.safe_load((SMOKE_CONFIGS / "data.yaml").read_text())
    data["root"] = str(data_root)
    (root / "data.yaml").write_text(yaml.safe_dump(data))
    experiment = yaml.safe_load((SMOKE_CONFIGS / "train.yaml").read_text())
    experiment["data_config"] = str(root / "data.yaml")
    experiment["preprocessing_config"] = str(SMOKE_CONFIGS / "preprocessing.yaml")
    path = root / "train.yaml"
    path.write_text(yaml.safe_dump(experiment))
    return path


def run_pipeline(root: Path) -> PipelineRun:
    data_root = root / "synthetic"
    write_dataset(data_root, cases=12, seed=0)
    paths = ArtifactPaths(root / "artifacts")

    experiment = train(write_smoke_experiment(root, data_root), paths)
    registry = ModelRegistry(paths)
    registry.register(experiment["experiment_id"], "v1.0")
    evaluation, _ = evaluate("v1.0", "test", paths, data_root=None, device_name="cpu")
    release = load_config(SMOKE_CONFIGS / "release.yaml", ReleaseConfig)
    registry.promote("v1.0", "validated", "smoke test evidence", release)
    registry.promote("v1.0", "production", "smoke test release", release)

    test_case = evaluation["cases"][0]["sample_id"]
    records = load_manifest(paths, experiment["dataset_version"])["records"]
    test_scan = data_root / next(r for r in records if r["sample_id"] == test_case)["source_path"]
    inference = run_inference(test_scan, "production", paths, "cpu")
    meshes = build_meshes(
        inference["inference_id"], paths, load_config(REPO_ROOT / "configs/mesh.yaml", MeshConfig)
    )
    return PipelineRun(root, paths, data_root, experiment, evaluation, inference, meshes, test_scan)
