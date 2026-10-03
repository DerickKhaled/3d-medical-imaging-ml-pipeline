"""Shared fixtures. Tests use synthetic NIfTI data written to a temp directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import DataConfig, PreprocessingConfig, load_config
from src.data.synthetic import write_dataset
from src.lineage.records import ArtifactPaths

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIGS = REPO_ROOT / "configs" / "smoke"


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic"
    write_dataset(root, cases=10, seed=0)
    return root


@pytest.fixture
def data_config(synthetic_root: Path) -> DataConfig:
    config = load_config(SMOKE_CONFIGS / "data.yaml", DataConfig)
    return config.model_copy(update={"root": synthetic_root})


@pytest.fixture
def preprocessing_config() -> PreprocessingConfig:
    return load_config(SMOKE_CONFIGS / "preprocessing.yaml", PreprocessingConfig)


@pytest.fixture
def artifact_paths(tmp_path: Path) -> ArtifactPaths:
    return ArtifactPaths(tmp_path / "artifacts")


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    """Train -> register -> evaluate -> promote -> infer -> mesh, once per test session."""
    from .pipeline_fixture import run_pipeline

    return run_pipeline(tmp_path_factory.mktemp("pipeline"))
