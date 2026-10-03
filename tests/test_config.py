from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import (
    DataConfig,
    PreprocessingConfig,
    ReleaseConfig,
    load_config,
    load_experiment,
)

from .conftest import REPO_ROOT


@pytest.mark.parametrize(
    ("relative_path", "schema"),
    [
        ("configs/data.yaml", DataConfig),
        ("configs/preprocessing.yaml", PreprocessingConfig),
        ("configs/release.yaml", ReleaseConfig),
    ],
)
def test_shipped_configs_are_valid(relative_path: str, schema: type) -> None:
    load_config(REPO_ROOT / relative_path, schema)


def test_shipped_experiments_are_consistent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    load_experiment("configs/train.yaml")
    load_experiment("configs/smoke/train.yaml")


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _preprocessing_dict() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs/preprocessing.yaml").read_text())


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    data = _preprocessing_dict() | {"target_spacng_mm": [1, 1, 1]}  # typo must not pass silently
    with pytest.raises(ValidationError, match="target_spacng_mm"):
        load_config(_write(tmp_path, data), PreprocessingConfig)


def test_missing_key_is_rejected(tmp_path: Path) -> None:
    data = _preprocessing_dict()
    del data["orientation"]
    with pytest.raises(ValidationError, match="orientation"):
        load_config(_write(tmp_path, data), PreprocessingConfig)


@pytest.mark.parametrize("code", ["RAA", "XYZ", "RA", "RASL"])
def test_invalid_orientation_is_rejected(tmp_path: Path, code: str) -> None:
    data = _preprocessing_dict() | {"orientation": code}
    with pytest.raises(ValidationError, match="orientation"):
        load_config(_write(tmp_path, data), PreprocessingConfig)


def test_ct_window_mode_requires_window(tmp_path: Path) -> None:
    data = _preprocessing_dict()
    data["intensity"]["mode"] = "ct_window"
    with pytest.raises(ValidationError, match="ct_window_level"):
        load_config(_write(tmp_path, data), PreprocessingConfig)


def test_preprocessing_version_tracks_content(tmp_path: Path) -> None:
    base = load_config(REPO_ROOT / "configs/preprocessing.yaml", PreprocessingConfig)
    same = load_config(REPO_ROOT / "configs/preprocessing.yaml", PreprocessingConfig)
    changed = base.model_copy(update={"target_spacing_mm": (1.5, 1.5, 1.5)})
    assert base.version == same.version
    assert base.version != changed.version
    assert base.version.startswith("pp-hippo-mri-")


def test_class_count_mismatch_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    experiment = yaml.safe_load((REPO_ROOT / "configs/smoke/train.yaml").read_text())
    experiment["model"]["num_classes"] = 5
    with pytest.raises(ValueError, match="num_classes"):
        load_experiment(_write(tmp_path, experiment))
