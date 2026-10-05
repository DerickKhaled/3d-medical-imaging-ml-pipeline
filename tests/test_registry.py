import shutil
from pathlib import Path

import pytest

from src.config import ReleaseConfig
from src.lineage.records import ArtifactPaths
from src.registry.model_registry import ModelRegistry, RegistryError

LENIENT = ReleaseConfig(required_split="test", min_mean_dice=0.0, min_cases=1)


@pytest.fixture
def registry(pipeline_run, tmp_path: Path) -> ModelRegistry:  # type: ignore[no-untyped-def]
    """A private copy of the pipeline artifacts, so tests can change registry state freely."""
    shutil.copytree(pipeline_run.paths.root, tmp_path / "artifacts")
    return ModelRegistry(ArtifactPaths(tmp_path / "artifacts"))


def test_registered_entry_is_complete(pipeline_run) -> None:  # type: ignore[no-untyped-def]
    entry = ModelRegistry(pipeline_run.paths).get("v1.0")
    for key in (
        "model_id",
        "model_version",
        "checkpoint_hash",
        "dataset_version",
        "preprocessing_version",
        "training_config",
        "evaluation_metrics",
        "created_at",
        "status",
    ):
        assert entry[key] is not None, key
    assert entry["status"] == "production"
    assert [h["to"] for h in entry["history"]] == ["candidate", "validated", "production"]
    assert ModelRegistry(pipeline_run.paths).get("production")["model_version"] == "v1.0"


def test_versions_are_immutable(registry: ModelRegistry) -> None:
    experiment_id = registry.get("v1.0")["experiment_id"]
    with pytest.raises(RegistryError, match="already exists"):
        registry.register(experiment_id, "v1.0")


def test_promotion_requires_evaluation_and_order(registry: ModelRegistry) -> None:
    registry.register(registry.get("v1.0")["experiment_id"], "v2.0")
    with pytest.raises(RegistryError, match="cannot move v2.0 from candidate to production"):
        registry.promote("v2.0", "production", "skip validation", LENIENT)
    with pytest.raises(RegistryError, match="no evaluation"):
        registry.promote("v2.0", "validated", "no evidence", LENIENT)
    with pytest.raises(RegistryError, match="written reason"):
        registry.promote("v2.0", "retired", "  ", LENIENT)


def test_release_gate_enforces_thresholds(registry: ModelRegistry, pipeline_run) -> None:  # type: ignore[no-untyped-def]
    registry.register(registry.get("v1.0")["experiment_id"], "v2.0")
    evaluation_file = Path(registry.get("v1.0")["evaluation_metrics"]["evaluation_file"])
    registry.attach_evaluation("v2.0", pipeline_run.evaluation, evaluation_file)

    strict = ReleaseConfig(required_split="test", min_mean_dice=0.999, min_cases=1000)
    with pytest.raises(RegistryError, match="release gate failed") as failure:
        registry.promote("v2.0", "validated", "try", strict)
    assert "mean Dice" in str(failure.value) and "cases" in str(failure.value)
    assert registry.get("v2.0")["status"] == "candidate"  # nothing changed


def test_new_production_model_retires_the_previous_one(
    registry: ModelRegistry, pipeline_run
) -> None:  # type: ignore[no-untyped-def]
    registry.register(registry.get("v1.0")["experiment_id"], "v2.0")
    evaluation_file = Path(registry.get("v1.0")["evaluation_metrics"]["evaluation_file"])
    registry.attach_evaluation("v2.0", pipeline_run.evaluation, evaluation_file)
    registry.promote("v2.0", "validated", "meets gate", LENIENT)
    registry.promote("v2.0", "production", "release 2", LENIENT)

    assert registry.get("production")["model_version"] == "v2.0"
    old = registry.get("v1.0")
    assert old["status"] == "retired"
    assert old["history"][-1]["reason"] == "replaced by v2.0"
