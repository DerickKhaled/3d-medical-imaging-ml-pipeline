from pathlib import Path

import pytest
import torch

from src.config import ModelConfig
from src.lineage.records import ArtifactPaths, read_json
from src.models.checkpoint import CheckpointIntegrityError, load_checkpoint, save_checkpoint
from src.models.unet3d import build_model, count_parameters
from src.training.train import train
from src.utils.hashing import sha256_file

from .pipeline_fixture import write_smoke_experiment


def _config(num_classes: int = 3) -> ModelConfig:
    return ModelConfig(name="unet3d", in_channels=1, num_classes=num_classes, base_channels=4, depth=2)


@pytest.mark.parametrize("num_classes", [2, 3, 5])
def test_forward_pass_shape(num_classes: int) -> None:
    model = build_model(_config(num_classes))
    logits = model(torch.randn(2, 1, 16, 24, 16))
    assert logits.shape == (2, num_classes, 16, 24, 16)


def test_demo_model_is_compact() -> None:
    config = ModelConfig(name="unet3d", in_channels=1, num_classes=3, base_channels=16, depth=3)
    assert count_parameters(build_model(config)) == 1_401_299


def test_checkpoint_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_model(_config()).eval()
    path = tmp_path / "model.pt"
    checkpoint_hash = save_checkpoint(path, model, {"model_config": _config().model_dump()})

    loaded, metadata = load_checkpoint(path, torch.device("cpu"), expected_hash=checkpoint_hash)
    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(model(x), loaded(x))
    assert metadata["model_config"]["num_classes"] == 3

    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(CheckpointIntegrityError):
        load_checkpoint(path, torch.device("cpu"), expected_hash=checkpoint_hash)


def test_experiment_record_contents(pipeline_run) -> None:  # type: ignore[no-untyped-def]
    record = pipeline_run.experiment
    for key in ("experiment_id", "dataset_version", "preprocessing_version", "model_name",
                "model_config", "training_config", "random_seed", "code_version",
                "best_checkpoint", "metrics", "timestamp"):
        assert key in record, key
    run_dir = pipeline_run.paths.experiment(record["experiment_id"])
    assert sha256_file(run_dir / "best.pt") == record["best_checkpoint"]["checkpoint_hash"]
    history = read_json(run_dir / "history.json")["epochs"]
    assert len(history) == record["training_config"]["epochs"]


@pytest.mark.slow
def test_training_is_reproducible(tmp_path: Path, synthetic_root: Path) -> None:
    """Same config + same data + same seed -> identical loss curve and identical weights."""
    config = write_smoke_experiment(tmp_path, synthetic_root)
    first = train(config, ArtifactPaths(tmp_path / "run_a"))
    second = train(config, ArtifactPaths(tmp_path / "run_b"))
    history_a = read_json(tmp_path / "run_a/experiments/EXP-001/history.json")["epochs"]
    history_b = read_json(tmp_path / "run_b/experiments/EXP-001/history.json")["epochs"]
    assert [e["train_loss"] for e in history_a] == [e["train_loss"] for e in history_b]
    assert first["best_checkpoint"]["checkpoint_hash"] == second["best_checkpoint"]["checkpoint_hash"]
