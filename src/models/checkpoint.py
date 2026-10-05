"""Save and load checkpoints.

A checkpoint holds the weights plus everything needed to use them: model
settings, class names, preprocessing and post-processing settings, and the
dataset and experiment it came from.

Loading uses torch.load(weights_only=True), so a checkpoint file can't run code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.config import ModelConfig
from src.models.unet3d import UNet3D, build_model
from src.utils.hashing import sha256_file


class CheckpointIntegrityError(RuntimeError):
    pass


def save_checkpoint(path: Path, model: torch.nn.Module, metadata: dict[str, Any]) -> str:
    """Save weights + metadata; return the SHA-256 of the written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    return sha256_file(path)


def load_checkpoint(
    path: Path, device: torch.device, expected_hash: str | None = None
) -> tuple[UNet3D, dict[str, Any]]:
    """Load a model in eval mode. If ``expected_hash`` is given, verify it first."""
    if expected_hash is not None:
        actual = sha256_file(path)
        if actual != expected_hash:
            raise CheckpointIntegrityError(
                f"{path} has hash {actual[:12]}..., expected {expected_hash[:12]}... "
                "(file changed after registration)"
            )
    payload = torch.load(path, map_location=device, weights_only=True)
    metadata = payload["metadata"]
    model = build_model(ModelConfig.model_validate(metadata["model_config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), metadata
