"""PyTorch dataset that reads the preprocessed scans, with augmentation.

The random augmentation is seeded from (seed, epoch, sample index), so a training
run gives the same result no matter how many DataLoader workers are used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import AugmentationConfig


class ProcessedVolumes(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        files: list[Path],
        augmentation: AugmentationConfig | None,
        seed: int,
    ) -> None:
        self.files = files
        self.augmentation = augmentation
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with np.load(self.files[index]) as data:
            image, label = data["image"].copy(), data["label"].copy()
        if self.augmentation is not None:
            rng = np.random.default_rng([self.seed, self.epoch, index])
            image, label = augment(image, label, self.augmentation, rng)
        return torch.from_numpy(image), torch.from_numpy(label)


def augment(
    image: np.ndarray, label: np.ndarray, config: AugmentationConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """image: (C, Z, Y, X) float32, label: (Z, Y, X) int64."""
    for axis_xyz in config.flip_axes_xyz:
        if rng.random() < 0.5:
            image = np.flip(image, axis=3 - axis_xyz)  # x->3, y->2, z->1 in (C, Z, Y, X)
            label = np.flip(label, axis=2 - axis_xyz)  # x->2, y->1, z->0 in (Z, Y, X)
    scale = 1.0 + rng.uniform(-config.intensity_scale, config.intensity_scale)
    shift = rng.uniform(-config.intensity_shift, config.intensity_shift)
    image = image * scale + shift
    if config.noise_std > 0:
        image = image + rng.normal(0.0, config.noise_std, size=image.shape)
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(label)
