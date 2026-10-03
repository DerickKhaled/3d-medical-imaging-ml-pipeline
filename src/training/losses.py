"""Dice + cross-entropy: the standard, well-understood loss for medical segmentation.

Cross-entropy gives stable per-voxel gradients; soft Dice counteracts the
heavy background/foreground imbalance of small anatomical structures.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DiceCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: (N, K, Z, Y, X); target: (N, Z, Y, X) integer labels."""
        cross_entropy = F.cross_entropy(logits, target)

        probabilities = logits.float().softmax(dim=1)[:, 1:]  # foreground classes only
        one_hot = F.one_hot(target, logits.shape[1]).permute(0, 4, 1, 2, 3).float()[:, 1:]
        dims = (0, 2, 3, 4)
        intersection = (probabilities * one_hot).sum(dims)
        denominator = probabilities.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return cross_entropy + (1 - dice.mean())
