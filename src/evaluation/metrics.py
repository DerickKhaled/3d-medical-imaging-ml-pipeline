"""Dice, IoU, precision and recall for segmentation masks.

All four come from the same voxel counts (true positives, false positives,
false negatives). Edge cases:
- prediction and label both empty: Dice and IoU are 1.0
- precision with nothing predicted, or recall with nothing to find: None.
  None values are left out of averages instead of being counted as 0 or 1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class OverlapCounts:
    tp: int
    fp: int
    fn: int

    @property
    def dice(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        return 1.0 if denominator == 0 else 2 * self.tp / denominator

    @property
    def iou(self) -> float:
        denominator = self.tp + self.fp + self.fn
        return 1.0 if denominator == 0 else self.tp / denominator

    @property
    def precision(self) -> float | None:
        return None if self.tp + self.fp == 0 else self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return None if self.tp + self.fn == 0 else self.tp / (self.tp + self.fn)

    def as_metrics(self) -> dict[str, float | int | None]:
        return {
            "dice": self.dice,
            "iou": self.iou,
            "precision": self.precision,
            "recall": self.recall,
            **asdict(self),
        }


def overlap_counts(prediction: np.ndarray, reference: np.ndarray, label: int) -> OverlapCounts:
    if prediction.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: prediction {prediction.shape} vs reference {reference.shape}"
        )
    pred, ref = prediction == label, reference == label
    return OverlapCounts(
        tp=int(np.count_nonzero(pred & ref)),
        fp=int(np.count_nonzero(pred & ~ref)),
        fn=int(np.count_nonzero(~pred & ref)),
    )


def per_class_metrics(
    prediction: np.ndarray, reference: np.ndarray, class_names: dict[int, str]
) -> dict[str, dict[str, float | int | None]]:
    """Metrics for every foreground class (label 0 = background is skipped)."""
    return {
        name: overlap_counts(prediction, reference, label).as_metrics()
        for label, name in sorted(class_names.items())
        if label != 0
    }


def mean_ignoring_none(values: list[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None
