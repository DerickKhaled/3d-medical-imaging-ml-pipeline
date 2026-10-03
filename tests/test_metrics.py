import numpy as np
import pytest

from src.evaluation.metrics import mean_ignoring_none, overlap_counts, per_class_metrics


def test_known_overlap() -> None:
    reference = np.zeros((4, 4, 4), np.uint8)
    reference[:2] = 1                      # 32 voxels
    prediction = np.zeros_like(reference)
    prediction[1:3] = 1                    # 32 voxels, 16 overlapping
    counts = overlap_counts(prediction, reference, label=1)
    assert (counts.tp, counts.fp, counts.fn) == (16, 16, 16)
    assert counts.dice == pytest.approx(0.5)
    assert counts.iou == pytest.approx(1 / 3)
    assert counts.precision == pytest.approx(0.5)
    assert counts.recall == pytest.approx(0.5)


def test_perfect_and_empty_cases() -> None:
    mask = np.zeros((3, 3, 3), np.uint8)
    mask[1, 1, 1] = 2
    perfect = overlap_counts(mask, mask, label=2)
    assert perfect.dice == perfect.iou == perfect.precision == perfect.recall == 1.0

    both_empty = overlap_counts(mask, mask, label=1)
    assert both_empty.dice == 1.0 and both_empty.iou == 1.0       # agreement on absence
    assert both_empty.precision is None and both_empty.recall is None  # undefined, not faked

    missed = overlap_counts(np.zeros_like(mask), mask, label=2)
    assert missed.dice == 0.0 and missed.recall == 0.0 and missed.precision is None


def test_per_class_skips_background_and_checks_shape() -> None:
    mask = np.ones((2, 2, 2), np.uint8)
    metrics = per_class_metrics(mask, mask, {0: "background", 1: "a", 2: "b"})
    assert list(metrics) == ["a", "b"]
    with pytest.raises(ValueError, match="shape mismatch"):
        overlap_counts(mask, np.ones((2, 2, 3), np.uint8), label=1)


def test_mean_ignores_undefined_values() -> None:
    assert mean_ignoring_none([1.0, None, 0.5]) == pytest.approx(0.75)
    assert mean_ignoring_none([None, None]) is None
