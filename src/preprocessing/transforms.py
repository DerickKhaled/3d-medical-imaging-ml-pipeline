"""Deterministic preprocessing, driven entirely by ``PreprocessingConfig``.

Pipeline for an image:

    reorient  ->  resample to target spacing  ->  intensity normalisation  ->  crop/pad

Labels go through the same geometric steps with nearest-neighbour interpolation.
``GridRecord`` keeps what is needed to map a prediction back onto the original
scan, so a segmentation is always delivered in the geometry of the input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from src.config import IntensityConfig, PreprocessingConfig


@dataclass(frozen=True)
class GridRecord:
    """Geometry needed to invert preprocessing for one image."""

    resampled_reference: sitk.Image  # empty image carrying the resampled grid's geometry
    crop_start_zyx: tuple[int, int, int]
    pad_before_zyx: tuple[int, int, int]
    resampled_shape_zyx: tuple[int, int, int]


# --- geometric steps ------------------------------------------------------------


def reorient(image: sitk.Image, orientation: str) -> sitk.Image:
    return sitk.DICOMOrient(image, orientation)


def resample(
    image: sitk.Image, spacing_xyz: tuple[float, float, float], is_label: bool
) -> sitk.Image:
    old_spacing = np.asarray(image.GetSpacing())
    old_size = np.asarray(image.GetSize())
    new_size = np.maximum(np.round(old_size * old_spacing / np.asarray(spacing_xyz)), 1)
    return sitk.Resample(
        image,
        [int(s) for s in new_size],
        sitk.Transform(),
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear,
        image.GetOrigin(),
        spacing_xyz,
        image.GetDirection(),
        0,
        image.GetPixelID() if is_label else sitk.sitkFloat32,
    )


def crop_or_pad(
    array: np.ndarray, target_zyx: tuple[int, int, int]
) -> tuple[np.ndarray, tuple[int, int, int], tuple[int, int, int]]:
    """Centre-crop or zero-pad each axis to ``target_zyx``.

    Returns the new array plus the crop start and pad size per axis, which is
    everything needed to undo the operation.
    """
    crop_start, pad_before, slices, pads = [], [], [], []
    for length, target in zip(array.shape, target_zyx, strict=True):
        start = max((length - target) // 2, 0)
        kept = min(length, target)
        before = (target - kept) // 2
        crop_start.append(start)
        pad_before.append(before)
        slices.append(slice(start, start + kept))
        pads.append((before, target - kept - before))
    out = np.pad(array[tuple(slices)], pads, mode="constant", constant_values=0)
    return out, tuple(crop_start), tuple(pad_before)


def undo_crop_or_pad(array: np.ndarray, grid: GridRecord) -> np.ndarray:
    out = np.zeros(grid.resampled_shape_zyx, dtype=array.dtype)
    src, dst = [], []
    for length, target, start, before in zip(
        grid.resampled_shape_zyx,
        array.shape,
        grid.crop_start_zyx,
        grid.pad_before_zyx,
        strict=True,
    ):
        kept = min(length, target)
        src.append(slice(before, before + kept))
        dst.append(slice(start, start + kept))
    out[tuple(dst)] = array[tuple(src)]
    return out


# --- intensity ------------------------------------------------------------------


def normalize_intensity(array: np.ndarray, config: IntensityConfig) -> np.ndarray:
    array = array.astype(np.float32)
    if config.mode == "ct_window":
        assert config.ct_window_level is not None and config.ct_window_width is not None
        low = config.ct_window_level - config.ct_window_width / 2
        high = config.ct_window_level + config.ct_window_width / 2
        return ((np.clip(array, low, high) - low) / (high - low)).astype(np.float32)

    assert config.clip_percentiles is not None
    low, high = np.percentile(array, config.clip_percentiles)
    clipped = np.clip(array, low, high)
    std = float(clipped.std())
    return ((clipped - float(clipped.mean())) / (std if std > 0 else 1.0)).astype(np.float32)


# --- full pipeline ----------------------------------------------------------------


def preprocess_image(
    image: sitk.Image, config: PreprocessingConfig
) -> tuple[np.ndarray, GridRecord]:
    """Return a (1, Z, Y, X) float32 array ready for the network, plus its grid record."""
    resampled = resample(reorient(image, config.orientation), config.target_spacing_mm, False)
    array = normalize_intensity(sitk.GetArrayFromImage(resampled), config.intensity)
    target_zyx = tuple(reversed(config.crop_or_pad_size_xyz))
    fitted, crop_start, pad_before = crop_or_pad(array, target_zyx)  # type: ignore[arg-type]

    reference = sitk.Image(resampled.GetSize(), sitk.sitkUInt8)
    reference.CopyInformation(resampled)
    grid = GridRecord(reference, crop_start, pad_before, array.shape)
    return fitted[np.newaxis].astype(np.float32), grid


def preprocess_label(label: sitk.Image, config: PreprocessingConfig) -> np.ndarray:
    """Return a (Z, Y, X) int64 label array aligned with ``preprocess_image`` output."""
    resampled = resample(reorient(label, config.orientation), config.target_spacing_mm, True)
    target_zyx = tuple(reversed(config.crop_or_pad_size_xyz))
    fitted, _, _ = crop_or_pad(sitk.GetArrayFromImage(resampled), target_zyx)  # type: ignore[arg-type]
    return fitted.astype(np.int64)


def restore_to_original(mask_zyx: np.ndarray, grid: GridRecord, original: sitk.Image) -> sitk.Image:
    """Map a mask from network space back onto the original scan's voxel grid."""
    on_resampled = sitk.GetImageFromArray(undo_crop_or_pad(mask_zyx.astype(np.uint8), grid))
    on_resampled.CopyInformation(grid.resampled_reference)
    # Resampling in physical space also undoes the reorientation.
    return sitk.Resample(
        on_resampled, original, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
    )
