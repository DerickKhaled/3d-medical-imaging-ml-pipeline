from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.config import IntensityConfig, PreprocessingConfig
from src.data.loaders import load_nifti
from src.preprocessing.transforms import (
    crop_or_pad,
    normalize_intensity,
    preprocess_image,
    preprocess_label,
    resample,
    restore_to_original,
)
from src.utils.hashing import sha256_array


def _case(root: Path, name: str = "synthetic_001.nii.gz") -> tuple[sitk.Image, sitk.Image]:
    return (
        load_nifti(root / "imagesTr" / name).image,
        load_nifti(root / "labelsTr" / name).image,
    )


def test_preprocessing_is_deterministic(
    synthetic_root: Path, preprocessing_config: PreprocessingConfig
) -> None:
    image, _ = _case(synthetic_root)
    first, _ = preprocess_image(image, preprocessing_config)
    second, _ = preprocess_image(image, preprocessing_config)
    assert sha256_array(first) == sha256_array(second)


def test_output_shape_and_normalisation(
    synthetic_root: Path, preprocessing_config: PreprocessingConfig
) -> None:
    image, label = _case(synthetic_root)
    array, _ = preprocess_image(image, preprocessing_config)
    expected_zyx = tuple(reversed(preprocessing_config.crop_or_pad_size_xyz))
    assert array.shape == (1, *expected_zyx)
    assert array.dtype == np.float32
    assert preprocess_label(label, preprocessing_config).shape == expected_zyx


def test_resampling_respects_physical_size() -> None:
    image = sitk.GetImageFromArray(np.random.default_rng(0).random((20, 30, 40), dtype=np.float32))
    image.SetSpacing((1.0, 1.0, 1.0))
    coarse = resample(image, (2.0, 2.0, 2.0), is_label=False)
    assert coarse.GetSize() == (20, 15, 10)  # 40x30x20 mm at 2 mm
    assert coarse.GetSpacing() == (2.0, 2.0, 2.0)


def test_label_resampling_keeps_integer_classes() -> None:
    label = sitk.GetImageFromArray(np.random.default_rng(0).integers(0, 3, (10, 10, 10)).astype(np.uint8))
    resampled = resample(label, (0.5, 0.5, 0.5), is_label=True)
    assert set(np.unique(sitk.GetArrayFromImage(resampled))) <= {0, 1, 2}


def test_crop_or_pad_handles_both_directions() -> None:
    array = np.arange(5 * 8 * 3).reshape(5, 8, 3)
    out, crop_start, pad_before = crop_or_pad(array, (7, 6, 3))
    assert out.shape == (7, 6, 3)
    assert crop_start == (0, 1, 0) and pad_before == (1, 0, 0)
    np.testing.assert_array_equal(out[1:6], array[:, 1:7])


def test_prediction_maps_back_onto_original_scan(
    synthetic_root: Path, preprocessing_config: PreprocessingConfig
) -> None:
    """A 'perfect prediction' (the preprocessed label) must restore to the original label."""
    image, label = _case(synthetic_root)
    _, grid = preprocess_image(image, preprocessing_config)
    restored = restore_to_original(preprocess_label(label, preprocessing_config), grid, image)
    assert restored.GetSize() == image.GetSize()
    np.testing.assert_array_equal(sitk.GetArrayFromImage(restored), sitk.GetArrayFromImage(label))


def test_restore_undoes_reorientation(
    synthetic_root: Path, preprocessing_config: PreprocessingConfig
) -> None:
    image, label = _case(synthetic_root)
    flipped_config = preprocessing_config.model_copy(update={"orientation": "LPI"})
    _, grid = preprocess_image(image, flipped_config)
    restored = restore_to_original(preprocess_label(label, flipped_config), grid, image)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(restored), sitk.GetArrayFromImage(label))


def test_ct_window_maps_hounsfield_range_to_unit_interval() -> None:
    window = IntensityConfig(
        mode="ct_window", clip_percentiles=None, ct_window_level=40, ct_window_width=400
    )
    hu = np.array([-1000.0, -160.0, 40.0, 240.0, 3000.0])
    np.testing.assert_allclose(normalize_intensity(hu, window), [0.0, 0.0, 0.5, 1.0, 1.0])
