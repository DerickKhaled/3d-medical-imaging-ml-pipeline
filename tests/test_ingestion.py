"""Loading, validation and the dataset manifest."""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from src.config import DataConfig
from src.data.loaders import Volume, VolumeLoadError, load_dicom_series, load_nifti, save_nifti
from src.data.manifest import assign_split, build_manifest
from src.data.synthetic import write_dataset
from src.data.validation import InvalidVolumeError, image_problems, label_problems, validate_image


def _volume(tmp_path: Path, array: np.ndarray, spacing=(1.0, 1.0, 1.0), name="v.nii.gz"):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    return load_nifti(save_nifti(image, tmp_path / name))


# --- loading ---------------------------------------------------------------------


def test_load_nifti_reads_geometry_and_hash(synthetic_root: Path) -> None:
    volume = load_nifti(synthetic_root / "imagesTr" / "synthetic_001.nii.gz")
    assert len(volume.size_xyz) == 3
    assert volume.spacing_xyz_mm == (1.0, 1.0, 1.0)
    assert volume.to_numpy().shape == tuple(reversed(volume.size_xyz))  # numpy is (z, y, x)
    assert len(volume.source_hash) == 64


def test_missing_and_corrupt_files_raise_load_error(tmp_path: Path) -> None:
    with pytest.raises(VolumeLoadError, match="does not exist"):
        load_nifti(tmp_path / "missing.nii.gz")
    corrupt = tmp_path / "corrupt.nii.gz"
    corrupt.write_bytes(b"this is not a nifti file")
    with pytest.raises(VolumeLoadError, match="unreadable"):
        load_nifti(corrupt)
    other = tmp_path / "scan.png"
    other.write_bytes(b"x")
    with pytest.raises(VolumeLoadError, match="not a NIfTI"):
        load_nifti(other)


def test_dicom_series_adapter_matches_nifti_geometry(tmp_path: Path) -> None:
    array = np.random.default_rng(0).integers(0, 1000, size=(6, 20, 24)).astype(np.int16)
    series_uid = "1.2.826.0.1.3680043.2.1125.1"
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()  # otherwise GDCM invents a new series UID per file
    for index, slice_array in enumerate(array):
        writer_image = sitk.GetImageFromArray(slice_array[np.newaxis])
        writer_image.SetSpacing((0.8, 0.8, 2.0))
        writer_image.SetOrigin((0.0, 0.0, 2.0 * index))
        for tag, value in {
            "0020|000e": series_uid,
            "0008|0018": f"{series_uid}.{index + 1}",
            "0020|0013": str(index + 1),
            "0020|0032": f"0\\0\\{2.0 * index}",
            "0020|0037": "1\\0\\0\\0\\1\\0",
            "0008|0060": "CT",
        }.items():
            writer_image.SetMetaData(tag, value)
        writer.SetFileName(str(tmp_path / f"slice_{index:03d}.dcm"))
        writer.Execute(writer_image)

    volume = load_dicom_series(tmp_path)
    assert volume.size_xyz == (24, 20, 6)
    assert np.allclose(volume.spacing_xyz_mm, (0.8, 0.8, 2.0))
    np.testing.assert_array_equal(volume.to_numpy(), array)


# --- validation ------------------------------------------------------------------


def test_valid_volume_has_no_problems(synthetic_root: Path, data_config: DataConfig) -> None:
    volume = load_nifti(synthetic_root / "imagesTr" / "synthetic_001.nii.gz")
    assert image_problems(volume, data_config.limits) == []


def test_malformed_volumes_are_rejected(tmp_path: Path, data_config: DataConfig) -> None:
    limits = data_config.limits
    rng = np.random.default_rng(0)

    tiny = _volume(tmp_path, rng.random((4, 4, 4), dtype=np.float32), name="tiny.nii.gz")
    assert any("below the minimum" in p for p in image_problems(tiny, limits))

    constant = _volume(tmp_path, np.zeros((20, 20, 20), np.float32), name="const.nii.gz")
    assert any("constant intensity" in p for p in image_problems(constant, limits))

    # Built in memory: ITK's NIfTI reader already replaces NaN with 0 on load, so the
    # NaN check protects the other entry points (DICOM adapter, arrays from upstream code).
    with_nan = rng.random((20, 20, 20), dtype=np.float32)
    with_nan[5, 5, 5] = np.nan
    nan_volume = Volume(sitk.GetImageFromArray(with_nan), tmp_path / "nan", "0" * 64)
    assert any("NaN" in p for p in image_problems(nan_volume, limits))

    coarse = _volume(
        tmp_path, rng.random((20, 20, 20), dtype=np.float32), (1, 1, 9), "coarse.nii.gz"
    )
    assert any("spacing" in p for p in image_problems(coarse, limits))

    flat = sitk.GetImageFromArray(rng.random((20, 20), dtype=np.float32))
    flat_volume = load_nifti(save_nifti(flat, tmp_path / "flat.nii.gz"))
    assert image_problems(flat_volume, limits) == ["expected a 3D volume, got 2D"]

    with pytest.raises(InvalidVolumeError, match="constant intensity"):
        validate_image(constant, limits)


def test_label_must_match_image(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    image = _volume(tmp_path, rng.random((20, 20, 20), dtype=np.float32), name="img.nii.gz")
    shifted = _volume(tmp_path, np.zeros((20, 20, 20), np.uint8), (2, 2, 2), "lbl1.nii.gz")
    assert any("spacing" in p for p in label_problems(shifted, image, num_classes=3))

    unknown = np.zeros((20, 20, 20), np.uint8)
    unknown[0, 0, 0] = 7
    bad_class = _volume(tmp_path, unknown, name="lbl2.nii.gz")
    assert label_problems(bad_class, image, num_classes=3) == ["label contains unknown classes [7]"]


# --- manifest --------------------------------------------------------------------


def test_manifest_records_and_version(data_config: DataConfig) -> None:
    manifest = build_manifest(data_config)
    assert manifest["summary"]["accepted"] == 10
    assert manifest["dataset_version"].startswith("ds-synthetic-")
    record = manifest["records"][0]
    assert set(record) == {
        "sample_id",
        "source_path",
        "source_hash",
        "label_path",
        "label_hash",
        "size_xyz",
        "spacing_xyz_mm",
        "modality",
        "dataset_version",
        "split",
        "timestamp",
    }
    assert record["sample_id"] == f"s-{record['source_hash'][:12]}"
    assert not Path(record["source_path"]).is_absolute()  # no machine paths in lineage
    assert {r["split"] for r in manifest["records"]} <= {"train", "val", "test"}

    # Same content -> same version, even though timestamps differ.
    assert build_manifest(data_config)["dataset_version"] == manifest["dataset_version"]


def test_manifest_rejects_bad_files_and_ignores_os_metadata(data_config: DataConfig) -> None:
    images = data_config.root / "imagesTr"
    (images / "._synthetic_001.nii.gz").write_bytes(b"macOS resource fork")
    (images / "broken.nii.gz").write_bytes(b"garbage")
    (data_config.root / "labelsTr" / "broken.nii.gz").write_bytes(b"garbage")

    manifest = build_manifest(data_config)
    assert manifest["summary"]["ignored_os_files"] == 1
    assert manifest["summary"]["accepted"] == 10
    assert [r["source_path"] for r in manifest["rejected"]] == ["imagesTr/broken.nii.gz"]


def test_changed_scan_changes_dataset_version(data_config: DataConfig) -> None:
    before = build_manifest(data_config)["dataset_version"]
    write_dataset(data_config.root, cases=1, seed=99)  # overwrites synthetic_001 with new content
    assert build_manifest(data_config)["dataset_version"] != before


def test_split_is_stable_when_dataset_grows() -> None:
    ids = [f"s-{i:012d}" for i in range(500)]
    splits = {i: assign_split(i, 0.15, 0.15, seed=42) for i in ids}
    # Assigning more samples never changes existing assignments (no test -> train leakage).
    assert all(assign_split(i, 0.15, 0.15, seed=42) == s for i, s in splits.items())
    test_share = sum(s == "test" for s in splits.values()) / len(ids)
    assert 0.10 < test_share < 0.20
