import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from src.reconstruction.mesh import mask_to_mesh
from src.visualization.scene import load_case


def _sphere(radius_vox: float, size: int = 40) -> np.ndarray:
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    c = (size - 1) / 2
    return ((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2) <= radius_vox**2


def _reference(spacing=(0.5, 0.5, 0.5), origin=(10.0, -20.0, 5.0), direction=None) -> sitk.Image:
    image = sitk.Image(40, 40, 40, sitk.sitkUInt8)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(direction)
    return image


@pytest.mark.parametrize("direction", [None, (-1, 0, 0, 0, -1, 0, 0, 0, 1)])
def test_sphere_mesh_is_closed_outward_and_in_millimetres(direction) -> None:  # type: ignore[no-untyped-def]
    mesh = mask_to_mesh(_sphere(12), _reference(direction=direction), step=1, smoothing_iterations=10)
    assert mesh is not None
    assert mesh.is_watertight
    expected_mm3 = 4 / 3 * np.pi * (12 * 0.5) ** 3  # radius 12 voxels x 0.5 mm
    assert mesh.volume > 0  # positive volume == outward-facing triangles
    assert mesh.volume == pytest.approx(expected_mm3, rel=0.05)

    center = mesh.vertices.mean(axis=0)
    ref = _reference(direction=direction)
    expected_center = ref.TransformContinuousIndexToPhysicalPoint((19.5, 19.5, 19.5))
    np.testing.assert_allclose(center, expected_center, atol=0.3)


def test_empty_mask_gives_no_mesh() -> None:
    assert mask_to_mesh(np.zeros((10, 10, 10), bool), _reference(), 1, 0) is None


def test_pipeline_meshes_exist_in_all_formats(pipeline_run) -> None:  # type: ignore[no-untyped-def]
    mesh_dir = pipeline_run.paths.inference_run(pipeline_run.inference["inference_id"]) / "meshes"
    assert pipeline_run.meshes["prediction_hash"] == pipeline_run.inference["prediction_hash"]
    assert pipeline_run.meshes["structures"], "smoke model should detect at least one structure"
    for structure in pipeline_run.meshes["structures"]:
        for fmt in ("stl", "ply", "obj"):
            assert (mesh_dir / structure["files"][fmt]["file"]).stat().st_size > 0


def test_viewer_scene_loads_with_provenance(pipeline_run) -> None:  # type: ignore[no-untyped-def]
    scene = load_case(pipeline_run.inference["inference_id"], pipeline_run.paths,
                      image_path=pipeline_run.test_scan)
    assert set(scene.meshes) == {s["name"] for s in pipeline_run.meshes["structures"]}
    assert scene.image.dimensions == sitk.ReadImage(str(pipeline_run.test_scan)).GetSize()
    summary = "\n".join(scene.summary_lines())
    assert pipeline_run.inference["model_version"] in summary
    assert pipeline_run.inference["dataset_version"] in summary


def test_viewer_refuses_a_different_scan(pipeline_run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    other = pipeline_run.data_root / "imagesTr" / "synthetic_012.nii.gz"
    if other == pipeline_run.test_scan:
        other = pipeline_run.data_root / "imagesTr" / "synthetic_011.nii.gz"
    with pytest.raises(ValueError, match="hash mismatch"):
        load_case(pipeline_run.inference["inference_id"], pipeline_run.paths, image_path=other)


@pytest.mark.gui
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="no OpenGL context in CI")
def test_viewer_renders_offscreen(pipeline_run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from src.visualization.viewer import build_plotter

    scene = load_case(pipeline_run.inference["inference_id"], pipeline_run.paths,
                      image_path=pipeline_run.test_scan)
    plotter = build_plotter(scene, off_screen=True)
    out = tmp_path / "viewer.png"
    plotter.screenshot(str(out))
    plotter.close()
    assert out.stat().st_size > 10_000
