"""Load everything the viewer needs for one case, without opening a window.

Kept apart from the viewer so it can be tested without a screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import SimpleITK as sitk

from src.lineage.records import ArtifactPaths, read_json
from src.utils.hashing import sha256_file

# One fixed colour per structure label; distinguishable under common colour-vision deficiencies.
STRUCTURE_COLORS = ["#e8743b", "#19a979", "#5899da", "#bf399e", "#945ecf", "#13a4b4"]


@dataclass
class CaseScene:
    inference_id: str
    record: dict[str, Any]
    image: pv.ImageData  # the input scan, in physical coordinates
    labels: pv.ImageData  # the predicted mask: one cell per voxel, so colours are never blended
    label_array: np.ndarray  # the predicted mask as (z, y, x) voxels
    meshes: dict[str, pv.PolyData]
    colors: dict[str, str]
    mesh_info: dict[str, dict[str, Any]]

    def summary_lines(self) -> list[str]:
        r = self.record
        lines = [
            f"Result ID        {self.inference_id}",
            f"Scan             {r['input_name']}  ({'x'.join(map(str, r['input_size_xyz']))} voxels)",
            f"Model            {r['model_version']}  ({r['model_status_at_inference']})",
            f"Trained in       {r['experiment_id']}",
            f"Training data    {r['dataset_version']}",
            f"Preprocessing    {r['preprocessing_version']}",
            f"Time             {r['runtime_ms']:.0f} ms on {r['device']}",
            "Volumes found:",
        ]
        for name, info in r["structures"].items():
            lines.append(
                f"  {name}: {info['volume_ml']:.2f} mL"
                if info["detected"]
                else f"  {name}: not detected"
            )
        return lines


def to_pyvista(image: sitk.Image, name: str) -> pv.ImageData:
    """SimpleITK image -> PyVista grid with the same physical geometry."""
    grid = pv.ImageData(
        dimensions=image.GetSize(),
        spacing=image.GetSpacing(),
        origin=image.GetOrigin(),
        direction_matrix=np.asarray(image.GetDirection()).reshape(3, 3),
    )
    # NumPy (z, y, x) in C order == VTK point order (x fastest).
    grid.point_data[name] = sitk.GetArrayFromImage(image).ravel()
    return grid


def to_pyvista_voxels(image: sitk.Image, name: str) -> pv.ImageData:
    """Label image -> grid with one *cell* per voxel.

    Point data would be interpolated across voxel corners when rendered, blending a
    structure's border with the background and drawing labels the model never
    predicted. Cell data is shown exactly as predicted.
    """
    spacing = np.asarray(image.GetSpacing())
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    grid = pv.ImageData(
        dimensions=np.asarray(image.GetSize()) + 1,
        spacing=spacing,
        origin=np.asarray(image.GetOrigin()) - direction @ (spacing / 2),  # voxel corners
        direction_matrix=direction,
    )
    grid.cell_data[name] = sitk.GetArrayFromImage(image).ravel()
    return grid


def load_case(inference_id: str, paths: ArtifactPaths, image_path: Path | None = None) -> CaseScene:
    run_dir = paths.inference_run(inference_id)
    record = read_json(run_dir / "record.json")
    meshes_file = run_dir / "meshes" / "meshes.json"
    if not meshes_file.exists():
        raise FileNotFoundError(
            f"no meshes for {inference_id}; run: python -m src.reconstruction.mesh "
            f"--inference-id {inference_id}"
        )

    source = image_path or Path(record["input_path"])
    if sha256_file(source) != record["input_hash"]:
        raise ValueError(f"{source} is not the scan recorded for {inference_id} (hash mismatch)")
    image = to_pyvista(sitk.ReadImage(str(source)), "intensity")
    mask = sitk.ReadImage(str(run_dir / record["prediction_file"]))
    labels = to_pyvista_voxels(mask, "label")

    meshes, colors, info = {}, {}, {}
    for structure in read_json(meshes_file)["structures"]:
        name = structure["name"]
        # STL, PLY and OBJ hold the same surface; load whichever was written first.
        any_format = next(iter(structure["files"].values()))
        meshes[name] = pv.read(run_dir / "meshes" / any_format["file"])
        colors[name] = STRUCTURE_COLORS[(structure["label"] - 1) % len(STRUCTURE_COLORS)]
        info[name] = structure
    return CaseScene(
        inference_id, record, image, labels, sitk.GetArrayFromImage(mask), meshes, colors, info
    )
