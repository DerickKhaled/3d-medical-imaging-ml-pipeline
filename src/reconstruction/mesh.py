"""Segmentation mask -> one surface mesh per anatomical structure.

    python -m src.reconstruction.mesh --inference-id INF-...

Deliberately independent of the ML model: it consumes a saved mask (and checks
the mask's hash against the inference record), so meshing can be re-run,
changed or validated without touching the model.

Method: marching cubes (Lorensen & Cline, 1987) on each binary structure,
vertices mapped from voxel indices to scanner (physical, mm) coordinates,
optional Taubin smoothing (volume-preserving), export as STL / PLY / OBJ.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import trimesh
from skimage.measure import marching_cubes

from src.config import MeshConfig, load_config
from src.lineage.records import ArtifactPaths, read_json, write_json
from src.utils.hashing import sha256_array, sha256_file
from src.utils.logging import get_logger
from src.utils.reproducibility import utc_now

log = get_logger(__name__)


def mask_to_mesh(
    binary_zyx: np.ndarray, reference: sitk.Image, step: int, smoothing_iterations: int
) -> trimesh.Trimesh | None:
    """Surface of a binary mask in physical (mm) coordinates; None if the mask is empty."""
    if not binary_zyx.any():
        return None
    padded = np.pad(binary_zyx.astype(np.float32), 1)  # closes surfaces touching the border
    vertices_zyx, faces, _, _ = marching_cubes(padded, level=0.5, step_size=step)
    index_xyz = vertices_zyx[:, ::-1] - 1.0  # undo padding; (z,y,x) -> (x,y,z)

    spacing = np.asarray(reference.GetSpacing())
    direction = np.asarray(reference.GetDirection()).reshape(3, 3)
    origin = np.asarray(reference.GetOrigin())
    physical = origin + (index_xyz * spacing) @ direction.T

    # A scan direction matrix with a reflection (negative determinant) turns triangles
    # inside out; restore outward-facing winding (verified by the sphere-volume test).
    if np.linalg.det(direction) < 0:
        faces = faces[:, ::-1]

    mesh = trimesh.Trimesh(vertices=physical, faces=faces, process=True)
    if smoothing_iterations > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=smoothing_iterations)
    return mesh


def build_meshes(inference_id: str, paths: ArtifactPaths, config: MeshConfig) -> dict[str, Any]:
    run_dir = paths.inference_run(inference_id)
    record = read_json(run_dir / "record.json")
    mask = sitk.ReadImage(str(run_dir / record["prediction_file"]))
    voxels = sitk.GetArrayFromImage(mask)
    if sha256_array(voxels) != record["prediction_hash"]:
        raise ValueError(f"{record['prediction_file']} does not match its inference record")

    out_dir = run_dir / "meshes"
    out_dir.mkdir(exist_ok=True)
    voxel_ml = float(np.prod(mask.GetSpacing())) / 1000.0
    structures = []
    for name, info in record["structures"].items():
        mesh = mask_to_mesh(
            voxels == info["label"], mask, config.marching_cubes_step, config.smoothing_iterations
        )
        if mesh is None:
            log.info("%s: not present in the prediction, no mesh", name)
            continue
        files = {}
        for fmt in config.formats:
            path = out_dir / f"{name}.{fmt}"
            mesh.export(path)
            files[fmt] = {"file": path.name, "sha256": sha256_file(path)}
        structures.append(
            {
                "name": name,
                "label": info["label"],
                "files": files,
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "watertight": bool(mesh.is_watertight),
                "surface_area_mm2": round(float(mesh.area), 2),
                "mesh_volume_ml": round(float(mesh.volume) / 1000.0, 4),
                "voxel_volume_ml": round(
                    int(np.count_nonzero(voxels == info["label"])) * voxel_ml, 4
                ),
            }
        )

    mesh_record = {
        "inference_id": inference_id,
        "prediction_hash": record["prediction_hash"],
        "mesh_config": config.model_dump(mode="json"),
        "coordinate_system": "scanner physical coordinates (ITK/LPS convention), millimetres",
        "structures": structures,
        "timestamp": utc_now(),
    }
    write_json(out_dir / "meshes.json", mesh_record)
    return mesh_record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build 3D meshes from a saved prediction.")
    parser.add_argument("--inference-id", required=True)
    parser.add_argument("--config", default=Path("configs/mesh.yaml"), type=Path)
    parser.add_argument("--artifacts-dir", default=Path("artifacts"), type=Path)
    args = parser.parse_args(argv)

    result = build_meshes(
        args.inference_id, ArtifactPaths(args.artifacts_dir), load_config(args.config, MeshConfig)
    )
    for s in result["structures"]:
        print(
            f"{s['name']:<24} {s['vertices']:>6} vertices  {s['mesh_volume_ml']:.3f} mL "
            f"(voxels {s['voxel_volume_ml']:.3f} mL)  watertight={s['watertight']}  "
            f"-> {', '.join(f['file'] for f in s['files'].values())}"
        )


if __name__ == "__main__":
    main()
