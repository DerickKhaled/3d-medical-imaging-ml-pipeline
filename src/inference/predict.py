"""Production-style inference: one scan in, one traceable segmentation out.

    python -m src.inference.predict --input scan.nii.gz --model-version v1.0
    python -m src.inference.predict --input scan.nii.gz --model-version production

Steps (each is a function below, in this order):

    1. load the input                      6. run the network
    2. validate it (model's input limits)  7. post-process the mask
    3. hash it                             8. map it back to the input geometry and save
    4. load the model's *registered*       9. measure runtime
       preprocessing (never the caller's)  10. write the inference record
    5. load the checkpoint (hash-verified)

The same ``segment()`` function is used by ``src.evaluation.evaluate``, so the
reported metrics describe exactly the code path that produces predictions.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
from skimage.measure import label as connected_components

from src.config import PostprocessingConfig, PreprocessingConfig, VolumeLimits
from src.data.loaders import load_volume, save_nifti
from src.data.validation import validate_image
from src.lineage.records import ArtifactPaths, new_run_id, write_json
from src.models.checkpoint import load_checkpoint
from src.models.unet3d import UNet3D
from src.preprocessing.transforms import preprocess_image, restore_to_original
from src.registry.model_registry import ModelRegistry, RegistryError
from src.utils.hashing import sha256_array
from src.utils.logging import get_logger
from src.utils.reproducibility import code_version, resolve_device, utc_now

log = get_logger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    entry: dict[str, Any]
    network: UNet3D
    preprocessing: PreprocessingConfig
    postprocessing: PostprocessingConfig
    input_limits: VolumeLimits
    class_names: dict[int, str]
    device: torch.device


def load_registered_model(registry: ModelRegistry, version: str, device: torch.device) -> LoadedModel:
    entry = registry.get(version)
    network, metadata = load_checkpoint(
        registry.checkpoint_path(entry), device, expected_hash=entry["checkpoint_hash"]
    )
    preprocessing = PreprocessingConfig.model_validate(entry["preprocessing_config"])
    if preprocessing.version != entry["preprocessing_version"] or (
        metadata["preprocessing_version"] != entry["preprocessing_version"]
    ):
        raise RegistryError(
            f"preprocessing mismatch for {entry['model_version']}: registry, checkpoint and "
            "stored config must agree"
        )
    return LoadedModel(
        entry=entry,
        network=network,
        preprocessing=preprocessing,
        postprocessing=PostprocessingConfig.model_validate(entry["postprocessing_config"]),
        input_limits=VolumeLimits.model_validate(entry["input_limits"]),
        class_names={int(k): v for k, v in entry["class_names"].items()},
        device=device,
    )


def keep_largest_component(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Per class, keep only the largest connected region (removes isolated false positives)."""
    cleaned = np.zeros_like(mask)
    for label in range(1, num_classes):
        components, count = connected_components(mask == label, connectivity=1, return_num=True)
        if count == 0:
            continue
        sizes = np.bincount(components.ravel())[1:]
        cleaned[components == int(np.argmax(sizes)) + 1] = label
    return cleaned


@torch.no_grad()
def segment(image: sitk.Image, model: LoadedModel) -> tuple[sitk.Image, float]:
    """Segment one scan. Returns the mask on the input's voxel grid and the forward time (ms)."""
    array, grid = preprocess_image(image, model.preprocessing)
    tensor = torch.from_numpy(array)[None].to(model.device)

    if model.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    logits = model.network(tensor)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - started) * 1000

    mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    restored = restore_to_original(mask, grid, image)
    if model.postprocessing.keep_largest_component:
        cleaned = keep_largest_component(sitk.GetArrayFromImage(restored), len(model.class_names))
        cleaned_image = sitk.GetImageFromArray(cleaned)
        cleaned_image.CopyInformation(restored)
        restored = cleaned_image
    return restored, forward_ms


def structure_summary(mask: sitk.Image, class_names: dict[int, str]) -> dict[str, dict[str, Any]]:
    voxels = sitk.GetArrayFromImage(mask)
    voxel_ml = float(np.prod(mask.GetSpacing())) / 1000.0
    summary = {}
    for label, name in sorted(class_names.items()):
        if label == 0:
            continue
        count = int(np.count_nonzero(voxels == label))
        summary[name] = {"label": label, "voxels": count, "volume_ml": round(count * voxel_ml, 4),
                         "detected": count > 0}
    return summary


def run_inference(
    input_path: Path, model_version: str, paths: ArtifactPaths, device_name: str = "auto"
) -> dict[str, Any]:
    started = time.perf_counter()
    device = resolve_device(device_name)
    model = load_registered_model(ModelRegistry(paths), model_version, device)

    volume = load_volume(input_path)
    validate_image(volume, model.input_limits)
    mask, forward_ms = segment(volume.image, model)

    inference_id = new_run_id("INF")
    run_dir = paths.inference_run(inference_id)
    prediction_path = save_nifti(mask, run_dir / "segmentation.nii.gz")
    entry = model.entry
    if entry["status"] != "production":
        log.warning("model %s has status %r (not production)", entry["model_version"], entry["status"])

    record = {
        "inference_id": inference_id,
        # Path as given on the command line (relative in this demo). In a hospital setting this
        # would be a PACS/archive reference: paths and file names can carry patient identifiers.
        "input_path": input_path.as_posix(),
        "input_name": input_path.name,
        "input_hash": volume.source_hash,
        "input_size_xyz": list(volume.size_xyz),
        "input_spacing_xyz_mm": list(volume.spacing_xyz_mm),
        "model_id": entry["model_id"],
        "model_version": entry["model_version"],
        "model_status_at_inference": entry["status"],
        "checkpoint_hash": entry["checkpoint_hash"],
        "experiment_id": entry["experiment_id"],
        "dataset_version": entry["dataset_version"],
        "preprocessing_version": entry["preprocessing_version"],
        "postprocessing_config": entry["postprocessing_config"],
        "prediction_file": prediction_path.name,
        "prediction_hash": sha256_array(sitk.GetArrayFromImage(mask)),
        "structures": structure_summary(mask, model.class_names),
        "forward_ms": round(forward_ms, 2),
        "runtime_ms": round((time.perf_counter() - started) * 1000, 2),
        "device": str(device),
        "code_version": code_version(),
        "timestamp": utc_now(),
    }
    write_json(run_dir / "record.json", record)
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Segment one CT/MRI volume with a registered model.")
    parser.add_argument("--input", required=True, type=Path, help="NIfTI file or DICOM directory")
    parser.add_argument("--model-version", required=True, help="e.g. v1.0, or 'production'")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--artifacts-dir", default=Path("artifacts"), type=Path)
    args = parser.parse_args(argv)

    paths = ArtifactPaths(args.artifacts_dir)
    record = run_inference(args.input, args.model_version, paths, args.device)
    print(f"inference_id: {record['inference_id']}")
    print(f"model: {record['model_version']} ({record['model_status_at_inference']}) | "
          f"runtime {record['runtime_ms']:.0f} ms (network {record['forward_ms']:.0f} ms)")
    for name, info in record["structures"].items():
        print(f"  {name:<24} {info['volume_ml']:.3f} mL  detected={info['detected']}")
    print(f"output: {paths.inference_run(record['inference_id'])}")


if __name__ == "__main__":
    main()
