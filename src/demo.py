"""Run the whole demo with one command.

    python -m src.demo                                   # default test scan
    python -m src.demo --input path/to/scan.nii.gz       # any other scan or DICOM folder
    python -m src.demo --no-viewer                       # skip opening the window

If there is no production model yet, it first prepares everything: download the
data, build the manifest, train, register, evaluate and promote. This takes
about an hour on a CPU and only happens once.

After that, each run segments the scan, builds the 3D meshes and the slice
picture, prints the lineage trace, and opens the viewer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DataConfig, MeshConfig, ReleaseConfig, load_config
from src.data.download import download
from src.data.manifest import build_and_save
from src.evaluation.evaluate import evaluate
from src.inference.predict import run_inference
from src.lineage.records import ArtifactPaths
from src.lineage.trace import render, trace_inference
from src.reconstruction.mesh import build_meshes
from src.registry.model_registry import ModelRegistry, RegistryError
from src.training.train import train
from src.visualization.scene import load_case
from src.visualization.slices import slice_figure

DEFAULT_SCAN = Path("data/Task04_Hippocampus/imagesTr/hippocampus_017.nii.gz")  # test split
VERSION = "v1.0"


def has_production_model(registry: ModelRegistry) -> bool:
    try:
        registry.get("production")
        return True
    except RegistryError:
        return False


def prepare(paths: ArtifactPaths) -> None:
    """First run only: get the data and train, evaluate and release a model."""
    print("No production model yet. Preparing everything once (about 1 hour on CPU).")
    download(Path("data"))
    build_and_save(load_config("configs/data.yaml", DataConfig), paths)

    experiment = train(Path("configs/train.yaml"), paths)
    registry = ModelRegistry(paths)
    registry.register(experiment["experiment_id"], VERSION)
    result, _ = evaluate(VERSION, "test", paths, data_root=None, device_name="auto")
    print(f"Test Dice: {result['dice']:.4f} on {result['n_cases']} scans")

    release = load_config("configs/release.yaml", ReleaseConfig)
    registry.promote(VERSION, "validated", "test results meet the release rules", release)
    registry.promote(VERSION, "production", "first release", release)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the full demo with one command.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_SCAN,
        help="NIfTI file or DICOM folder (default: a test-split scan)",
    )
    parser.add_argument("--no-viewer", action="store_true", help="don't open the 3D window")
    parser.add_argument("--artifacts-dir", default=Path("artifacts"), type=Path)
    args = parser.parse_args(argv)

    paths = ArtifactPaths(args.artifacts_dir)
    if not has_production_model(ModelRegistry(paths)):
        prepare(paths)

    record = run_inference(args.input, "production", paths)
    case = record["inference_id"]
    print(
        f"\nSegmented {args.input.name} with model {record['model_version']} "
        f"in {record['runtime_ms']:.0f} ms  ->  {case}"
    )
    for name, info in record["structures"].items():
        print(f"  {name:<24} {info['volume_ml']:.2f} mL")

    build_meshes(case, paths, load_config("configs/mesh.yaml", MeshConfig))
    scene = load_case(case, paths)
    picture = slice_figure(scene, paths.inference_run(case) / "slices.png")
    print(f"Meshes and slice picture saved in {picture.parent}\n")

    trace = trace_inference(case, paths)
    print(render(trace))

    if not args.no_viewer:
        from src.visualization.viewer import build_plotter  # needs a screen, so import late

        print("\nOpening the viewer (close the window to finish)...")
        build_plotter(scene).show()


if __name__ == "__main__":
    main()
