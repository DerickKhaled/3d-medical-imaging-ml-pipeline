import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from src.data.loaders import save_nifti
from src.data.validation import InvalidVolumeError
from src.inference.predict import keep_largest_component, run_inference
from src.lineage.records import ArtifactPaths
from src.lineage.trace import render, trace_inference
from src.utils.hashing import sha256_array


def test_inference_record_and_output(pipeline_run) -> None:  # type: ignore[no-untyped-def]
    record = pipeline_run.inference
    for key in ("inference_id", "input_hash", "model_version", "dataset_version",
                "preprocessing_version", "prediction_hash", "runtime_ms", "device", "timestamp"):
        assert record[key] is not None, key

    run_dir = pipeline_run.paths.inference_run(record["inference_id"])
    mask = sitk.ReadImage(str(run_dir / record["prediction_file"]))
    scan = sitk.ReadImage(str(pipeline_run.test_scan))
    assert mask.GetSize() == scan.GetSize()  # delivered in the input's geometry
    assert np.allclose(mask.GetSpacing(), scan.GetSpacing())
    assert np.allclose(mask.GetOrigin(), scan.GetOrigin())
    assert sha256_array(sitk.GetArrayFromImage(mask)) == record["prediction_hash"]
    assert record["model_status_at_inference"] == "production"


def test_evaluation_output(pipeline_run) -> None:  # type: ignore[no-untyped-def]
    evaluation = pipeline_run.evaluation
    assert evaluation["split"] == "test" and evaluation["n_cases"] >= 1
    assert 0.0 <= evaluation["dice"] <= 1.0 and 0.0 <= evaluation["iou"] <= 1.0
    assert evaluation["inference_ms"]["forward_mean"] > 0
    assert set(evaluation["per_class"]) == {"anterior_hippocampus", "posterior_hippocampus"}


def test_malformed_input_is_rejected_without_a_record(pipeline_run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    constant = save_nifti(sitk.GetImageFromArray(np.zeros((30, 30, 30), np.float32)),
                          tmp_path / "blank.nii.gz")
    before = set(pipeline_run.paths.inference.iterdir())
    with pytest.raises(InvalidVolumeError, match="constant intensity"):
        run_inference(constant, "production", pipeline_run.paths, "cpu")
    assert set(pipeline_run.paths.inference.iterdir()) == before


def test_keep_largest_component_removes_islands() -> None:
    mask = np.zeros((10, 10, 10), np.uint8)
    mask[1:5, 1:5, 1:5] = 1
    mask[8, 8, 8] = 1  # isolated false positive
    cleaned = keep_largest_component(mask, num_classes=2)
    assert cleaned[8, 8, 8] == 0 and cleaned[2, 2, 2] == 1


def test_trace_links_prediction_to_source(pipeline_run) -> None:  # type: ignore[no-untyped-def]
    trace = trace_inference(pipeline_run.inference["inference_id"], pipeline_run.paths)
    assert trace.ok, render(trace)
    steps = [link["step"] for link in trace.chain]
    assert steps == ["Inference", "Model", "Evaluation", "Experiment", "Dataset",
                     "Preprocessing", "Source scan"]
    source = trace.chain[-1]["details"]["provenance"]
    assert "split: test" in source  # the demo scan was never seen in training
    assert not any("TRAINING split" in w for w in trace.warnings)


@pytest.fixture
def copied_paths(pipeline_run, tmp_path: Path) -> ArtifactPaths:  # type: ignore[no-untyped-def]
    shutil.copytree(pipeline_run.paths.root, tmp_path / "artifacts")
    return ArtifactPaths(tmp_path / "artifacts")


def test_trace_detects_modified_prediction(pipeline_run, copied_paths: ArtifactPaths) -> None:  # type: ignore[no-untyped-def]
    inference_id = pipeline_run.inference["inference_id"]
    path = copied_paths.inference_run(inference_id) / "segmentation.nii.gz"
    mask = sitk.ReadImage(str(path))
    edited = sitk.GetArrayFromImage(mask)
    edited[0, 0, 0] = 1
    tampered = sitk.GetImageFromArray(edited)
    tampered.CopyInformation(mask)
    sitk.WriteImage(tampered, str(path))

    trace = trace_inference(inference_id, copied_paths)
    assert not trace.ok
    assert ("prediction file matches prediction_hash", False) in trace.checks


def test_trace_detects_edited_manifest(pipeline_run, copied_paths: ArtifactPaths) -> None:  # type: ignore[no-untyped-def]
    manifest_path = copied_paths.manifest(pipeline_run.experiment["dataset_version"])
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][0]["split"] = "train" if manifest["records"][0]["split"] != "train" else "test"
    manifest_path.write_text(json.dumps(manifest))

    trace = trace_inference(pipeline_run.inference["inference_id"], copied_paths)
    assert ("dataset manifest content matches its version", False) in trace.checks
