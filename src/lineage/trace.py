"""Trace a prediction back to the scan, model and data that produced it.

    python -m src.lineage.trace --inference-id INF-20261005-abc123

It shows the chain prediction -> model -> evaluation -> experiment -> dataset ->
preprocessing -> source scan. Along the way it re-computes the hashes of the
files, so it notices if anything was edited afterwards. It exits with code 1
if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import SimpleITK as sitk

from src.config import PreprocessingConfig
from src.data.manifest import compute_dataset_version, load_manifest
from src.lineage.records import ArtifactPaths, read_json
from src.registry.model_registry import ModelRegistry
from src.utils.hashing import sha256_array, sha256_file


@dataclass
class Trace:
    chain: list[dict[str, Any]] = field(default_factory=list)
    checks: list[tuple[str, bool]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def check(self, description: str, passed: bool) -> bool:
        self.checks.append((description, passed))
        return passed

    @property
    def ok(self) -> bool:
        return all(passed for _, passed in self.checks)


def trace_inference(inference_id: str, paths: ArtifactPaths) -> Trace:
    trace = Trace()

    # 1. Prediction -------------------------------------------------------------------
    run_dir = paths.inference_run(inference_id)
    inference = read_json(run_dir / "record.json")
    mask = sitk.ReadImage(str(run_dir / inference["prediction_file"]))
    trace.check(
        "prediction file matches prediction_hash",
        sha256_array(sitk.GetArrayFromImage(mask)) == inference["prediction_hash"],
    )
    trace.chain.append(
        {
            "step": "Inference",
            "id": inference_id,
            "details": {
                "timestamp": inference["timestamp"],
                "device": inference["device"],
                "runtime_ms": inference["runtime_ms"],
                "prediction_hash": inference["prediction_hash"][:16],
                "model_status_at_inference": inference["model_status_at_inference"],
            },
        }
    )
    meshes_file = run_dir / "meshes" / "meshes.json"
    if meshes_file.exists():
        meshes = read_json(meshes_file)
        trace.check(
            "3D meshes were built from this exact prediction",
            meshes["prediction_hash"] == inference["prediction_hash"],
        )

    # 2. Model ------------------------------------------------------------------------
    registry = ModelRegistry(paths)
    entry = registry.get(inference["model_version"])
    trace.check(
        "registry checkpoint hash matches the inference record",
        entry["checkpoint_hash"] == inference["checkpoint_hash"],
    )
    trace.check(
        "checkpoint file on disk matches its registered hash",
        sha256_file(registry.checkpoint_path(entry)) == entry["checkpoint_hash"],
    )
    trace.chain.append(
        {
            "step": "Model",
            "id": f"{entry['model_version']} ({entry['model_id']})",
            "details": {
                "current_status": entry["status"],
                "checkpoint_hash": entry["checkpoint_hash"][:16],
                "promotions": [
                    f"{h['from'] or 'new'} -> {h['to']} at {h['at']} by {h['by']}: {h['reason']}"
                    for h in entry["history"]
                ],
            },
        }
    )

    # 3. Evaluation evidence ------------------------------------------------------------
    evaluation = entry["evaluation_metrics"]
    if evaluation is None:
        trace.warnings.append("model has no evaluation evidence")
    else:
        evaluation_path = Path(evaluation["evaluation_file"])
        trace.check(
            "evaluation file is unchanged since it was attached",
            evaluation_path.exists()
            and sha256_file(evaluation_path) == evaluation["evaluation_file_hash"],
        )
        trace.chain.append(
            {
                "step": "Evaluation",
                "id": evaluation["evaluation_id"],
                "details": {
                    "split": evaluation["split"],
                    "n_cases": evaluation["n_cases"],
                    "mean_dice": round(evaluation["dice"], 4),
                    "mean_iou": round(evaluation["iou"], 4),
                    "per_class_dice": {k: round(v, 4) for k, v in evaluation["per_class"].items()},
                },
            }
        )

    # 4. Experiment ---------------------------------------------------------------------
    experiment = read_json(paths.experiment(entry["experiment_id"]) / "experiment.json")
    trace.check(
        "registered checkpoint is the experiment's best checkpoint",
        experiment["best_checkpoint"]["checkpoint_hash"] == entry["checkpoint_hash"],
    )
    git = experiment["code_version"]
    if git.get("git_dirty"):
        trace.warnings.append("experiment was trained from a working tree with uncommitted changes")
    trace.chain.append(
        {
            "step": "Experiment",
            "id": experiment["experiment_id"],
            "details": {
                "git_commit": (git.get("git_commit") or "unknown")[:12],
                "random_seed": experiment["random_seed"],
                "epochs": experiment["training_config"]["epochs"],
                "best_epoch": experiment["best_checkpoint"]["epoch"],
                "best_val_dice": round(experiment["best_checkpoint"]["val_mean_dice"], 4),
                "config_file": experiment["config_file"],
            },
        }
    )

    # 5. Dataset ------------------------------------------------------------------------
    manifest = load_manifest(paths, entry["dataset_version"])
    recomputed = compute_dataset_version(
        manifest["dataset_name"], manifest["modality"], manifest["class_names"], manifest["records"]
    )
    trace.check(
        "dataset manifest content matches its version", recomputed == entry["dataset_version"]
    )
    trace.check(
        "experiment trained on this dataset version",
        experiment["dataset_version"] == entry["dataset_version"],
    )
    summary = manifest["summary"]
    trace.chain.append(
        {
            "step": "Dataset",
            "id": manifest["dataset_version"],
            "details": {
                "name": manifest["dataset_name"],
                "modality": manifest["modality"],
                "scans": summary["accepted"],
                "splits": {s: summary[s] for s in ("train", "val", "test")},
            },
        }
    )

    # 6. Preprocessing ------------------------------------------------------------------
    preprocessing = PreprocessingConfig.model_validate(entry["preprocessing_config"])
    trace.check(
        "preprocessing config re-hashes to the recorded version",
        preprocessing.version == inference["preprocessing_version"],
    )
    trace.chain.append(
        {
            "step": "Preprocessing",
            "id": preprocessing.version,
            "details": {
                "orientation": preprocessing.orientation,
                "target_spacing_mm": list(preprocessing.target_spacing_mm),
                "intensity": preprocessing.intensity.mode,
                "network_grid_xyz": list(preprocessing.crop_or_pad_size_xyz),
            },
        }
    )

    # 7. Source scan --------------------------------------------------------------------
    match = next(
        (r for r in manifest["records"] if r["source_hash"] == inference["input_hash"]), None
    )
    where = (
        f"in dataset as {match['sample_id']} (split: {match['split']})"
        if match
        else "not part of the training dataset (external scan)"
    )
    if match and match["split"] == "train":
        trace.warnings.append(
            "input scan was in the TRAINING split: this result is not a fair test"
        )
    trace.chain.append(
        {
            "step": "Source scan",
            "id": inference["input_hash"][:16],
            "details": {
                "file_name": inference["input_name"],
                "size_xyz": inference["input_size_xyz"],
                "spacing_mm": inference["input_spacing_xyz_mm"],
                "provenance": where,
            },
        }
    )
    return trace


def render(trace: Trace) -> str:
    lines = []
    for index, link in enumerate(trace.chain):
        if index:
            lines.append("   |")
            lines.append("   v")
        lines.append(f"{link['step']}: {link['id']}")
        for key, value in link["details"].items():
            if key == "promotions":  # the release history reads best as one event per line
                lines.append(f"   {key}:")
                lines.extend(f"     - {event}" for event in value)
            else:
                lines.append(f"   {key}: {value}")
    lines.append("")
    lines.append("Integrity checks:")
    lines.extend(
        f"  [{'PASS' if ok else 'FAIL'}] {description}" for description, ok in trace.checks
    )
    lines.extend(f"  [WARN] {warning}" for warning in trace.warnings)
    lines.append("")
    lines.append("LINEAGE VERIFIED" if trace.ok else "LINEAGE BROKEN: see FAIL lines above")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trace a prediction to its data and model.")
    parser.add_argument("--inference-id", required=True)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--artifacts-dir", default=Path("artifacts"), type=Path)
    args = parser.parse_args(argv)

    trace = trace_inference(args.inference_id, ArtifactPaths(args.artifacts_dir))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # box-drawing characters on Windows consoles
    if args.json:
        print(
            json.dumps(
                {
                    "chain": trace.chain,
                    "checks": trace.checks,
                    "warnings": trace.warnings,
                    "ok": trace.ok,
                },
                indent=2,
            )
        )
    else:
        print(render(trace))
    raise SystemExit(0 if trace.ok else 1)


if __name__ == "__main__":
    main()
