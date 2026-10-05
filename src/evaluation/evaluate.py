"""Evaluate a registered model on a split of its own dataset version.

    python -m src.evaluation.evaluate --model-version v1.0 --split test

Runs the production inference path (``src.inference.predict.segment``) on every
scan of the split and compares the result with the reference label *in the
original scan geometry*. Writes ``artifacts/evaluations/EVAL-*.json`` and
attaches the summary to the registry entry, where the release gate reads it.

Metrics are deterministic for a given checkpoint and dataset; only the
latency figures vary between runs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from src.data.loaders import load_nifti
from src.data.manifest import load_manifest
from src.evaluation.metrics import mean_ignoring_none, per_class_metrics
from src.inference.predict import load_registered_model, segment
from src.lineage.records import ArtifactPaths, new_run_id, read_json, write_json
from src.registry.model_registry import ModelRegistry
from src.utils.logging import get_logger
from src.utils.reproducibility import code_version, resolve_device, utc_now

log = get_logger(__name__)


def evaluate(
    model_version: str, split: str, paths: ArtifactPaths, data_root: Path | None, device_name: str
) -> tuple[dict[str, Any], Path]:
    registry = ModelRegistry(paths)
    model = load_registered_model(registry, model_version, resolve_device(device_name))
    entry = model.entry
    manifest = load_manifest(paths, entry["dataset_version"])
    if data_root is None:
        experiment = read_json(paths.experiment(entry["experiment_id"]) / "experiment.json")
        data_root = Path(experiment["data_config"]["root"])

    records = [r for r in manifest["records"] if r["split"] == split]
    if not records:
        raise ValueError(f"split {split!r} is empty in {entry['dataset_version']}")
    log.info(
        "evaluating %s on %d %s cases of %s",
        entry["model_version"],
        len(records),
        split,
        entry["dataset_version"],
    )

    # One untimed warm-up pass so the first case does not carry one-off setup cost.
    segment(load_nifti(data_root / records[0]["source_path"]).image, model)

    cases, forward_ms, total_ms = [], [], []
    for record in records:
        image = load_nifti(data_root / record["source_path"])
        label = load_nifti(data_root / record["label_path"])
        if image.source_hash != record["source_hash"]:
            raise ValueError(f"{record['source_path']} differs from the manifest")

        started = time.perf_counter()
        mask, forward = segment(image.image, model)
        total_ms.append((time.perf_counter() - started) * 1000)
        forward_ms.append(forward)

        metrics = per_class_metrics(
            sitk.GetArrayFromImage(mask), label.to_numpy().astype(np.uint8), model.class_names
        )
        cases.append(
            {
                "sample_id": record["sample_id"],
                "source_hash": record["source_hash"],
                "per_class": metrics,
            }
        )

    class_names = [name for label, name in sorted(model.class_names.items()) if label]
    per_class = {
        name: {
            metric: mean_ignoring_none([c["per_class"][name][metric] for c in cases])
            for metric in ("dice", "iou", "precision", "recall")
        }
        for name in class_names
    }
    evaluation_id = new_run_id("EVAL")
    result = {
        "evaluation_id": evaluation_id,
        "model_version": entry["model_version"],
        "model_id": entry["model_id"],
        "checkpoint_hash": entry["checkpoint_hash"],
        "experiment_id": entry["experiment_id"],
        "dataset_version": entry["dataset_version"],
        "preprocessing_version": entry["preprocessing_version"],
        "split": split,
        "n_cases": len(cases),
        "dice": float(np.mean([per_class[n]["dice"] for n in class_names])),
        "iou": float(np.mean([per_class[n]["iou"] for n in class_names])),
        "precision": mean_ignoring_none([per_class[n]["precision"] for n in class_names]),
        "recall": mean_ignoring_none([per_class[n]["recall"] for n in class_names]),
        "per_class": per_class,
        "inference_ms": {
            "forward_mean": round(float(np.mean(forward_ms)), 2),
            "forward_p95": round(float(np.percentile(forward_ms, 95)), 2),
            "end_to_end_mean": round(float(np.mean(total_ms)), 2),
        },
        "device": str(model.device),
        "code_version": code_version(),
        "timestamp": utc_now(),
        "cases": cases,
    }
    path = write_json(paths.evaluations / f"{evaluation_id}.json", result)
    registry.attach_evaluation(entry["model_version"], result, path)
    return result, path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a registered model.")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="dataset root on this machine (default: from the experiment record)",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--artifacts-dir", default=Path("artifacts"), type=Path)
    args = parser.parse_args(argv)

    result, path = evaluate(
        args.model_version,
        args.split,
        ArtifactPaths(args.artifacts_dir),
        args.data_root,
        args.device,
    )
    print(
        f"evaluation_id: {result['evaluation_id']}  ({result['n_cases']} {result['split']} cases)"
    )
    print(
        f"mean dice {result['dice']:.4f} | mean IoU {result['iou']:.4f} | "
        f"forward {result['inference_ms']['forward_mean']:.1f} ms/case"
    )
    for name, metrics in result["per_class"].items():
        print(
            f"  {name:<24} dice {metrics['dice']:.4f}  iou {metrics['iou']:.4f}  "
            f"precision {metrics['precision']:.4f}  recall {metrics['recall']:.4f}"
        )
    print(f"written: {path}")


if __name__ == "__main__":
    main()
