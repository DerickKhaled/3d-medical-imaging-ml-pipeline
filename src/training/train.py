"""Train a model from one experiment config.

    python -m src.training.train --config configs/train.yaml

Writes artifacts/experiments/EXP-NNN/ with:
    experiment.json   what was trained, on which data, with which code and settings
    history.json      loss and validation Dice per epoch
    best.pt, last.pt  checkpoints

The best epoch is picked on the validation scans. The final test evaluation is a
separate step (src.evaluation.evaluate).
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import DataConfig, ExperimentConfig, PreprocessingConfig, load_experiment
from src.data.manifest import build_and_save
from src.evaluation.metrics import overlap_counts
from src.lineage.records import ArtifactPaths, next_experiment_id, write_json
from src.models.checkpoint import save_checkpoint
from src.models.unet3d import build_model, count_parameters
from src.preprocessing.cache import ensure_processed, sample_file
from src.training.dataset import ProcessedVolumes
from src.training.losses import DiceCELoss
from src.utils.logging import get_logger
from src.utils.reproducibility import code_version, resolve_device, set_seed, utc_now

log = get_logger(__name__)


def train(config_path: Path, paths: ArtifactPaths) -> dict[str, Any]:
    experiment, data_config, preprocessing = load_experiment(config_path)
    settings = experiment.training
    set_seed(settings.seed)
    # Captured before training starts: the commit that ran, even if the tree changes meanwhile.
    git_state = code_version()
    device = resolve_device(settings.device)
    use_amp = settings.amp and device.type == "cuda"

    manifest = build_and_save(data_config, paths)
    index = ensure_processed(manifest, data_config, preprocessing, paths)
    files = {
        split: [
            sample_file(paths, index, s["sample_id"])
            for s in index["samples"]
            if s["split"] == split
        ]
        for split in ("train", "val")
    }
    if settings.max_train_samples is not None:
        files["train"] = files["train"][: settings.max_train_samples]
    if not files["train"] or not files["val"]:
        raise ValueError("training needs at least one train and one validation sample")

    train_set = ProcessedVolumes(files["train"], settings.augmentation, settings.seed)
    val_set = ProcessedVolumes(files["val"], augmentation=None, seed=settings.seed)
    generator = torch.Generator().manual_seed(settings.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=settings.num_workers)

    model = build_model(experiment.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)
    loss_fn = DiceCELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    experiment_id = next_experiment_id(paths)
    run_dir = paths.experiment(experiment_id)
    run_dir.mkdir(parents=True)
    log.info(
        "%s | %d train / %d val samples | %s parameters | device=%s amp=%s",
        experiment_id,
        len(train_set),
        len(val_set),
        f"{count_parameters(model):,}",
        device,
        use_amp,
    )

    checkpoint_metadata = _checkpoint_metadata(
        experiment_id, experiment, data_config, preprocessing, manifest["dataset_version"]
    )
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"val_mean_dice": -1.0}
    started = time.perf_counter()

    for epoch in range(1, settings.epochs + 1):
        train_set.set_epoch(epoch)
        model.train()
        losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
            with autocast:
                loss = loss_fn(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()

        val_dice = _validate(model, val_loader, device, data_config.class_names)
        val_mean = float(np.mean(list(val_dice.values())))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_mean_dice": val_mean,
                "val_dice": val_dice,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        log.info(
            "epoch %3d | loss %.4f | val dice %.4f %s",
            epoch,
            np.mean(losses),
            val_mean,
            {k: round(v, 3) for k, v in val_dice.items()},
        )

        metadata = checkpoint_metadata | {"epoch": epoch, "val_mean_dice": val_mean}
        last_hash = save_checkpoint(run_dir / "last.pt", model, metadata)
        if val_mean > best["val_mean_dice"]:
            best = {
                "epoch": epoch,
                "val_mean_dice": val_mean,
                "val_dice": val_dice,
                "checkpoint": "best.pt",
                "checkpoint_hash": save_checkpoint(run_dir / "best.pt", model, metadata),
            }

    write_json(run_dir / "history.json", {"experiment_id": experiment_id, "epochs": history})
    record = {
        "experiment_id": experiment_id,
        "status": "completed",
        "dataset_version": manifest["dataset_version"],
        "preprocessing_version": preprocessing.version,
        "processed_index": str(
            paths.processed(manifest["dataset_version"], preprocessing.version) / "index.json"
        ),
        "model_name": experiment.model.name,
        "model_config": experiment.model.model_dump(mode="json"),
        "training_config": settings.model_dump(mode="json"),
        "preprocessing_config": preprocessing.model_dump(mode="json"),
        "postprocessing_config": experiment.postprocessing.model_dump(mode="json"),
        "data_config": data_config.model_dump(mode="json"),
        "random_seed": settings.seed,
        "code_version": git_state,
        "environment": {"torch": torch.__version__, "device": str(device), "amp": use_amp},
        "samples": {"train": len(train_set), "val": len(val_set)},
        "best_checkpoint": best,
        "last_checkpoint": {"checkpoint": "last.pt", "checkpoint_hash": last_hash},
        "metrics": {"best_val_mean_dice": best["val_mean_dice"], "best_val_dice": best["val_dice"]},
        "duration_s": round(time.perf_counter() - started, 1),
        "config_file": str(config_path),
        "timestamp": utc_now(),
    }
    write_json(run_dir / "experiment.json", record)
    log.info(
        "%s done | best epoch %d | val dice %.4f | %s",
        experiment_id,
        best["epoch"],
        best["val_mean_dice"],
        run_dir,
    )
    return record


@torch.no_grad()
def _validate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, class_names: dict[int, str]
) -> dict[str, float]:
    model.eval()
    per_class: dict[str, list[float]] = {name: [] for label, name in class_names.items() if label}
    for images, labels in loader:
        prediction = model(images.to(device)).argmax(dim=1).cpu().numpy()
        reference = labels.numpy()
        for label, name in class_names.items():
            if label:
                per_class[name].append(overlap_counts(prediction, reference, label).dice)
    return {name: float(np.mean(scores)) for name, scores in per_class.items()}


def _checkpoint_metadata(
    experiment_id: str,
    experiment: ExperimentConfig,
    data_config: DataConfig,
    preprocessing: PreprocessingConfig,
    dataset_version: str,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "model_config": experiment.model.model_dump(mode="json"),
        "class_names": {str(k): v for k, v in data_config.class_names.items()},
        "preprocessing_config": preprocessing.model_dump(mode="json"),
        "preprocessing_version": preprocessing.version,
        "postprocessing_config": experiment.postprocessing.model_dump(mode="json"),
        "dataset_version": dataset_version,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a 3D segmentation model.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifacts-dir", default=Path("artifacts"), type=Path)
    args = parser.parse_args(argv)
    record = train(args.config, ArtifactPaths(args.artifacts_dir))
    print(f"experiment_id: {record['experiment_id']}")


if __name__ == "__main__":
    main()
