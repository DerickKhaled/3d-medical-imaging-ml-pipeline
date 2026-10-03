"""A lightweight local model registry with explicit, evidence-gated promotion.

    python -m src.registry.model_registry register --experiment EXP-001 --version v1.0
    python -m src.registry.model_registry list
    python -m src.registry.model_registry inspect --version v1.0
    python -m src.registry.model_registry promote --version v1.0 --to validated --reason "..."

Lifecycle:

    candidate ──(evaluation meets release gate)──▶ validated ──(explicit decision)──▶ production
        │                                              │                                  │
        └──────────────────────────────────────────────┴──────────▶ retired ◀─────────────┘

* Registration copies the checkpoint into the registry and records its hash;
  every later load verifies that hash.
* A model is promoted only by an explicit command with a written reason.
  Every transition is appended to the entry's history (who, when, why).
* Only one model is in production at a time; promoting a new one retires the old.

Storage is a single JSON file (``artifacts/registry/registry.json``) designed
for one writer at a time. This is a demonstration of controlled model lifecycle
management, not a claim of regulatory compliance.
"""

from __future__ import annotations

import argparse
import getpass
import json
import shutil
from pathlib import Path
from typing import Any

from src.config import ReleaseConfig, load_config
from src.lineage.records import ArtifactPaths, read_json, write_json
from src.utils.hashing import sha256_file
from src.utils.reproducibility import utc_now

STATUSES = ("candidate", "validated", "production", "retired")
ALLOWED_TRANSITIONS = {
    "candidate": {"validated", "retired"},
    "validated": {"production", "retired"},
    "production": {"retired"},
    "retired": set(),
}


class RegistryError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, paths: ArtifactPaths) -> None:
        self.paths = paths
        self.file = paths.registry / "registry.json"

    # --- reading -------------------------------------------------------------------

    def entries(self) -> list[dict[str, Any]]:
        if not self.file.exists():
            return []
        return list(read_json(self.file)["models"])

    def get(self, version: str) -> dict[str, Any]:
        """Look up a version; the alias 'production' resolves to the production model."""
        for entry in self.entries():
            if entry["model_version"] == version or (
                version == "production" and entry["status"] == "production"
            ):
                return entry
        known = ", ".join(e["model_version"] for e in self.entries()) or "none"
        raise RegistryError(f"model version {version!r} not found (registered: {known})")

    def checkpoint_path(self, entry: dict[str, Any]) -> Path:
        return self.paths.registry / entry["checkpoint_path"]

    # --- writing -------------------------------------------------------------------

    def register(self, experiment_id: str, version: str, note: str = "") -> dict[str, Any]:
        if any(e["model_version"] == version for e in self.entries()):
            raise RegistryError(f"version {version} already exists; versions are immutable")

        experiment = read_json(self.paths.experiment(experiment_id) / "experiment.json")
        best = experiment["best_checkpoint"]
        source = self.paths.experiment(experiment_id) / best["checkpoint"]
        if sha256_file(source) != best["checkpoint_hash"]:
            raise RegistryError(f"{source} does not match the hash in {experiment_id}'s record")

        relative = Path("models") / version / "model.pt"
        target = self.paths.registry / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        checkpoint_hash = sha256_file(target)

        entry = {
            "model_id": f"{experiment['model_name']}-{checkpoint_hash[:8]}",
            "model_name": experiment["model_name"],
            "model_version": version,
            "status": "candidate",
            "experiment_id": experiment_id,
            "checkpoint_path": relative.as_posix(),
            "checkpoint_hash": checkpoint_hash,
            "dataset_version": experiment["dataset_version"],
            "preprocessing_version": experiment["preprocessing_version"],
            "preprocessing_config": experiment["preprocessing_config"],
            "postprocessing_config": experiment["postprocessing_config"],
            "model_config": experiment["model_config"],
            "training_config": experiment["training_config"],
            "class_names": experiment["data_config"]["class_names"],
            "input_limits": experiment["data_config"]["limits"],
            "code_version": experiment["code_version"],
            "training_metrics": experiment["metrics"],
            "evaluation_metrics": None,
            "created_at": utc_now(),
            "history": [self._event(None, "candidate", note or f"registered from {experiment_id}")],
        }
        self._save([*self.entries(), entry])
        return entry

    def attach_evaluation(self, version: str, evaluation: dict[str, Any], path: Path) -> None:
        entries = self.entries()
        entry = self._find(entries, version)
        entry["evaluation_metrics"] = {
            "evaluation_id": evaluation["evaluation_id"],
            "evaluation_file": path.as_posix(),
            "evaluation_file_hash": sha256_file(path),
            "split": evaluation["split"],
            "n_cases": evaluation["n_cases"],
            "dataset_version": evaluation["dataset_version"],
            "checkpoint_hash": evaluation["checkpoint_hash"],
            "dice": evaluation["dice"],
            "iou": evaluation["iou"],
            "per_class": {k: v["dice"] for k, v in evaluation["per_class"].items()},
            "inference_ms": evaluation["inference_ms"],
        }
        self._save(entries)

    def promote(
        self, version: str, to_status: str, reason: str, release: ReleaseConfig
    ) -> dict[str, Any]:
        if not reason.strip():
            raise RegistryError("a promotion needs a written reason")
        entries = self.entries()
        entry = self._find(entries, version)
        current = entry["status"]
        if to_status not in ALLOWED_TRANSITIONS[current]:
            raise RegistryError(f"cannot move {version} from {current} to {to_status}")
        if to_status == "validated":
            self._check_release_gate(entry, release)
        if to_status == "production":
            for other in entries:
                if other["status"] == "production":
                    other["history"].append(
                        self._event("production", "retired", f"replaced by {version}")
                    )
                    other["status"] = "retired"
        entry["history"].append(self._event(current, to_status, reason))
        entry["status"] = to_status
        self._save(entries)
        return entry

    # --- internals -----------------------------------------------------------------

    @staticmethod
    def _check_release_gate(entry: dict[str, Any], release: ReleaseConfig) -> None:
        evaluation = entry["evaluation_metrics"]
        if evaluation is None:
            raise RegistryError(f"{entry['model_version']} has no evaluation; run evaluate first")
        problems = []
        if evaluation["checkpoint_hash"] != entry["checkpoint_hash"]:
            problems.append("evaluation was run on a different checkpoint")
        if evaluation["dataset_version"] != entry["dataset_version"]:
            problems.append("evaluation used a different dataset version")
        if evaluation["split"] != release.required_split:
            problems.append(f"evaluation split is {evaluation['split']}, gate requires "
                            f"{release.required_split}")
        if evaluation["n_cases"] < release.min_cases:
            problems.append(f"{evaluation['n_cases']} cases < required {release.min_cases}")
        if evaluation["dice"] < release.min_mean_dice:
            problems.append(f"mean Dice {evaluation['dice']:.4f} < required {release.min_mean_dice}")
        if problems:
            raise RegistryError("release gate failed: " + "; ".join(problems))

    @staticmethod
    def _event(from_status: str | None, to_status: str, reason: str) -> dict[str, Any]:
        return {"from": from_status, "to": to_status, "at": utc_now(),
                "by": getpass.getuser(), "reason": reason}

    @staticmethod
    def _find(entries: list[dict[str, Any]], version: str) -> dict[str, Any]:
        for entry in entries:
            if entry["model_version"] == version:
                return entry
        raise RegistryError(f"model version {version!r} not found")

    def _save(self, entries: list[dict[str, Any]]) -> None:
        write_json(self.file, {"models": entries})


# --- CLI -------------------------------------------------------------------------------


def _print_table(entries: list[dict[str, Any]]) -> None:
    print(f"{'VERSION':<10}{'STATUS':<12}{'MODEL_ID':<20}{'EXPERIMENT':<12}{'TEST DICE':<11}DATASET")
    for e in entries:
        evaluation = e["evaluation_metrics"]
        dice = f"{evaluation['dice']:.4f}" if evaluation else "-"
        print(f"{e['model_version']:<10}{e['status']:<12}{e['model_id']:<20}"
              f"{e['experiment_id']:<12}{dice:<11}{e['dataset_version']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Local model registry.")
    parser.add_argument("--artifacts-dir", default=Path("artifacts"), type=Path)
    commands = parser.add_subparsers(dest="command", required=True)

    register = commands.add_parser("register", help="register an experiment's best checkpoint")
    register.add_argument("--experiment", required=True)
    register.add_argument("--version", required=True)
    register.add_argument("--note", default="")

    commands.add_parser("list", help="list registered models")

    inspect = commands.add_parser("inspect", help="show one registry entry")
    inspect.add_argument("--version", required=True)

    promote = commands.add_parser("promote", help="move a model to a new status")
    promote.add_argument("--version", required=True)
    promote.add_argument("--to", required=True, choices=STATUSES[1:])
    promote.add_argument("--reason", required=True)
    promote.add_argument("--release-config", default=Path("configs/release.yaml"), type=Path)

    args = parser.parse_args(argv)
    registry = ModelRegistry(ArtifactPaths(args.artifacts_dir))
    try:
        if args.command == "register":
            entry = registry.register(args.experiment, args.version, args.note)
            print(f"registered {entry['model_version']} ({entry['model_id']}) as candidate")
        elif args.command == "list":
            _print_table(registry.entries())
        elif args.command == "inspect":
            print(json.dumps(registry.get(args.version), indent=2))
        elif args.command == "promote":
            release = load_config(args.release_config, ReleaseConfig)
            entry = registry.promote(args.version, args.to, args.reason, release)
            print(f"{entry['model_version']} is now {entry['status']}")
    except RegistryError as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
