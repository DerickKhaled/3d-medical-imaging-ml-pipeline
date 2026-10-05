"""Load and check the YAML config files.

Every setting has to be written in the config file. There are no hidden defaults,
and an unknown key (for example a typo) is an error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.utils.hashing import sha256_json, short_hash

Triple = tuple[int, int, int]
TripleF = tuple[float, float, float]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- data -------------------------------------------------------------------


class SplitConfig(StrictModel):
    val_fraction: float = Field(gt=0, lt=1)
    test_fraction: float = Field(ge=0, lt=1)
    seed: int

    @model_validator(mode="after")
    def _leaves_training_data(self) -> SplitConfig:
        if self.val_fraction + self.test_fraction >= 0.9:
            raise ValueError("val_fraction + test_fraction must leave at least 10% for training")
        return self


class VolumeLimits(StrictModel):
    """Acceptance rules for an input volume. Shipped with every released model."""

    min_size_xyz: Triple
    max_voxels: int = Field(gt=0)
    min_spacing_mm: float = Field(gt=0)
    max_spacing_mm: float = Field(gt=0)


class DataConfig(StrictModel):
    name: str
    root: Path
    images_dir: str
    labels_dir: str
    modality: Literal["CT", "MRI"]
    class_names: dict[int, str]
    split: SplitConfig
    limits: VolumeLimits

    @model_validator(mode="after")
    def _classes_are_contiguous(self) -> DataConfig:
        if sorted(self.class_names) != list(range(len(self.class_names))):
            raise ValueError("class_names keys must be 0..N-1 with 0 = background")
        return self


# --- preprocessing ----------------------------------------------------------


class IntensityConfig(StrictModel):
    mode: Literal["zscore", "ct_window"]
    clip_percentiles: tuple[float, float] | None
    ct_window_level: float | None
    ct_window_width: float | None

    @model_validator(mode="after")
    def _mode_fields_present(self) -> IntensityConfig:
        if self.mode == "zscore" and self.clip_percentiles is None:
            raise ValueError("intensity.mode=zscore needs clip_percentiles")
        if self.mode == "ct_window" and (
            self.ct_window_level is None or self.ct_window_width is None
        ):
            raise ValueError("intensity.mode=ct_window needs ct_window_level and ct_window_width")
        return self


class PreprocessingConfig(StrictModel):
    name: str
    orientation: str
    target_spacing_mm: TripleF
    intensity: IntensityConfig
    crop_or_pad_size_xyz: Triple

    @field_validator("orientation")
    @classmethod
    def _valid_orientation_code(cls, code: str) -> str:
        """A code like 'RAS' or 'LPS': one letter from each anatomical axis pair."""
        axis_pairs = [{"R", "L"}, {"A", "P"}, {"S", "I"}]
        used = [next((i for i, pair in enumerate(axis_pairs) if c in pair), None) for c in code]
        if len(code) != 3 or None in used or len(set(used)) != 3:
            raise ValueError(f"invalid orientation code {code!r}; expected e.g. 'RAS' or 'LPS'")
        return code

    @field_validator("target_spacing_mm")
    @classmethod
    def _positive_spacing(cls, spacing: TripleF) -> TripleF:
        if any(s <= 0 for s in spacing):
            raise ValueError("target_spacing_mm must be positive")
        return spacing

    @property
    def version(self) -> str:
        """Content-derived version: any parameter change produces a new version.

        Nobody has to remember to bump a number, and two identical configs can
        never claim different versions.
        """
        return f"pp-{self.name}-{short_hash(sha256_json(self.model_dump(mode='json')))}"


# --- model / training ---------------------------------------------------------


class ModelConfig(StrictModel):
    name: Literal["unet3d"]
    in_channels: int = Field(ge=1)
    num_classes: int = Field(ge=2)
    base_channels: int = Field(ge=4)
    depth: int = Field(ge=1, le=5)


class AugmentationConfig(StrictModel):
    flip_axes_xyz: list[int]
    intensity_scale: float = Field(ge=0)
    intensity_shift: float = Field(ge=0)
    noise_std: float = Field(ge=0)


class TrainingConfig(StrictModel):
    seed: int
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    num_workers: int = Field(ge=0)
    device: Literal["auto", "cpu", "cuda"]
    amp: bool
    max_train_samples: int | None
    augmentation: AugmentationConfig


class PostprocessingConfig(StrictModel):
    keep_largest_component: bool


class ExperimentConfig(StrictModel):
    """One training run = data + preprocessing + model + training + post-processing."""

    data_config: Path
    preprocessing_config: Path
    model: ModelConfig
    training: TrainingConfig
    postprocessing: PostprocessingConfig


# --- release / reconstruction ---------------------------------------------------


class ReleaseConfig(StrictModel):
    """Minimum evidence required before a model may be promoted to 'validated'."""

    required_split: Literal["test", "val"]
    min_mean_dice: float = Field(ge=0, le=1)
    min_cases: int = Field(ge=1)


class MeshConfig(StrictModel):
    formats: list[Literal["stl", "ply", "obj"]]
    upsample_factor: int = Field(ge=1, le=4)
    presmooth_sigma_vox: float = Field(ge=0, le=3)
    smoothing_iterations: int = Field(ge=0)
    marching_cubes_step: int = Field(ge=1)


# --- loading ----------------------------------------------------------------------

ConfigT = TypeVar("ConfigT", bound=StrictModel)


def load_config(path: str | Path, schema: type[ConfigT]) -> ConfigT:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return schema.model_validate(raw)


def load_experiment(
    path: str | Path,
) -> tuple[ExperimentConfig, DataConfig, PreprocessingConfig]:
    """Load an experiment config and the data/preprocessing configs it references."""
    experiment = load_config(path, ExperimentConfig)
    data = load_config(experiment.data_config, DataConfig)
    preprocessing = load_config(experiment.preprocessing_config, PreprocessingConfig)
    if experiment.model.num_classes != len(data.class_names):
        raise ValueError(
            f"model.num_classes={experiment.model.num_classes} but the dataset defines "
            f"{len(data.class_names)} classes"
        )
    divisor = 2**experiment.model.depth
    if any(size % divisor for size in preprocessing.crop_or_pad_size_xyz):
        raise ValueError(
            f"crop_or_pad_size_xyz {preprocessing.crop_or_pad_size_xyz} must be divisible by "
            f"2**depth={divisor} for the U-Net"
        )
    return experiment, data, preprocessing
