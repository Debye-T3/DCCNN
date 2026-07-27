"""Strict, serializable configuration for reproducible denoising training."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


def _keys(
    mapping: object,
    required: set[str],
    context: str,
    *,
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise TypeError(f"{context} must be a mapping")
    allowed = required | (optional or set())
    unknown = set(mapping).difference(allowed)
    missing = required.difference(mapping)
    if unknown:
        raise ValueError(f"{context} has unknown key(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"{context} is missing required key(s): {', '.join(sorted(missing))}")
    return mapping


def _pair(value: object, context: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{context} must contain exactly two values")
    result = (_number(value[0], context), _number(value[1], context))
    if result[0] > result[1]:
        raise ValueError(f"{context} must be a finite ascending range")
    return result


def _integer(value: object, context: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{context} must be an integer")
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _text(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be text")
    return value


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{context} must be boolean")
    return value


@dataclass(frozen=True, slots=True)
class PathConfig:
    manifest: Path
    pairs: Path
    splits: Path
    output: Path
    provenance_path: Path | None = None
    smoke_provenance_path: Path | None = None

    def __post_init__(self) -> None:
        for field in (
            "manifest",
            "pairs",
            "splits",
            "output",
            "provenance_path",
            "smoke_provenance_path",
        ):
            value = getattr(self, field)
            object.__setattr__(self, field, None if value is None else Path(value))


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    channels: int
    blocks: int

    def __post_init__(self) -> None:
        if type(self.channels) is not int or type(self.blocks) is not int:
            raise TypeError("model channels and blocks must be integers")
        if self.name != "residual_denoiser_2d":
            raise ValueError("model.name must be residual_denoiser_2d")
        if self.channels <= 0 or self.blocks <= 0:
            raise ValueError("model channels and blocks must be positive")


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    A: float
    B: float
    C: float

    def __post_init__(self) -> None:
        values = (self.A, self.B, self.C)
        if not all(
            not isinstance(value, bool)
            and isinstance(value, int | float)
            and math.isfinite(value)
            and value >= 0
            for value in values
        ):
            raise ValueError("data.sampling weights must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-8):
            raise ValueError("data.sampling weights must sum to one within 1e-8")

    def as_dict(self) -> dict[str, float]:
        return {"A": self.A, "B": self.B, "C": self.C}


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    poisson_peak_counts: tuple[float, float]
    background_fraction: tuple[float, float]
    stripe_probability: float
    stripe_fraction: tuple[float, float]

    def __post_init__(self) -> None:
        values = (
            *self.poisson_peak_counts,
            *self.background_fraction,
            self.stripe_probability,
            *self.stripe_fraction,
        )
        if not all(
            not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)
            for value in values
        ):
            raise ValueError("noise parameters must be finite numbers")
        if (
            self.poisson_peak_counts[0] > self.poisson_peak_counts[1]
            or self.background_fraction[0] > self.background_fraction[1]
            or self.stripe_fraction[0] > self.stripe_fraction[1]
        ):
            raise ValueError("noise ranges must be ascending")
        if self.poisson_peak_counts[0] <= 0:
            raise ValueError("poisson peak counts must be positive")
        if self.background_fraction[0] < 0 or self.stripe_fraction[0] < 0:
            raise ValueError("noise fractions must be non-negative")
        if not 0 <= self.stripe_probability <= 1:
            raise ValueError("stripe probability must be between zero and one")


@dataclass(frozen=True, slots=True)
class DataConfig:
    crop_size: tuple[int, int]
    samples_per_epoch: int
    sampling: SamplingConfig
    identity_probability: float
    noise: NoiseConfig

    def __post_init__(self) -> None:
        if (
            len(self.crop_size) != 2
            or any(type(size) is not int for size in self.crop_size)
            or any(size <= 0 for size in self.crop_size)
        ):
            raise ValueError("data.crop_size must contain two positive dimensions")
        if type(self.samples_per_epoch) is not int or self.samples_per_epoch <= 0:
            raise ValueError("data.samples_per_epoch must be positive")
        if not math.isfinite(self.identity_probability) or not 0 <= self.identity_probability <= 1:
            raise ValueError("data.identity_probability must be between zero and one")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    workers: int
    device: str
    amp: bool

    def __post_init__(self) -> None:
        if (
            type(self.batch_size) is not int
            or type(self.epochs) is not int
            or self.batch_size <= 0
            or self.epochs <= 0
        ):
            raise ValueError("training batch_size and epochs must be positive")
        if (
            not math.isfinite(self.learning_rate)
            or not math.isfinite(self.weight_decay)
            or self.learning_rate <= 0
            or self.weight_decay < 0
        ):
            raise ValueError(
                "training learning_rate must be positive and weight_decay non-negative"
            )
        if type(self.workers) is not int or self.workers < 0:
            raise ValueError("training.workers must be non-negative")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("training.device must be cpu or cuda")
        if not isinstance(self.amp, bool):
            raise TypeError("training.amp must be boolean")


@dataclass(frozen=True, slots=True)
class LossConfig:
    charbonnier: float
    ms_ssim: float
    gradient: float

    def __post_init__(self) -> None:
        values = (self.charbonnier, self.ms_ssim, self.gradient)
        if any(value < 0 or not math.isfinite(value) for value in values) or not math.isclose(
            sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise ValueError("loss weights must be nonnegative and sum to one within 1e-8")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    paths: PathConfig
    seed: int
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    loss: LossConfig
    smoke_test: bool = False
    scientific_use: bool = True

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not isinstance(self.smoke_test, bool) or not isinstance(self.scientific_use, bool):
            raise TypeError("smoke_test and scientific_use must be boolean")
        if self.smoke_test == self.scientific_use:
            raise ValueError("smoke runs must be non-scientific and non-smoke runs scientific")

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["paths"] = {
            key: None if value is None else str(value) for key, value in result["paths"].items()
        }
        result["data"]["sampling"] = self.data.sampling.as_dict()
        return result

    def for_smoke_test(self, *, device: str | None = None) -> TrainConfig:
        """Return bounded in-memory smoke settings without changing the source YAML."""
        training = replace(
            self.training,
            epochs=2,
            device=device or self.training.device,
            amp=self.training.amp and (device or self.training.device) == "cuda",
        )
        return replace(
            self,
            paths=replace(
                self.paths,
                output=self.paths.output / "smoke",
                provenance_path=self.paths.smoke_provenance_path or self.paths.provenance_path,
            ),
            model=replace(self.model, channels=4, blocks=1),
            data=replace(self.data, samples_per_epoch=2),
            training=training,
            smoke_test=True,
            scientific_use=False,
        )


def load_train_config(path: str | Path) -> TrainConfig:
    """Parse a YAML file while rejecting every unknown or omitted key."""
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        root = _keys(
            yaml.safe_load(stream) or {},
            {"paths", "seed", "model", "data", "training", "loss"},
            "config",
        )
    paths = _keys(
        root["paths"],
        {"manifest", "pairs", "splits", "output"},
        "paths",
        optional={"provenance_path", "smoke_provenance_path"},
    )
    model = _keys(root["model"], {"name", "channels", "blocks"}, "model")
    data = _keys(
        root["data"],
        {"crop_size", "samples_per_epoch", "sampling", "identity_probability", "noise"},
        "data",
    )
    sampling = _keys(data["sampling"], {"A", "B", "C"}, "data.sampling")
    noise = _keys(
        data["noise"],
        {
            "poisson_peak_counts",
            "background_fraction",
            "stripe_probability",
            "stripe_fraction",
        },
        "data.noise",
    )
    training = _keys(
        root["training"],
        {
            "batch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "workers",
            "device",
            "amp",
        },
        "training",
    )
    loss = _keys(root["loss"], {"charbonnier", "ms_ssim", "gradient"}, "loss")
    crop_size = data["crop_size"]
    if not isinstance(crop_size, list | tuple) or len(crop_size) != 2:
        raise ValueError("data.crop_size must contain exactly two values")
    return TrainConfig(
        paths=PathConfig(
            manifest=Path(_text(paths["manifest"], "paths.manifest")),
            pairs=Path(_text(paths["pairs"], "paths.pairs")),
            splits=Path(_text(paths["splits"], "paths.splits")),
            output=Path(_text(paths["output"], "paths.output")),
            provenance_path=(
                Path(_text(paths["provenance_path"], "paths.provenance_path"))
                if "provenance_path" in paths
                else None
            ),
            smoke_provenance_path=(
                Path(_text(paths["smoke_provenance_path"], "paths.smoke_provenance_path"))
                if "smoke_provenance_path" in paths
                else None
            ),
        ),
        seed=_integer(root["seed"], "seed"),
        model=ModelConfig(
            name=_text(model["name"], "model.name"),
            channels=_integer(model["channels"], "model.channels"),
            blocks=_integer(model["blocks"], "model.blocks"),
        ),
        data=DataConfig(
            crop_size=(
                _integer(crop_size[0], "data.crop_size"),
                _integer(crop_size[1], "data.crop_size"),
            ),
            samples_per_epoch=_integer(data["samples_per_epoch"], "data.samples_per_epoch"),
            sampling=SamplingConfig(
                **{key: _number(value, f"data.sampling.{key}") for key, value in sampling.items()}
            ),
            identity_probability=_number(data["identity_probability"], "data.identity_probability"),
            noise=NoiseConfig(
                poisson_peak_counts=_pair(
                    noise["poisson_peak_counts"], "data.noise.poisson_peak_counts"
                ),
                background_fraction=_pair(
                    noise["background_fraction"], "data.noise.background_fraction"
                ),
                stripe_probability=_number(
                    noise["stripe_probability"], "data.noise.stripe_probability"
                ),
                stripe_fraction=_pair(noise["stripe_fraction"], "data.noise.stripe_fraction"),
            ),
        ),
        training=TrainingConfig(
            batch_size=_integer(training["batch_size"], "training.batch_size"),
            epochs=_integer(training["epochs"], "training.epochs"),
            learning_rate=_number(training["learning_rate"], "training.learning_rate"),
            weight_decay=_number(training["weight_decay"], "training.weight_decay"),
            workers=_integer(training["workers"], "training.workers"),
            device=_text(training["device"], "training.device"),
            amp=_boolean(training["amp"], "training.amp"),
        ),
        loss=LossConfig(**{key: _number(value, f"loss.{key}") for key, value in loss.items()}),
    )
