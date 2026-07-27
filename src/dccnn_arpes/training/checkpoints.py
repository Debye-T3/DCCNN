"""Versioned, resumable training checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import TrainConfig

_SCHEMA_VERSION = 2


def _metadata_value(value: object) -> str | int | float | bool | None:
    if value is None or type(value) in {str, int, float, bool}:
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class CheckpointState:
    epoch: int
    best_metric: float
    config: dict[str, object]
    hashes: dict[str, str]
    versions: dict[str, object]
    smoke_test: bool
    scientific_use: bool


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_metric: float,
    config: TrainConfig,
    hashes: dict[str, str],
    versions: dict[str, object],
) -> None:
    """Atomically save every state required to reproduce and continue a run."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "config": config.as_dict(),
        "hashes": dict(hashes),
        "versions": {key: _metadata_value(value) for key, value in versions.items()},
        "smoke_test": config.smoke_test,
        "scientific_use": config.scientific_use,
    }
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointState:
    """Restore supplied stateful objects and return checkpoint metadata."""
    payload: dict[str, Any] = torch.load(Path(path), map_location=map_location, weights_only=True)
    if "schema_version" not in payload:
        raise ValueError("checkpoint is missing required key(s): schema_version")
    schema_version = payload["schema_version"]
    if type(schema_version) is int and schema_version == 1:
        raise ValueError(
            "legacy checkpoint schema 1 has no trustworthy run-intent metadata "
            "and cannot be resumed"
        )
    if type(schema_version) is not int or schema_version != _SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema {schema_version!r}")
    required = {
        "schema_version",
        "model_state",
        "optimizer_state",
        "scaler_state",
        "epoch",
        "best_metric",
        "config",
        "hashes",
        "versions",
        "smoke_test",
        "scientific_use",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint is missing required key(s): {', '.join(sorted(missing))}")
    for field in ("smoke_test", "scientific_use"):
        if type(payload[field]) is not bool:
            raise TypeError(f"checkpoint {field} must be a boolean")
    smoke_test = payload["smoke_test"]
    scientific_use = payload["scientific_use"]
    if smoke_test == scientific_use:
        raise ValueError("checkpoint intent flags must be complementary")
    config = payload["config"]
    if not isinstance(config, dict):
        raise TypeError("checkpoint config must be a mapping")
    for field in ("smoke_test", "scientific_use"):
        if type(config.get(field)) is not bool:
            raise TypeError(f"checkpoint embedded config {field} must be a boolean")
    if config["smoke_test"] is not smoke_test or config["scientific_use"] is not scientific_use:
        raise ValueError("checkpoint intent flags do not match embedded config")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler_state"])
    return CheckpointState(
        epoch=int(payload["epoch"]),
        best_metric=float(payload["best_metric"]),
        config=dict(config),
        hashes=dict(payload["hashes"]),
        versions=dict(payload["versions"]),
        smoke_test=smoke_test,
        scientific_use=scientific_use,
    )
