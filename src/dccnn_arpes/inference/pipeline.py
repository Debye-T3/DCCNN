"""Safe file-level inference for canonical ARPES cuts."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import numpy as np
import torch
import xarray as xr

from dccnn_arpes.data.transforms import IntensityTransform
from dccnn_arpes.io import load_cut, write_cut
from dccnn_arpes.models import ResidualDenoiser2D
from dccnn_arpes.safety import guard_output_path
from dccnn_arpes.training.checkpoints import CheckpointState, load_checkpoint

from .tiling import tiled_predict


def _load_inference_model(
    checkpoint_path: Path,
) -> tuple[ResidualDenoiser2D, CheckpointState, str]:
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha256 = sha256(checkpoint_bytes).hexdigest()
    payload = torch.load(BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    config = payload.get("config")
    if not isinstance(config, dict):
        raise TypeError("checkpoint config must be a mapping")
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise TypeError("checkpoint model config must be a mapping")
    if model_config.get("name") != "residual_denoiser_2d":
        raise ValueError("checkpoint model name must be residual_denoiser_2d")
    channels = model_config.get("channels")
    blocks = model_config.get("blocks")
    if type(channels) is not int or channels <= 0 or type(blocks) is not int or blocks <= 0:
        raise ValueError("checkpoint model channels and blocks must be positive integers")

    model = ResidualDenoiser2D(channels=channels, blocks=blocks)
    snapshot_path: Path | None = None
    try:
        with NamedTemporaryFile(mode="wb", suffix=".pt", delete=False) as snapshot:
            snapshot.write(checkpoint_bytes)
            snapshot_path = Path(snapshot.name)
        state = load_checkpoint(snapshot_path, model=model, map_location="cpu")
    finally:
        if snapshot_path is not None and snapshot_path.exists():
            snapshot_path.unlink()
    model.eval()
    return model, state, checkpoint_sha256


def _inference_tile_size(state: CheckpointState) -> int:
    data_config = state.config.get("data")
    if not isinstance(data_config, dict):
        raise TypeError("checkpoint data config must be a mapping")
    crop_size = data_config.get("crop_size")
    if (
        not isinstance(crop_size, list | tuple)
        or len(crop_size) != 2
        or any(type(value) is not int or value <= 0 for value in crop_size)
    ):
        raise ValueError("checkpoint data crop_size must contain two positive integers")
    return min(crop_size)


def _write_new_cut(data: xr.DataArray, destination: Path) -> None:
    staging_path = destination.with_name(f".{destination.name}.{uuid4().hex}.stage")
    try:
        write_cut(data, staging_path)
        os.link(staging_path, destination)
    finally:
        if staging_path.exists():
            staging_path.unlink()


def denoise_file(input_path: Path, checkpoint_path: Path, output_dir: Path) -> Path:
    """Denoise one canonical cut to a new atomic HDF5 artifact."""
    source = Path(input_path)
    checkpoint = Path(checkpoint_path)
    destination_directory = guard_output_path(
        output_dir,
        input_sources=(source, checkpoint),
    )
    destination = guard_output_path(
        destination_directory / f"{source.stem}_denoised.h5",
        input_sources=(source, checkpoint),
    )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing file {destination}")

    cut = load_cut(source)
    model, checkpoint_state, checkpoint_sha256 = _load_inference_model(checkpoint)
    transform = IntensityTransform()
    statistics = transform.fit(cut.values)
    normalized = transform.forward(cut.values, statistics).astype(np.float32)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    tile_size = _inference_tile_size(checkpoint_state)
    predicted = tiled_predict(
        model,
        tensor,
        tile_size=tile_size,
        overlap=tile_size // 4,
    )
    physical = np.clip(
        transform.inverse(predicted.squeeze(0).squeeze(0).cpu().numpy(), statistics),
        0.0,
        None,
    ).astype(np.float32)

    attrs = dict(cut.attrs)
    attrs.update(
        {
            "denoising_model": "residual_denoiser_2d",
            "denoising_checkpoint_sha256": checkpoint_sha256,
            "denoising_timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "denoising_transform": json.dumps(
                {
                    "lower": statistics.lower,
                    "name": "clip-negative-log1p-robust-quantile",
                    "scale": statistics.scale,
                    "statistics": "input-derived",
                },
                sort_keys=True,
            ),
            "smoke_test": str(checkpoint_state.smoke_test).lower(),
            "scientific_use": str(checkpoint_state.scientific_use).lower(),
        }
    )
    result = xr.DataArray(
        physical,
        dims=cut.dims,
        coords={name: coordinate.copy(deep=True) for name, coordinate in cut.coords.items()},
        name=cut.name,
        attrs=attrs,
    )
    destination_directory.mkdir(parents=True, exist_ok=True)
    _write_new_cut(result, destination)
    return destination
