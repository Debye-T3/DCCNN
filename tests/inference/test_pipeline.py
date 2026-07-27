"""Safe coordinate-preserving file inference tests."""

from hashlib import sha256

import numpy as np
import pytest
import torch
import xarray as xr

from dccnn_arpes.inference import denoise_file
from dccnn_arpes.inference import pipeline as inference_pipeline
from dccnn_arpes.io import write_cut, xarray_h5
from dccnn_arpes.models import ResidualDenoiser2D


def _sha256(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smoke_checkpoint(path):
    model = ResidualDenoiser2D(channels=4, blocks=1)
    torch.save(
        {
            "schema_version": 2,
            "model_state": model.state_dict(),
            "optimizer_state": {},
            "scaler_state": {},
            "epoch": 2,
            "best_metric": 0.25,
            "config": {
                "model": {
                    "name": "residual_denoiser_2d",
                    "channels": 4,
                    "blocks": 1,
                },
                "data": {"crop_size": [8, 8]},
                "smoke_test": True,
                "scientific_use": False,
            },
            "hashes": {"data_sha256": "a" * 64},
            "versions": {"device": "cpu"},
            "smoke_test": True,
            "scientific_use": False,
        },
        path,
    )


def test_denoise_file_preserves_source_coords_attrs_and_checkpoint_intent(tmp_path, canonical_cut):
    """Inference must not mutate the source or strip physical coordinates and provenance."""
    canonical_cut = canonical_cut.assign_coords(photon_energy_eV=21.2)
    input_path = tmp_path / "cut001.h5"
    checkpoint_path = tmp_path / "best.pt"
    output_dir = tmp_path / "denoised"
    write_cut(canonical_cut, input_path)
    _smoke_checkpoint(checkpoint_path)
    source_hash = _sha256(input_path)

    output_path = denoise_file(input_path, checkpoint_path, output_dir)

    assert output_path == output_dir / "cut001_denoised.h5"
    assert _sha256(input_path) == source_hash
    actual = xr.load_dataarray(output_path)
    assert actual.dims == canonical_cut.dims
    assert set(actual.coords) == set(canonical_cut.coords)
    for coordinate in canonical_cut.coords:
        np.testing.assert_array_equal(actual.coords[coordinate], canonical_cut.coords[coordinate])
    for key, value in canonical_cut.attrs.items():
        assert actual.attrs[key] == value
    assert actual.attrs["denoising_model"] == "residual_denoiser_2d"
    assert actual.attrs["denoising_checkpoint_sha256"] == _sha256(checkpoint_path)
    assert actual.attrs["denoising_timestamp_utc"].endswith("Z")
    assert "input-derived" in actual.attrs["denoising_transform"]
    assert actual.attrs["smoke_test"] == "true"
    assert actual.attrs["scientific_use"] == "false"


def test_denoise_file_refuses_to_overwrite_existing_destination(tmp_path, canonical_cut):
    """A repeated command must fail before changing an existing inference artifact."""
    input_path = tmp_path / "cut001.h5"
    checkpoint_path = tmp_path / "best.pt"
    output_dir = tmp_path / "denoised"
    write_cut(canonical_cut, input_path)
    _smoke_checkpoint(checkpoint_path)
    output_dir.mkdir()
    destination = output_dir / "cut001_denoised.h5"
    destination.write_bytes(b"existing output")
    destination_hash = _sha256(destination)

    with pytest.raises(FileExistsError, match="refusing to overwrite existing file"):
        denoise_file(input_path, checkpoint_path, output_dir)

    assert _sha256(destination) == destination_hash


def test_denoise_file_records_the_exact_checkpoint_snapshot_used(
    tmp_path, canonical_cut, monkeypatch
):
    """A concurrently replaced best checkpoint must not change the recorded model digest."""
    input_path = tmp_path / "cut001.h5"
    checkpoint_path = tmp_path / "best.pt"
    replacement_path = tmp_path / "replacement.pt"
    output_dir = tmp_path / "denoised"
    write_cut(canonical_cut, input_path)
    _smoke_checkpoint(checkpoint_path)
    _smoke_checkpoint(replacement_path)
    original_checkpoint_hash = _sha256(checkpoint_path)
    assert _sha256(replacement_path) != original_checkpoint_hash
    real_load_checkpoint = inference_pipeline.load_checkpoint

    def load_then_replace_source(path, **kwargs):
        state = real_load_checkpoint(path, **kwargs)
        checkpoint_path.write_bytes(replacement_path.read_bytes())
        return state

    monkeypatch.setattr(inference_pipeline, "load_checkpoint", load_then_replace_source)

    output_path = denoise_file(input_path, checkpoint_path, output_dir)

    actual = xr.load_dataarray(output_path)
    assert actual.attrs["denoising_checkpoint_sha256"] == original_checkpoint_hash
    assert _sha256(checkpoint_path) != original_checkpoint_hash


def test_denoise_file_does_not_clobber_destination_created_during_publish(
    tmp_path, canonical_cut, monkeypatch
):
    """Atomic publication must preserve a competing result that wins the destination name."""
    input_path = tmp_path / "cut001.h5"
    checkpoint_path = tmp_path / "best.pt"
    output_dir = tmp_path / "denoised"
    destination = output_dir / "cut001_denoised.h5"
    competing_bytes = b"competing inference output"
    write_cut(canonical_cut, input_path)
    _smoke_checkpoint(checkpoint_path)
    real_replace = xarray_h5.os.replace
    competing_output_created = False

    def replace_after_competing_publish(source, target):
        nonlocal competing_output_created
        if not competing_output_created:
            destination.write_bytes(competing_bytes)
            competing_output_created = True
        return real_replace(source, target)

    monkeypatch.setattr(xarray_h5.os, "replace", replace_after_competing_publish)

    with pytest.raises(FileExistsError):
        denoise_file(input_path, checkpoint_path, output_dir)

    assert destination.read_bytes() == competing_bytes
