"""End-to-end CPU training smoke tests over controlled canonical H5 cuts."""

import csv
import json
from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch
import xarray as xr

from dccnn_arpes.data.discovery import write_manifest_csv
from dccnn_arpes.data.pairing import PairRecord, read_pairs_csv, write_pairs_csv
from dccnn_arpes.data.schema import ManifestRecord
from dccnn_arpes.io.xarray_h5 import write_cut
from dccnn_arpes.models import ResidualDenoiser2D
from dccnn_arpes.training.checkpoints import load_checkpoint
from dccnn_arpes.training.config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    NoiseConfig,
    PathConfig,
    SamplingConfig,
    TrainConfig,
    TrainingConfig,
)
from dccnn_arpes.training.trainer import _dataset, _read_manifest, run_training


def _write_cut_record(tmp_path, record_id, split, *, nonfinite=False):
    axis = np.linspace(-1.0, 1.0, 192, dtype=np.float32)
    values = np.outer(np.exp(-(axis**2) * 3), 1.2 + np.cos(axis * 4)).astype(np.float32)
    path = tmp_path / f"{record_id}.h5"
    write_cut(
        xr.DataArray(
            values,
            dims=("eV", "alpha"),
            coords={"eV": axis, "alpha": axis * 12},
            name=record_id,
        ),
        path,
    )
    if nonfinite:
        with h5py.File(path, "r+") as stream:
            stream[record_id][0, 0] = np.nan
    return ManifestRecord(
        record_id=record_id,
        source_path=str(tmp_path / f"{record_id}.pxt"),
        converted_path=str(path),
        review_status="reviewed",
        split=split,
    )


def _training_config(tmp_path, *, nonfinite=False):
    records = [
        _write_cut_record(tmp_path, "controlled-train", "train", nonfinite=nonfinite),
        _write_cut_record(tmp_path, "controlled-val", "val"),
    ]
    manifest = tmp_path / "records.csv"
    pairs = tmp_path / "pairs.csv"
    splits = tmp_path / "splits"
    write_manifest_csv(records, manifest)
    write_pairs_csv([], pairs)
    (tmp_path / "data_provenance.json").write_text(
        json.dumps(
            {
                "kind": "controlled_canonical_h5",
                "generator": "tests/training/test_trainer_smoke.py",
                "record_ids": [record.record_id for record in records],
            }
        ),
        encoding="utf-8",
    )
    write_manifest_csv([records[0]], splits / "train.csv")
    write_manifest_csv([records[1]], splits / "val.csv")
    write_manifest_csv([], splits / "test.csv")
    return TrainConfig(
        paths=PathConfig(manifest=manifest, pairs=pairs, splits=splits, output=tmp_path / "runs"),
        seed=20260727,
        model=ModelConfig(name="residual_denoiser_2d", channels=4, blocks=1),
        data=DataConfig(
            crop_size=(192, 192),
            samples_per_epoch=1,
            sampling=SamplingConfig(A=0.0, B=0.0, C=1.0),
            identity_probability=0.0,
            noise=NoiseConfig(
                poisson_peak_counts=(500.0, 500.0),
                background_fraction=(0.0, 0.0),
                stripe_probability=0.0,
                stripe_fraction=(0.0, 0.0),
            ),
        ),
        training=TrainingConfig(
            batch_size=1,
            epochs=2,
            learning_rate=1.0e-4,
            weight_decay=1.0e-4,
            workers=0,
            device="cpu",
            amp=False,
        ),
        loss=LossConfig(charbonnier=0.8, ms_ssim=0.15, gradient=0.05),
    )


def test_two_epoch_cpu_training_writes_metrics_checkpoints_and_provenance(tmp_path):
    """Dropping any reproducibility artifact or component metric must fail the smoke contract."""
    config = _training_config(tmp_path)

    result = run_training(config)

    assert result.best_checkpoint.is_file()
    assert result.last_checkpoint.is_file()
    with result.metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert set(rows[0]) == {
        "epoch",
        "train_total",
        "train_charbonnier",
        "train_ms_ssim",
        "train_gradient",
        "val_total",
        "val_charbonnier",
        "val_ms_ssim",
        "val_gradient",
    }
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["seed"] == 20260727
    assert len(run["manifest_sha256"]) == 64
    assert len(run["split_sha256"]) == 64
    assert len(run["pairs_sha256"]) == 64
    assert len(run["data_sha256"]) == 64
    assert set(run["data_file_sha256"]) == {"controlled-train", "controlled-val"}
    assert run["git_commit"]
    assert isinstance(run["git_dirty"], bool)
    assert "source_diff_sha256" in run
    assert run["python_version"]
    assert run["pytorch_version"] == torch.__version__
    assert "cuda_version" in run
    assert run["device"] == "cpu"
    assert run["data_provenance"]["kind"] == "controlled_canonical_h5"

    restored_model = ResidualDenoiser2D(channels=4, blocks=1)
    state = load_checkpoint(result.last_checkpoint, model=restored_model, map_location="cpu")
    assert state.epoch == 2
    checkpoint = torch.load(result.last_checkpoint, map_location="cpu", weights_only=True)
    reference_model = ResidualDenoiser2D(channels=4, blocks=1)
    reference_model.load_state_dict(checkpoint["model_state"])
    probe = torch.linspace(0, 1, 192 * 192).reshape(1, 1, 192, 192)
    torch.testing.assert_close(
        restored_model(probe)[0],
        reference_model(probe)[0],
        rtol=0,
        atol=0,
    )


def test_nonfinite_input_stops_immediately(tmp_path):
    """Allowing NaN data to enter optimization must never produce a checkpoint."""
    config = _training_config(tmp_path, nonfinite=True)

    with pytest.raises(FloatingPointError, match="nonfinite"):
        run_training(replace(config, training=replace(config.training, epochs=1)))
    run_directories = list(config.paths.output.iterdir())
    assert len(run_directories) == 1
    assert not list(run_directories[0].glob("*.pt"))


def test_repeated_cpu_run_is_metric_reproducible_and_uses_distinct_directory(tmp_path):
    """Unseeded model, data-loader, or epoch randomness must change repeated-run metrics."""
    config = _training_config(tmp_path)

    first = run_training(config)
    second = run_training(config)

    assert first.output_dir != second.output_dir
    assert first.metrics_path.read_bytes() == second.metrics_path.read_bytes()


def test_split_record_must_be_present_in_hashed_master_manifest(tmp_path):
    """A standalone split CSV must not bypass review provenance in the master manifest."""
    config = _training_config(tmp_path)
    write_manifest_csv([], config.paths.manifest)

    with pytest.raises(ValueError, match="not present in the master manifest"):
        run_training(replace(config, training=replace(config.training, epochs=1)))


def test_reviewed_pair_endpoints_must_exist_in_master_manifest(tmp_path):
    """A malformed reviewed pair must not be silently discarded by split filtering."""
    config = _training_config(tmp_path)
    write_pairs_csv(
        [PairRecord("bad-pair", "controlled-train", "missing-record", "A", "approved")],
        config.paths.pairs,
    )

    with pytest.raises(ValueError, match="missing-record.*master manifest"):
        run_training(replace(config, training=replace(config.training, epochs=1)))


def test_noise_ranges_are_sampled_reproducibly_by_seed_epoch_and_index(tmp_path):
    """Collapsing configured ranges to one midpoint must not make every C sample identical."""
    config = _training_config(tmp_path)
    ranged = replace(
        config,
        data=replace(
            config.data,
            noise=NoiseConfig(
                poisson_peak_counts=(100.0, 10_000.0),
                background_fraction=(0.0, 0.08),
                stripe_probability=0.3,
                stripe_fraction=(0.0, 0.05),
            ),
        ),
    )
    records = _read_manifest(config.paths.splits / "train.csv")
    pairs = read_pairs_csv(config.paths.pairs)
    first = _dataset(records, pairs, ranged)
    second = _dataset(records, pairs, ranged)
    first.set_epoch(3)
    second.set_epoch(3)

    first[0]
    second[0]
    epoch_three = first.noise_parameters
    assert epoch_three == second.noise_parameters
    assert 100.0 <= epoch_three.poisson_peak_counts <= 10_000.0
    assert 0.0 <= epoch_three.background_fraction <= 0.08
    assert 0.0 <= epoch_three.stripe_fraction <= 0.05

    first.set_epoch(4)
    first[0]
    assert first.noise_parameters != epoch_three


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime is unavailable")
def test_cuda_amp_smoke_keeps_composite_loss_gradients_finite(tmp_path):
    """An unsafe loss scale must not overflow the mandatory composite-loss backward pass."""
    config = _training_config(tmp_path)
    cuda_config = replace(
        config,
        training=replace(config.training, epochs=1, device="cuda", amp=True),
    )

    result = run_training(cuda_config)

    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["device"] == torch.cuda.get_device_name(0)
    assert result.last_checkpoint.is_file()
