"""End-to-end CPU training smoke tests over controlled canonical H5 cuts."""

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import xarray as xr

from dccnn_arpes import safety
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
from dccnn_arpes.training.trainer import _dataset, _provenance, _read_manifest, run_training


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


def _write_provenance(tmp_path, records, *, classification="controlled_smoke_fixture"):
    scientific_use = classification == "reviewed_scientific_dataset"
    (tmp_path / "data_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "classification": classification,
                "scientific_use": scientific_use,
                "record_ids": [record.record_id for record in records],
                "input_sha256": {
                    record.record_id: hashlib.sha256(
                        Path(record.converted_path).read_bytes()
                    ).hexdigest()
                    for record in records
                },
                "generator": "tests/training/test_trainer_smoke.py",
            }
        ),
        encoding="utf-8",
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
    _write_provenance(tmp_path, records)
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
        smoke_test=True,
        scientific_use=False,
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
    assert run["data_provenance"]["classification"] == "controlled_smoke_fixture"
    assert run["smoke_test"] is True
    assert run["scientific_use"] is False
    assert run["status"] == "completed"
    assert run["started_at_utc"]
    assert run["completed_at_utc"]
    assert run["output_dir"] == str(result.output_dir)
    assert not list(result.output_dir.glob("run.json.tmp"))

    restored_model = ResidualDenoiser2D(channels=4, blocks=1)
    state = load_checkpoint(result.last_checkpoint, model=restored_model, map_location="cpu")
    assert state.epoch == 2
    checkpoint = torch.load(result.last_checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["smoke_test"] is True
    assert checkpoint["scientific_use"] is False
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
    failed_run = json.loads((run_directories[0] / "run.json").read_text(encoding="utf-8"))
    assert failed_run["status"] == "failed"
    assert failed_run["started_at_utc"]
    assert failed_run["completed_at_utc"]
    assert failed_run["output_dir"] == str(run_directories[0])
    assert failed_run["error_class"] == "FloatingPointError"
    assert not list(run_directories[0].glob("run.json.tmp"))


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


def test_tampered_test_partition_row_is_rejected(tmp_path):
    """Hashing test.csv without parsing it must not admit a row differing from the master."""
    config = _training_config(tmp_path)
    records = _read_manifest(config.paths.manifest)
    test_record = _write_cut_record(tmp_path, "controlled-test", "test")
    records.append(test_record)
    write_manifest_csv(records, config.paths.manifest)
    write_manifest_csv([replace(test_record, notes="tampered")], config.paths.splits / "test.csv")
    _write_provenance(tmp_path, records)

    with pytest.raises(ValueError, match="does not match the master manifest"):
        run_training(replace(config, training=replace(config.training, epochs=1)))


def test_eligible_master_row_omitted_from_declared_partition_is_rejected(tmp_path):
    """Every reviewed converted master row must appear once in its declared partition."""
    config = _training_config(tmp_path)
    write_manifest_csv([], config.paths.splits / "val.csv")

    with pytest.raises(ValueError, match="controlled-val.*missing from val"):
        run_training(replace(config, training=replace(config.training, epochs=1)))


def test_record_duplicated_across_partitions_is_rejected(tmp_path):
    """The same record ID must not be accepted in two partition files."""
    config = _training_config(tmp_path)
    val_record = _read_manifest(config.paths.splits / "val.csv")[0]
    write_manifest_csv([val_record], config.paths.splits / "test.csv")

    with pytest.raises(ValueError, match="controlled-val.*multiple partitions"):
        run_training(replace(config, training=replace(config.training, epochs=1)))


def test_connected_group_leakage_is_rejected_before_output_creation(tmp_path):
    """Self-consistent split rows must not bypass sample/acquisition/source connectivity."""
    config = _training_config(tmp_path)
    train_record, val_record = _read_manifest(config.paths.manifest)
    train_record = replace(train_record, sample_id="shared-sample")
    val_record = replace(val_record, sample_id="shared-sample")
    records = [train_record, val_record]
    write_manifest_csv(records, config.paths.manifest)
    write_manifest_csv([train_record], config.paths.splits / "train.csv")
    write_manifest_csv([val_record], config.paths.splits / "val.csv")
    _write_provenance(tmp_path, records)

    with pytest.raises(ValueError, match="connected component.*multiple splits"):
        run_training(replace(config, training=replace(config.training, epochs=1)))

    assert not config.paths.output.exists()


def test_training_rejects_output_inside_a_resolved_read_only_root_before_mkdir(
    tmp_path, monkeypatch
):
    """Training must share the central guard and leave protected data roots untouched."""
    config = _training_config(tmp_path)
    read_only_root = tmp_path / "read-only"
    read_only_root.mkdir()
    forbidden_output = read_only_root / "training-runs"
    monkeypatch.setattr(safety, "READ_ONLY_DATA_ROOTS", (read_only_root,))
    config = replace(
        config,
        paths=replace(config.paths, output=forbidden_output),
        training=replace(config.training, epochs=1),
    )

    with pytest.raises(ValueError, match="read-only data root"):
        run_training(config)

    assert not forbidden_output.exists()


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


def test_missing_provenance_is_a_failed_run_not_an_invented_fallback(tmp_path):
    """A missing provenance file must never be labeled as a reviewed manifest."""
    config = _training_config(tmp_path)
    (tmp_path / "data_provenance.json").unlink()

    with pytest.raises(FileNotFoundError, match="data_provenance.json"):
        run_training(replace(config, training=replace(config.training, epochs=1)))

    run_directory = next(config.paths.output.iterdir())
    run = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert run["error_class"] == "FileNotFoundError"


def test_malformed_provenance_json_is_rejected(tmp_path):
    """Malformed provenance must fail explicitly instead of falling back."""
    config = _training_config(tmp_path)
    (tmp_path / "data_provenance.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON.*data_provenance.json"):
        run_training(replace(config, training=replace(config.training, epochs=1)))


def test_hash_mismatched_provenance_is_rejected(tmp_path):
    """Declared record IDs and hashes must exactly describe partition inputs."""
    config = _training_config(tmp_path)
    provenance_path = tmp_path / "data_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["input_sha256"]["controlled-train"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="input_sha256"):
        run_training(replace(config, training=replace(config.training, epochs=1)))


def test_non_smoke_run_rejects_controlled_fixture_provenance(tmp_path):
    """Controlled fixture approval must never authorize scientific training."""
    config = replace(_training_config(tmp_path), smoke_test=False, scientific_use=True)

    with pytest.raises(ValueError, match="reviewed_scientific_dataset"):
        run_training(replace(config, training=replace(config.training, epochs=1)))


def test_same_manifest_selects_distinct_smoke_and_scientific_provenance_files(tmp_path):
    """The run intent must select explicit authority instead of a fixed sibling filename."""
    config = _training_config(tmp_path)
    records = _read_manifest(config.paths.manifest)
    default_provenance = tmp_path / "data_provenance.json"
    smoke_provenance = tmp_path / "controlled-smoke-provenance.json"
    scientific_provenance = tmp_path / "reviewed-scientific-provenance.json"
    default_provenance.replace(smoke_provenance)
    _write_provenance(tmp_path, records, classification="reviewed_scientific_dataset")
    default_provenance.replace(scientific_provenance)
    paths = replace(
        config.paths,
        provenance_path=scientific_provenance,
        smoke_provenance_path=smoke_provenance,
    )
    file_hashes = {
        record.record_id: hashlib.sha256(Path(record.converted_path).read_bytes()).hexdigest()
        for record in records
    }

    smoke = replace(config, paths=paths).for_smoke_test(device="cpu")
    scientific = replace(
        config,
        paths=paths,
        smoke_test=False,
        scientific_use=True,
    )

    assert _provenance(smoke, file_hashes)["classification"] == "controlled_smoke_fixture"
    assert _provenance(scientific, file_hashes)["classification"] == "reviewed_scientific_dataset"


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
