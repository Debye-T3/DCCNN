"""Integration tests for reproducible mixed A/B/C ARPES cut sampling."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch
import xarray as xr

from dccnn_arpes.data.dataset import ArpesCutDataset
from dccnn_arpes.data.noise import NoiseParameters
from dccnn_arpes.data.pairing import PairRecord
from dccnn_arpes.data.schema import ManifestRecord
from dccnn_arpes.data.transforms import IntensityTransform
from dccnn_arpes.io.xarray_h5 import write_cut


def _write_record(tmp_path, record_id, values, **changes):
    path = tmp_path / f"{record_id}.h5"
    cut = xr.DataArray(
        np.asarray(values, dtype=np.float32),
        dims=("eV", "alpha"),
        coords={
            "eV": np.linspace(-0.4, 0.2, np.shape(values)[0]),
            "alpha": np.linspace(-12.0, 12.0, np.shape(values)[1]),
        },
        name=record_id,
    )
    write_cut(cut, path)
    defaults = {
        "record_id": record_id,
        "source_path": str(tmp_path / f"{record_id}.pxt"),
        "converted_path": str(path),
        "split": "train",
        "review_status": "reviewed",
    }
    defaults.update(changes)
    return ManifestRecord(**defaults)


@pytest.fixture
def mixed_records(tmp_path):
    base = np.arange(1, 16 * 18 + 1, dtype=np.float32).reshape(16, 18)
    records = [
        _write_record(tmp_path, "a-short", base * 2.0, acquisition_time_s=2.0),
        _write_record(tmp_path, "a-long", base * 20.0 * 2.0, acquisition_time_s=20.0),
        _write_record(tmp_path, "b-one", base + 3.0),
        _write_record(tmp_path, "b-two", base + 7.0),
        _write_record(tmp_path, "c-clean", base + 11.0),
    ]
    pairs = [
        PairRecord("pair-a", "a-long", "a-short", "A"),
        PairRecord("pair-b", "b-one", "b-two", "B"),
    ]
    return records, pairs


def _dataset(records, pairs, **changes):
    defaults = {
        "crop_size": (8, 10),
        "samples_per_epoch": 1000,
        "sampling": {"A": 0.5, "B": 0.3, "C": 0.2},
        "identity_probability": 0.0,
        "base_seed": 90210,
        "noise_parameters": NoiseParameters(
            poisson_peak_counts=80.0,
            background_fraction=0.06,
            stripe_probability=1.0,
            stripe_fraction=0.04,
        ),
    }
    defaults.update(changes)
    return ArpesCutDataset(records, pairs, **defaults)


def test_fixed_manifest_samples_approximately_50_30_20(mixed_records):
    """Wrong mixture weighting must move at least one count outside tolerance."""
    records, pairs = mixed_records
    dataset = _dataset(records, pairs)

    counts = Counter(dataset[index][2]["pair_type"] for index in range(1000))

    assert counts["A"] == pytest.approx(500, abs=55)
    assert counts["B"] == pytest.approx(300, abs=50)
    assert counts["C"] == pytest.approx(200, abs=45)


def test_a_pair_uses_count_rates_shared_stats_and_identical_crop_origin(mixed_records):
    """Independent rate scaling, statistics, or crop origins must break the 2x relation."""
    records, pairs = mixed_records
    dataset = _dataset(
        records,
        pairs,
        samples_per_epoch=1,
        sampling={"A": 1.0, "B": 0.0, "C": 0.0},
    )

    input_tensor, target_tensor, metadata = dataset[0]
    stats = metadata["transform_stats"]
    transform = IntensityTransform()
    input_rate = transform.inverse(input_tensor.squeeze(0).numpy(), stats)
    target_rate = transform.inverse(target_tensor.squeeze(0).numpy(), stats)

    assert input_tensor.shape == target_tensor.shape == (1, 8, 10)
    assert input_tensor.dtype == target_tensor.dtype == torch.float32
    np.testing.assert_allclose(target_rate, input_rate * 2.0, rtol=1.0e-5, atol=1.0e-5)
    assert metadata["record_id"] == "a-short"
    assert metadata["pair_type"] == "A"
    assert metadata["crop_eV"] in range(9)
    assert metadata["crop_alpha"] in range(9)


def test_seed_and_epoch_control_noise_and_crop_reproducibly(mixed_records):
    """Omitting epoch or record ID from stable seeding must repeat or drift samples."""
    records, pairs = mixed_records
    first = _dataset(
        records,
        pairs,
        sampling={"A": 0.0, "B": 0.0, "C": 1.0},
        samples_per_epoch=1,
    )
    second = _dataset(
        records,
        pairs,
        sampling={"A": 0.0, "B": 0.0, "C": 1.0},
        samples_per_epoch=1,
    )

    sample_epoch_zero = first[0]
    repeated_epoch_zero = first[0]
    independent_epoch_zero = second[0]
    first.set_epoch(1)
    sample_epoch_one = first[0]

    torch.testing.assert_close(sample_epoch_zero[0], repeated_epoch_zero[0], rtol=0, atol=0)
    torch.testing.assert_close(sample_epoch_zero[0], independent_epoch_zero[0], rtol=0, atol=0)
    assert sample_epoch_zero[2] == repeated_epoch_zero[2] == independent_epoch_zero[2]
    assert not torch.equal(sample_epoch_zero[0], sample_epoch_one[0])


def test_identity_constraints_keep_input_equal_to_target(mixed_records):
    """Ignoring the identity probability must leave noisy input-target pairs."""
    records, pairs = mixed_records
    dataset = _dataset(records, pairs, identity_probability=1.0, samples_per_epoch=12)

    for index in range(len(dataset)):
        input_tensor, target_tensor, metadata = dataset[index]
        torch.testing.assert_close(input_tensor, target_tensor, rtol=0, atol=0)
        assert metadata["identity_constraint"] is True


def test_unscaled_a_pair_requires_explicit_manual_approval(tmp_path):
    """Treating ordinary reviewed rows as manual scale approval must admit invalid A data."""
    base = np.arange(1, 13 * 15 + 1, dtype=np.float32).reshape(13, 15)
    left = _write_record(tmp_path, "left", base)
    right = _write_record(tmp_path, "right", base + 1.0)
    ordinary = PairRecord("pair-a", "left", "right", "A", review_status="reviewed")

    with pytest.raises(ValueError, match="no usable acquisition scale"):
        _dataset(
            [left, right],
            [ordinary],
            sampling={"A": 1.0, "B": 0.0, "C": 0.0},
            samples_per_epoch=1,
        )

    approved = replace(ordinary, review_status="manually_approved")
    dataset = _dataset(
        [left, right],
        [approved],
        sampling={"A": 1.0, "B": 0.0, "C": 0.0},
        samples_per_epoch=1,
    )
    assert dataset[0][2]["pair_type"] == "A"


def test_a_pair_orientation_compares_a_shared_acquisition_field(tmp_path):
    """Comparing seconds from one record with sweeps from another must reverse this pair."""
    base = np.arange(1, 13 * 15 + 1, dtype=np.float32).reshape(13, 15)
    short = _write_record(
        tmp_path,
        "short",
        base * 100.0,
        acquisition_time_s=100.0,
        sweep_count=1,
    )
    long = _write_record(tmp_path, "long", base * 10.0, sweep_count=10)
    pair = PairRecord("pair-a", "short", "long", "A")
    dataset = _dataset(
        [short, long],
        [pair],
        sampling={"A": 1.0, "B": 0.0, "C": 0.0},
        samples_per_epoch=1,
    )

    assert dataset[0][2]["record_id"] == "short"
