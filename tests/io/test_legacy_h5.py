"""Tests for the read-only legacy HDF5 adapter."""

import h5py
import numpy as np
import pytest

from dccnn_arpes.io.legacy_h5 import load_legacy_cut
from dccnn_arpes.io.xarray_h5 import load_cut


def _write_legacy_cut(path, spectrum, energy, thetax):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("spectrum", data=spectrum)
        handle.create_dataset("energy", data=energy)
        handle.create_dataset("thetax", data=thetax)
        handle.attrs["sample_id"] = "legacy-sample"
        handle.attrs["temperature_K"] = 20.0


def test_load_legacy_cut_adapts_root_datasets_and_serializable_attrs(tmp_path):
    """Maps legacy spectrum, energy, and thetax into the canonical cut shape."""
    path = tmp_path / "legacy.h5"
    spectrum = np.arange(12, dtype=np.float64).reshape(3, 4)
    _write_legacy_cut(path, spectrum, [-0.2, -0.1, 0.0], [-5.0, 0.0, 5.0, 10.0])

    cut = load_legacy_cut(path)

    np.testing.assert_array_equal(cut.values, spectrum)
    assert cut.dims == ("eV", "alpha")
    assert cut.name == "legacy"
    assert cut.attrs["legacy_source"] == str(path)
    assert cut.attrs["sample_id"] == "legacy-sample"
    assert cut.attrs["temperature_K"] == 20.0


def test_load_legacy_cut_transposes_only_when_coordinates_require_it(tmp_path):
    """Corrects a legacy spectrum whose two axes were stored reversed."""
    path = tmp_path / "transposed.h5"
    stored = np.arange(12, dtype=np.float32).reshape(4, 3)
    _write_legacy_cut(path, stored, [-0.2, -0.1, 0.0], [-5.0, 0.0, 5.0, 10.0])

    cut = load_legacy_cut(path)

    np.testing.assert_array_equal(cut.values, stored.T)
    assert cut.shape == (3, 4)


def test_load_legacy_cut_rejects_ambiguous_square_orientation(tmp_path):
    """Rejects square legacy data because either orientation would fit coordinates."""
    path = tmp_path / "ambiguous.h5"
    _write_legacy_cut(path, np.ones((3, 3)), [-0.2, -0.1, 0.0], [-5.0, 0.0, 5.0])

    with pytest.raises(ValueError, match="ambiguous"):
        load_legacy_cut(path)


def test_load_cut_uses_legacy_adapter_only_when_enabled(tmp_path):
    """Keeps legacy adaptation explicit at the public loading boundary."""
    path = tmp_path / "legacy.h5"
    _write_legacy_cut(path, np.arange(12).reshape(3, 4), [-0.2, -0.1, 0.0], [-5.0, 0.0, 5.0, 10.0])

    cut = load_cut(path, allow_legacy=True)

    assert cut.attrs["legacy_source"] == str(path)
