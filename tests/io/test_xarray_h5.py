"""Tests for the canonical xarray/HDF5 cut format."""

import numpy as np
import pytest
import xarray as xr

from dccnn_arpes.io import xarray_h5
from dccnn_arpes.io.xarray_h5 import load_cut, validate_cut, write_cut


def test_write_cut_round_trips_canonical_data(tmp_path, canonical_cut):
    """Writes a validated cut that xarray can load with metadata intact."""
    path = tmp_path / "cut001.h5"

    write_cut(canonical_cut, path)

    restored = xr.load_dataarray(path)
    np.testing.assert_array_equal(restored.values, canonical_cut.values)
    np.testing.assert_array_equal(restored.coords["eV"].values, canonical_cut.coords["eV"].values)
    np.testing.assert_array_equal(
        restored.coords["alpha"].values, canonical_cut.coords["alpha"].values
    )
    assert restored.dims == ("eV", "alpha")
    assert restored.attrs == canonical_cut.attrs


def test_engine_preference_skips_xarray_versions_without_the_option(monkeypatch, tmp_path, canonical_cut):
    """Keeps imports and canonical writes usable with xarray's older option set."""
    def unsupported_set_options(**kwargs):
        pytest.fail(f"unsupported option was configured: {kwargs}")

    monkeypatch.setattr(xarray_h5.xr, "get_options", dict)
    monkeypatch.setattr(xarray_h5.xr, "set_options", unsupported_set_options)

    xarray_h5._prefer_h5netcdf_for_default_loading()
    write_cut(canonical_cut, tmp_path / "compatible.h5")


def test_validate_cut_requires_eV_and_alpha_dimensions(canonical_cut):
    """Rejects cuts whose named axis no longer represents alpha."""
    with pytest.raises(ValueError, match="missing required dimension"):
        validate_cut(canonical_cut.rename({"alpha": "x"}))


def test_validate_cut_requires_strictly_monotonic_coordinates(canonical_cut):
    """Rejects repeated energy positions that break a physical axis."""
    with pytest.raises(ValueError, match="strictly monotonic"):
        validate_cut(canonical_cut.assign_coords(eV=[0.0, 0.0, 0.1, 0.2]))


def test_validate_cut_rejects_descending_unsigned_coordinate_step(canonical_cut):
    """Rejects unsigned coordinates whose decreasing step would otherwise wrap."""
    invalid = canonical_cut.assign_coords(eV=np.array([1, 0, 2, 3], dtype=np.uint16))

    with pytest.raises(ValueError, match="strictly monotonic"):
        validate_cut(invalid)


def test_validate_cut_normalizes_dimension_order_and_data_dtype(canonical_cut):
    """Returns eV-by-alpha float32 data without altering coordinates or attributes."""
    reversed_cut = canonical_cut.astype(np.float64).transpose("alpha", "eV")

    validated = validate_cut(reversed_cut)

    assert validated.dims == ("eV", "alpha")
    assert validated.dtype == np.dtype("float32")
    np.testing.assert_array_equal(validated.coords["eV"].values, canonical_cut.coords["eV"].values)
    np.testing.assert_array_equal(
        validated.coords["alpha"].values, canonical_cut.coords["alpha"].values
    )
    assert validated.attrs == canonical_cut.attrs


def test_validate_cut_rejects_nonfinite_data(canonical_cut):
    """Rejects NaN values instead of accepting corrupt intensity data."""
    invalid = canonical_cut.copy(data=np.full(canonical_cut.shape, np.nan, dtype=np.float32))

    with pytest.raises(ValueError, match="finite"):
        validate_cut(invalid)


def test_write_cut_refuses_to_replace_existing_file(tmp_path, canonical_cut):
    """Preserves an existing destination unless overwrite was requested explicitly."""
    path = tmp_path / "cut001.h5"
    write_cut(canonical_cut, path)

    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_cut(canonical_cut, path)


def test_load_cut_explains_required_canonical_format(tmp_path):
    """Names an unreadable file and points callers to the canonical format."""
    path = tmp_path / "not-a-cut.h5"
    path.write_bytes(b"not hdf5")

    with pytest.raises(ValueError, match="convert xarray/HDF5 is required") as error:
        load_cut(path)

    assert str(path) in str(error.value)
