"""Shared fixtures for ARPES data-boundary tests."""

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def canonical_cut() -> xr.DataArray:
    """Return a valid standard ARPES two-dimensional cut."""
    return xr.DataArray(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        dims=("eV", "alpha"),
        coords={"eV": np.linspace(-0.3, 0.1, 4), "alpha": np.linspace(-10, 10, 5)},
        name="cut001",
        attrs={"temperature_K": 20.0, "sample_id": "sample-a"},
    )
