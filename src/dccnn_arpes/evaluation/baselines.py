"""Coordinate-preserving traditional denoising baselines."""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, median_filter


def _rebuild(source: xr.DataArray, values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=source.dims,
        coords={name: coordinate.copy(deep=True) for name, coordinate in source.coords.items()},
        name=source.name,
        attrs=dict(source.attrs),
    )


def gaussian_baseline(data: xr.DataArray, *, sigma: float = 1.0) -> xr.DataArray:
    """Apply a Gaussian filter while preserving the complete DataArray contract."""
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    values = gaussian_filter(np.asarray(data.values), sigma=float(sigma), mode="reflect")
    return _rebuild(data, values)


def median_baseline(data: xr.DataArray, *, size: int = 3) -> xr.DataArray:
    """Apply a median filter while preserving the complete DataArray contract."""
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray")
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")
    values = median_filter(np.asarray(data.values), size=size, mode="reflect")
    return _rebuild(data, values)
