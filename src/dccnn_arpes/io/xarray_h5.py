"""Canonical xarray/HDF5 boundary for ARPES two-dimensional cuts."""

import os
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import xarray as xr

_CUT_DIMS = ("eV", "alpha")

# Prefer the bundled pure-HDF5 backend for xarray's unqualified loading path.
# netCDF4 remains installed as a fallback for formats h5netcdf cannot open.
xr.set_options(netcdf_engine_order=("h5netcdf", "netcdf4", "scipy"))


def _is_real_numeric(dtype: np.dtype) -> bool:
    """Return whether a dtype can be represented by the canonical float32 format."""
    return np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.complexfloating)


def _validate_axis(data: xr.DataArray, dimension: str) -> None:
    """Check that one physical coordinate is a finite, strictly monotonic axis."""
    if dimension not in data.coords:
        raise ValueError(f"missing dimension coordinate: {dimension}")
    coordinate = data.coords[dimension]
    if coordinate.ndim != 1 or coordinate.dims != (dimension,):
        raise ValueError(f"dimension coordinate {dimension} must be one-dimensional")
    if coordinate.size != data.sizes[dimension]:
        raise ValueError(f"dimension coordinate {dimension} has a mismatched length")
    if not _is_real_numeric(coordinate.dtype):
        raise ValueError(f"dimension coordinate {dimension} must be numeric")

    values = np.asarray(coordinate.values)
    if not np.isfinite(values).all():
        raise ValueError(f"dimension coordinate {dimension} must contain only finite values")
    differences = np.diff(values.astype(np.float64))
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError(f"dimension coordinate {dimension} must be strictly monotonic")


def validate_cut(data: xr.DataArray) -> xr.DataArray:
    """Validate and normalize a standard two-dimensional ARPES cut."""
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray")
    if data.ndim != 2:
        raise ValueError("cut must have exactly two dimensions")
    missing_dimensions = set(_CUT_DIMS).difference(data.dims)
    if missing_dimensions:
        raise ValueError(f"missing required dimension(s): {', '.join(sorted(missing_dimensions))}")
    if set(data.dims) != set(_CUT_DIMS):
        raise ValueError("cut dimensions must be exactly eV and alpha")

    normalized = data.transpose(*_CUT_DIMS)
    if normalized.size == 0:
        raise ValueError("cut must not be empty")
    if not _is_real_numeric(normalized.dtype):
        raise ValueError("cut data must be numeric")
    values = np.asarray(normalized.values)
    if not np.isfinite(values).all():
        raise ValueError("cut data must contain only finite values")
    for dimension in _CUT_DIMS:
        _validate_axis(normalized, dimension)

    return normalized.astype(np.float32)


def load_cut(path: Path, *, allow_legacy: bool = False) -> xr.DataArray:
    """Load a canonical cut, explicitly opting into legacy adaptation when needed."""
    path = Path(path)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The 'phony_dims' kwarg now defaults to 'access'.*",
                category=UserWarning,
            )
            loaded = xr.load_dataarray(path)
    except Exception as canonical_error:
        if allow_legacy:
            from .legacy_h5 import load_legacy_cut

            return load_legacy_cut(path)
        raise ValueError(
            f"could not load {path} as canonical xarray/HDF5; convert xarray/HDF5 is required"
        ) from canonical_error
    return validate_cut(loaded)


def write_cut(data: xr.DataArray, path: Path, *, overwrite: bool = False) -> None:
    """Atomically write a validated canonical cut to HDF5."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file {path}; pass overwrite=True")

    normalized = validate_cut(data)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb", suffix=".h5", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        normalized.to_netcdf(temporary_path, engine="h5netcdf")
        validate_cut(xr.load_dataarray(temporary_path, engine="h5netcdf"))
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
