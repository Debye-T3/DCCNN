"""Read-only adapter for legacy ARPES HDF5 files."""

from pathlib import Path

import h5py
import numpy as np
import xarray as xr


def _serializable(value: object) -> object:
    """Convert h5py attribute values into ordinary serializable Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return _serializable(value.item())
    if isinstance(value, np.ndarray):
        return [_serializable(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_serializable(item) for item in value]
    return value


def load_legacy_cut(path: Path) -> xr.DataArray:
    """Adapt a legacy ``spectrum``/``energy``/``thetax`` file without modifying it."""
    path = Path(path)
    required = ("spectrum", "energy", "thetax")
    with h5py.File(path, "r") as handle:
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"legacy HDF5 file {path} is missing required dataset(s): {', '.join(missing)}")
        spectrum = np.asarray(handle["spectrum"][()])
        energy = np.asarray(handle["energy"][()])
        thetax = np.asarray(handle["thetax"][()])
        root_attrs = {key: _serializable(value) for key, value in handle.attrs.items()}

    if spectrum.ndim != 2 or energy.ndim != 1 or thetax.ndim != 1:
        raise ValueError(f"legacy HDF5 file {path} must contain 2D spectrum and 1D energy/thetax")

    expected_shape = (energy.size, thetax.size)
    normal_matches = spectrum.shape == expected_shape
    transposed_matches = spectrum.T.shape == expected_shape
    if normal_matches and transposed_matches:
        raise ValueError(f"legacy HDF5 file {path} has ambiguous spectrum orientation")
    if not normal_matches and not transposed_matches:
        raise ValueError(
            f"legacy HDF5 file {path} spectrum shape {spectrum.shape} does not match "
            f"energy/thetax lengths {expected_shape}"
        )
    if transposed_matches:
        spectrum = spectrum.T

    cut = xr.DataArray(
        spectrum,
        dims=("eV", "alpha"),
        coords={"eV": energy, "alpha": thetax},
        name=path.stem,
        attrs={"legacy_source": str(path), **root_attrs},
    )
    from .xarray_h5 import validate_cut

    return validate_cut(cut)
