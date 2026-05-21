"""HDF5 writer for converted ARPES data with full metadata attributes."""

from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np


def write_h5(
    spectrum: np.ndarray,
    energy: np.ndarray,
    thetax: np.ndarray,
    destination: Path,
    *,
    source_format: str,
    source_path: str,
    manual_params: Dict[str, Any],
    ses_params: Dict[str, Any],
    raw_channels: Optional[np.ndarray] = None,
    overwrite: bool = False,
) -> None:
    """Write spectrum + axes + metadata to HDF5.

    Args:
        spectrum: 2D float32 array [energy, angle]
        energy: 1D float32 energy axis
        thetax: 1D float32 angle axis
        destination: output .h5 path
        source_format: "txt", "pxt", or "bin"
        source_path: original file path string
        manual_params: user-entered parameters dict
        ses_params: auto-extracted SES header parameters dict
        raw_channels: optional [C, H, W] array for multi-channel PXT data
        overwrite: if True, overwrite existing file
    """
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists. Use overwrite=True.")

    destination.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(destination, "w") as f:
        f.create_dataset("spectrum", data=spectrum)
        f.create_dataset("energy", data=energy)
        f.create_dataset("thetax", data=thetax)
        f.create_dataset("raw_channels", data=spectrum[None, ...].astype(np.float32))

        if raw_channels is not None:
            f.create_dataset("raw_channels_all", data=raw_channels)

        f.attrs["source_format"] = source_format
        f.attrs["source_path"] = source_path
        f.attrs["shape"] = str(tuple(int(d) for d in spectrum.shape))

        for key, value in manual_params.items():
            if value is not None and value != "":
                try:
                    f.attrs[key] = value
                except TypeError:
                    f.attrs[key] = str(value)

        for key, value in ses_params.items():
            if isinstance(value, bool):
                f.attrs[key] = int(value)
            elif isinstance(value, (int, float, str, np.integer, np.floating)):
                f.attrs[key] = value
            else:
                try:
                    f.attrs[key] = str(value)
                except Exception:
                    pass
