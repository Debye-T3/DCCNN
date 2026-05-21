"""Conversion engine — pure-Python orchestrator. No Qt dependency."""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from converter.readers.txt_reader import read_txt
from converter.readers.pxt_reader import read_pxt, load_bin
from converter.writer import write_h5
from converter.preview import generate_preview


MANUAL_PARAM_KEYS = [
    "sample_name", "sample_id",
    "position_x", "position_y", "position_z",
    "position_polar", "position_tilt", "position_azimuth",
    "temperature_K", "photon_energy_eV", "polarization", "slit",
    "work_function_eV",
]


def detect_format(path: Path) -> str:
    """Detect file format from extension. Returns 'txt', 'pxt', or 'bin'."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return "txt"
    elif suffix == ".pxt":
        return "pxt"
    elif suffix == ".bin":
        return "bin"
    raise ValueError(f"Unsupported file extension: {suffix}")


def merge_params(batch_defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge batch defaults with per-file overrides. Override wins if non-empty."""
    merged = dict(batch_defaults)
    for key, value in overrides.items():
        if value is not None and value != "":
            merged[key] = value
    return merged


def convert_file(
    path: Path,
    output_dir: Path,
    params: Dict[str, Any],
    *,
    preview_enabled: bool = False,
    preview_settings: Optional[Dict[str, Any]] = None,
    preview_base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Convert a single file to HDF5. Returns dict with status info.

    Args:
        path: source file path
        output_dir: directory for .h5 output
        params: merged manual + SES params
        preview_enabled: whether to generate preview PNG
        preview_settings: dict with cmap, pmin, pmax, use_log, use_kspace
        preview_base_dir: base dir for preview output (defaults to output_dir.parent / "previews")

    Returns:
        dict with keys: success (bool), message (str), output_path (str or None)
    """
    fmt = detect_format(path)

    if fmt == "txt":
        result = read_txt(path)
    elif fmt == "pxt":
        result = read_pxt(path)
    elif fmt == "bin":
        spectrum_3d = load_bin(path, (365, 571, 51), "float32")
        result = {
            "spectrum": spectrum_3d[:, :, 25],
            "energy": np.arange(spectrum_3d.shape[0], dtype=np.float32),
            "thetax": np.arange(spectrum_3d.shape[1], dtype=np.float32),
            "ses_params": {},
        }
    else:
        raise ValueError(f"Unknown format: {fmt}")

    spectrum = result["spectrum"]
    energy = result["energy"]
    thetax = result["thetax"]
    ses_params = result.get("ses_params", {})
    raw_channels = result.get("raw_channels")

    out_path = output_dir / (path.stem + ".h5")

    write_h5(
        spectrum, energy, thetax, out_path,
        source_format=fmt,
        source_path=str(path),
        manual_params=params,
        ses_params=ses_params,
        raw_channels=raw_channels,
        overwrite=True,
    )

    messages = [f"[OK] {path.name} -> {out_path}"]

    if preview_enabled:
        try:
            p_settings = preview_settings or {}
            prev_dir = preview_base_dir or (output_dir.parent / "previews")
            prev_path = prev_dir / (path.stem + ".png")
            generate_preview(
                spectrum, energy, thetax, prev_path,
                cmap=p_settings.get("cmap", "inferno"),
                pmin=p_settings.get("pmin", 1.0),
                pmax=p_settings.get("pmax", 99.5),
                use_log=p_settings.get("use_log", True),
                use_kspace=p_settings.get("use_kspace", False),
                hv=params.get("photon_energy_eV"),
                work_function=float(params.get("work_function_eV", 4.2)),
            )
            messages.append(f"  Preview saved -> {prev_path}")
        except Exception as exc:
            messages.append(f"  Preview failed: {exc}")

    return {"success": True, "message": "\n".join(messages), "output_path": str(out_path)}
