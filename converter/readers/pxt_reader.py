"""Scienta PXT binary and raw .bin cube parser."""

import math
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def load_bin(path: Path, shape: Tuple[int, ...], dtype: str) -> np.ndarray:
    """Load a raw binary cube with given shape and dtype."""
    dtype_obj = np.dtype(dtype)
    expected_bytes = math.prod(shape) * dtype_obj.itemsize
    actual_bytes = path.stat().st_size
    if expected_bytes != actual_bytes:
        raise ValueError(
            f"{path}: size mismatch. Expected {expected_bytes} bytes, got {actual_bytes}."
        )
    data = np.fromfile(path, dtype=dtype_obj)
    return data.reshape(shape)


def read_pxt(
    path: Path,
    *,
    energy_offset_override: Optional[float] = None,
    energy_step_override: Optional[float] = None,
    angle_offset_override: Optional[float] = None,
    angle_step_override: Optional[float] = None,
    channel: int = 0,
    subtract_dark: bool = False,
) -> dict:
    """Read a Scienta PXT binary file.

    Returns dict with keys: spectrum (2D float32), energy (1D float32),
    thetax (1D float32), ses_params (dict), raw_channels (3D float32 array
    [channels, H, W]).
    """
    raw = path.read_bytes()
    if len(raw) < 256:
        raise ValueError(f"{path}: file too small to be a valid PXT container.")

    def _uint(idx: int) -> int:
        return struct.unpack_from("<I", raw, idx * 4)[0]

    def _double(idx: int) -> float:
        return struct.unpack_from("<d", raw, idx * 4)[0]

    total_points = _uint(21)
    channel_count = max(1, _uint(22))
    frame_type_bytes = raw[25 * 4: 27 * 4]
    frame_type = frame_type_bytes.split(b"\x00", 1)[0].decode("ascii", errors="ignore") or "unknown"

    width = _uint(35)
    height = _uint(36)
    if width == 0 or height == 0:
        raise ValueError(f"{path}: reported shape {width}x{height} is invalid.")

    energy_step_raw = _double(39)
    angle_step_raw = _double(41)
    energy_offset_raw = _double(47)
    angle_offset_raw = _double(49)

    energy_step = energy_step_override if energy_step_override is not None else energy_step_raw
    angle_step = angle_step_override if angle_step_override is not None else angle_step_raw
    energy_offset = energy_offset_override if energy_offset_override is not None else energy_offset_raw
    angle_offset = angle_offset_override if angle_offset_override is not None else angle_offset_raw

    itemsize = np.dtype("<i2").itemsize
    data_bytes = width * height * channel_count * itemsize
    header_bytes = len(raw) - data_bytes
    if header_bytes < 0:
        raise ValueError(f"{path}: negative header size computed.")

    payload = np.frombuffer(
        raw, dtype="<i2", count=width * height * channel_count, offset=header_bytes
    )
    payload = payload.reshape(height, width, channel_count)

    chosen_channel = channel
    if channel < 0:
        pos_means = []
        for ch in range(channel_count):
            ch_data = payload[..., ch].astype(np.float32)
            pos_means.append(float(np.mean(np.clip(ch_data, a_min=0.0, a_max=None))))
        chosen_channel = int(np.argmax(pos_means))

    if not 0 <= chosen_channel < channel_count:
        raise ValueError(
            f"{path}: channel {chosen_channel} out of range ({channel_count} channels)."
        )

    signal = payload[..., chosen_channel].astype(np.float32)
    subtracted_from = None
    if subtract_dark and channel_count > 1:
        dark_idx = 1 if chosen_channel == 0 else (chosen_channel - 1)
        if 0 <= dark_idx < channel_count:
            signal = signal - payload[..., dark_idx].astype(np.float32)
            subtracted_from = dark_idx

    signal = np.clip(signal, a_min=0.0, a_max=None)
    spectrum = signal.T.copy()

    energy_axis = (np.arange(width, dtype=np.float32) * energy_step + energy_offset).astype(np.float32)
    angle_axis = (np.arange(height, dtype=np.float32) * angle_step + angle_offset).astype(np.float32)

    ses_params = {
        "frame_type": frame_type,
        "channels_total": int(channel_count),
        "channel_used": int(chosen_channel),
        "energy_offset_eV": float(energy_offset_raw),
        "energy_step_eV": float(energy_step_raw),
        "angle_offset_deg": float(angle_offset_raw),
        "angle_step_deg": float(angle_step_raw),
        "total_points": int(total_points),
        "width": int(width),
        "height": int(height),
    }
    if subtracted_from is not None:
        ses_params["subtracted_channel"] = int(subtracted_from)

    raw_channels = payload.transpose(2, 0, 1).copy()

    return {
        "spectrum": spectrum.astype(np.float32, copy=False),
        "energy": energy_axis,
        "thetax": angle_axis,
        "ses_params": ses_params,
        "raw_channels": raw_channels,
    }
