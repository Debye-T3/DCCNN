"""Scienta DA30L .txt export parser."""

import re
from pathlib import Path
from typing import Tuple

import numpy as np


def parse_axes(lines: list) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Parse dimension sizes and axes from DA30L txt header."""

    def _find(prefix: str) -> str:
        for line in lines:
            if line.startswith(prefix):
                return line.split("=", 1)[1].strip()
        raise ValueError(f"Missing '{prefix}' in txt header.")

    n_energy = int(_find("Dimension 1 size"))
    n_angle = int(_find("Dimension 2 size"))
    energy_axis = np.fromstring(_find("Dimension 1 scale"), sep=" ")
    angle_axis = np.fromstring(_find("Dimension 2 scale"), sep=" ")
    if energy_axis.size != n_energy:
        raise ValueError(f"Energy axis length {energy_axis.size} != {n_energy}")
    if angle_axis.size != n_angle:
        raise ValueError(f"Angle axis length {angle_axis.size} != {n_angle}")
    return energy_axis.astype(np.float32), angle_axis.astype(np.float32), n_energy, n_angle


def parse_data(lines: list, start_idx: int, n_energy: int, n_angle: int) -> np.ndarray:
    """Parse numeric data rows starting at start_idx."""
    data_rows = []
    for line in lines[start_idx:]:
        if not line.strip():
            continue
        nums = np.fromstring(line, sep=" ")
        if nums.size == 0:
            continue
        if nums.size == n_angle + 1:
            nums = nums[1:]
        elif nums.size > n_angle + 1:
            nums = nums[-n_angle:]
        elif nums.size < n_angle:
            nums = np.pad(nums, (0, n_angle - nums.size), mode="constant", constant_values=0)
        data_rows.append(nums)
        if len(data_rows) >= n_energy:
            break
    if len(data_rows) != n_energy:
        missing = n_energy - len(data_rows)
        if missing > 0:
            pad_row = np.zeros((missing, n_angle), dtype=np.float32)
            data_arr = np.vstack([data_rows, pad_row])
        else:
            data_arr = np.stack(data_rows[:n_energy])
    else:
        data_arr = np.stack(data_rows)
    return data_arr.astype(np.float32)


def extract_ses_params(lines: list) -> dict:
    """Extract all SES/DA30L parameters from the header lines into a dict."""
    params = {}
    for line in lines:
        if "=" in line and not line.strip()[0].isdigit():
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and value:
                params[key] = value
    return params


def read_txt(path: Path) -> dict:
    """Read a DA30L .txt file and return spectrum + axes + SES params.

    Returns dict with keys: spectrum (2D float32), energy (1D float32),
    thetax (1D float32), ses_params (dict of str).
    """
    txt_str = path.read_text(encoding="utf-8", errors="ignore")
    lines = txt_str.splitlines()

    numeric_re = re.compile(r"^[0-9eE+\-.\s]+$")
    start_idx = None
    for i, line in enumerate(lines):
        if numeric_re.match(line.strip()) and len(line.split()) > 5:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"Could not find numeric data block in {path}")

    energy_axis, angle_axis, n_energy, n_angle = parse_axes(lines)
    spectrum = parse_data(lines, start_idx + 1, n_energy, n_angle)
    ses_params = extract_ses_params(lines[:start_idx])

    return {
        "spectrum": spectrum,
        "energy": energy_axis,
        "thetax": angle_axis,
        "ses_params": ses_params,
    }
