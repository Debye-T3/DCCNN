"""Count-rate-normalized evaluation helpers for reviewed A-type pairs."""

from __future__ import annotations

import math

import numpy as np
import xarray as xr

from dccnn_arpes.data.schema import ManifestRecord

from .metrics import evaluate_pair

_PAIR_METRIC_NAMES = (
    "mae",
    "nrmse",
    "psnr",
    "ssim",
    "edc_correlation",
    "mdc_correlation",
    "peak_position_error_eV",
    "peak_position_error_alpha",
    "fwhm_relative_error_eV",
    "fwhm_relative_error_alpha",
    "fwhm_relative_error",
    "integrated_intensity_relative_error",
    "noise_region_reduction",
)


def effective_exposure(record: ManifestRecord) -> float:
    """Return the acquisition-time scale, falling back to sweep count."""
    for field in ("acquisition_time_s", "sweep_count"):
        value = getattr(record, field)
        if value is None or value == "":
            continue
        scale = float(value)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"record {record.record_id} has invalid exposure scale in {field}")
        return scale
    raise ValueError(f"record {record.record_id} has no valid exposure scale")


def orient_pair(
    left: ManifestRecord, right: ManifestRecord
) -> tuple[ManifestRecord, ManifestRecord]:
    """Return the lower-exposure input and higher-exposure reference records."""
    left_scale = effective_exposure(left)
    right_scale = effective_exposure(right)
    if left_scale == right_scale:
        raise ValueError(f"A pair {left.record_id}/{right.record_id} has equal exposure scales")
    return (left, right) if left_scale < right_scale else (right, left)


def count_rate_normalize(data: xr.DataArray, scale: float) -> xr.DataArray:
    """Divide a canonical cut by its exposure and neutralize scale metadata."""
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("count-rate normalization scale must be finite and positive")
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray")
    values = np.asarray(data.values, dtype=np.float64) / scale
    normalized = data.copy(data=values)
    attrs = dict(data.attrs)
    attrs["acquisition_time_s"] = 1.0
    attrs.pop("sweep_count", None)
    normalized.attrs = attrs
    return normalized


def compare_pair(
    input_data: xr.DataArray,
    output_data: xr.DataArray,
    reference_data: xr.DataArray,
) -> dict[str, dict[str, float | str]]:
    """Compare raw input and denoised output against one normalized reference."""
    return {
        "raw": evaluate_pair(input_data, input_data, reference_data),
        "denoised": evaluate_pair(input_data, output_data, reference_data),
    }


def build_pair_row(
    *,
    pair_id: str,
    split: str,
    input_record: ManifestRecord,
    reference_record: ManifestRecord,
    metrics: dict[str, dict[str, float | str]],
) -> dict[str, object]:
    """Flatten one raw/denoised comparison into a CSV-ready row."""
    row: dict[str, object] = {
        "pair_id": pair_id,
        "pair_type": "A",
        "split": split,
        "input_record_id": input_record.record_id,
        "reference_record_id": reference_record.record_id,
        "input_file_id": input_record.file_id,
        "reference_file_id": reference_record.file_id,
        "input_exposure": effective_exposure(input_record),
        "reference_exposure": effective_exposure(reference_record),
    }
    raw = metrics["raw"]
    denoised = metrics["denoised"]
    for name in _PAIR_METRIC_NAMES:
        row[f"raw_{name}"] = raw.get(name)
        row[f"denoised_{name}"] = denoised.get(name)
    for name in ("mae", "nrmse"):
        raw_value = raw.get(name)
        denoised_value = denoised.get(name)
        if raw_value is None or denoised_value is None or float(raw_value) == 0:
            row[f"{name}_improvement"] = None
        else:
            row[f"{name}_improvement"] = (float(raw_value) - float(denoised_value)) / float(raw_value)
    return row
