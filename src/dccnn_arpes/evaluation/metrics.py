"""Coordinate-aware quantitative and physical-fidelity metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from skimage.metrics import structural_similarity

_AXES = ("eV", "alpha")
_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass(frozen=True, slots=True)
class _PeakFit:
    status: str
    reason: str
    center: float
    fwhm: float


def _validate_array(data: xr.DataArray, label: str) -> None:
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"{label} must be an xarray.DataArray")
    if data.dims != _AXES:
        raise ValueError(f"{label} dimensions must be exactly eV and alpha")
    values = np.asarray(data.values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{label} values must be non-empty and finite")
    for axis in _AXES:
        coordinate = np.asarray(data.coords[axis].values, dtype=np.float64)
        if coordinate.ndim != 1 or coordinate.size != data.sizes[axis]:
            raise ValueError(f"{label} coordinate {axis} must be one-dimensional")
        differences = np.diff(coordinate)
        if (
            not np.isfinite(coordinate).all()
            or not differences.size
            or not (np.all(differences > 0) or np.all(differences < 0))
        ):
            raise ValueError(f"{label} coordinate {axis} must be finite and strictly monotonic")


def _coordinates_equal(left: xr.DataArray, right: xr.DataArray) -> bool:
    if left.dims != right.dims or left.sizes != right.sizes:
        return False
    if set(left.coords) != set(right.coords):
        return False
    return all(
        left.coords[name].dims == right.coords[name].dims
        and np.array_equal(left.coords[name].values, right.coords[name].values, equal_nan=True)
        for name in left.coords
    )


def _validate_alignment(
    input_da: xr.DataArray,
    output_da: xr.DataArray,
    reference_da: xr.DataArray,
) -> None:
    for label, data in (
        ("input", input_da),
        ("output", output_da),
        ("reference", reference_da),
    ):
        _validate_array(data, label)
    if not (_coordinates_equal(input_da, output_da) and _coordinates_equal(input_da, reference_da)):
        raise ValueError("input, output and reference coordinates must match exactly")


def _profile(data: xr.DataArray, axis: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(data.values, dtype=np.float64)
    if axis == "eV":
        coordinate = np.asarray(data.coords["eV"].values, dtype=np.float64)
        integration_coordinate = np.asarray(data.coords["alpha"].values, dtype=np.float64)
        profile = np.trapezoid(values, x=integration_coordinate, axis=1)
    elif axis == "alpha":
        coordinate = np.asarray(data.coords["alpha"].values, dtype=np.float64)
        integration_coordinate = np.asarray(data.coords["eV"].values, dtype=np.float64)
        profile = np.trapezoid(values, x=integration_coordinate, axis=0)
    else:  # pragma: no cover - private callers are fixed to the canonical axes
        raise ValueError(f"unsupported profile axis {axis}")
    if integration_coordinate[-1] < integration_coordinate[0]:
        profile = -profile
    return coordinate, np.asarray(profile, dtype=np.float64)


def _gaussian_with_offset(
    coordinate: np.ndarray,
    offset: float,
    amplitude: float,
    center: float,
    sigma: float,
) -> np.ndarray:
    return offset + amplitude * np.exp(-0.5 * ((coordinate - center) / sigma) ** 2)


def _fit_peak(coordinate: np.ndarray, profile: np.ndarray) -> _PeakFit:
    if coordinate.size < 5:
        return _PeakFit("failed", "profile has fewer than five samples", np.nan, np.nan)
    profile_range = float(np.ptp(profile))
    scale = max(float(np.max(np.abs(profile))), 1.0)
    if not np.isfinite(profile).all() or profile_range <= np.finfo(np.float64).eps * scale:
        return _PeakFit("failed", "profile has no resolvable peak", np.nan, np.nan)

    order = np.argsort(coordinate)
    x = coordinate[order]
    y = profile[order]
    baseline = float(np.percentile(y, 10.0))
    weights = np.clip(y - baseline, 0.0, None)
    if float(weights.sum()) <= np.finfo(np.float64).eps * scale:
        return _PeakFit("failed", "profile has no positive peak above baseline", np.nan, np.nan)
    center = float(x[np.argmax(y)])
    weighted_sigma = float(
        np.sqrt(
            np.sum(weights * np.square(x - np.sum(weights * x) / weights.sum())) / weights.sum()
        )
    )
    span = float(x[-1] - x[0])
    sampling_step = float(np.min(np.abs(np.diff(x))))
    initial_sigma = min(max(weighted_sigma, sampling_step), span / 2.0)
    try:
        parameters, _ = curve_fit(
            _gaussian_with_offset,
            x,
            y,
            p0=(baseline, float(np.max(y) - baseline), center, initial_sigma),
            bounds=(
                (-np.inf, 0.0, float(x[0]), sampling_step / 100.0),
                (np.inf, np.inf, float(x[-1]), span * 2.0),
            ),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        return _PeakFit("failed", f"Gaussian fit failed: {error}", np.nan, np.nan)
    fitted_center = float(parameters[2])
    fitted_fwhm = float(abs(parameters[3]) * _FWHM_FACTOR)
    if not np.isfinite(fitted_center) or not np.isfinite(fitted_fwhm) or fitted_fwhm <= 0:
        return _PeakFit("failed", "Gaussian fit returned invalid parameters", np.nan, np.nan)
    return _PeakFit("ok", "", fitted_center, fitted_fwhm)


def _count_rate_scale(data: xr.DataArray) -> float:
    for name in ("acquisition_time_s", "sweep_count"):
        value = data.attrs.get(name)
        if value is None or value == "":
            continue
        scale = float(value)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{name} must be finite and positive when provided")
        return scale
    return 1.0


def _integrated_intensity(data: xr.DataArray) -> float:
    values = np.asarray(data.values, dtype=np.float64) / _count_rate_scale(data)
    alpha = np.asarray(data.coords["alpha"].values, dtype=np.float64)
    energy = np.asarray(data.coords["eV"].values, dtype=np.float64)
    integral = np.trapezoid(np.trapezoid(values, x=alpha, axis=1), x=energy)
    return float(abs(integral))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0:
        return 1.0 if np.array_equal(left, right) else np.nan
    return float(np.dot(left_centered, right_centered) / denominator)


def physical_features(data: xr.DataArray) -> dict[str, float | str]:
    """Return coordinate-valued peak, width and count-rate-integrated features."""
    _validate_array(data, "data")
    fits = {axis: _fit_peak(*_profile(data, axis)) for axis in _AXES}
    failed = [f"{axis}: {fit.reason}" for axis, fit in fits.items() if fit.status != "ok"]
    return {
        "fit_status": "failed" if failed else "ok",
        "fit_failure_reason": "; ".join(failed),
        "peak_eV": fits["eV"].center,
        "peak_alpha": fits["alpha"].center,
        "fwhm_eV": fits["eV"].fwhm,
        "fwhm_alpha": fits["alpha"].fwhm,
        "integrated_intensity": _integrated_intensity(data),
    }


def _relative_error(actual: float, reference: float) -> float:
    if not np.isfinite(actual) or not np.isfinite(reference) or reference == 0:
        return np.nan
    return float(abs(actual - reference) / abs(reference))


def evaluate_pair(
    input_da: xr.DataArray,
    output_da: xr.DataArray,
    reference_da: xr.DataArray,
) -> dict[str, float | str]:
    """Evaluate one output without ever aligning or replacing physical coordinates."""
    _validate_alignment(input_da, output_da, reference_da)
    input_values = np.asarray(input_da.values, dtype=np.float64)
    output_values = np.asarray(output_da.values, dtype=np.float64)
    reference_values = np.asarray(reference_da.values, dtype=np.float64)
    difference = output_values - reference_values
    reference_range = float(np.ptp(reference_values))
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    nrmse = 0.0 if rmse == 0 else np.nan if reference_range == 0 else rmse / reference_range
    psnr = (
        np.inf
        if rmse == 0
        else np.nan
        if reference_range == 0
        else float(20.0 * np.log10(reference_range / rmse))
    )

    minimum_dimension = min(reference_values.shape)
    win_size = min(7, minimum_dimension if minimum_dimension % 2 else minimum_dimension - 1)
    if win_size >= 3:
        ssim_range = reference_range
        if ssim_range == 0:
            ssim = 1.0 if np.array_equal(output_values, reference_values) else np.nan
        else:
            ssim = float(
                structural_similarity(
                    reference_values,
                    output_values,
                    data_range=ssim_range,
                    win_size=win_size,
                )
            )
    else:
        ssim = np.nan

    output_features = physical_features(output_da)
    reference_features = physical_features(reference_da)
    input_features = physical_features(input_da)
    failed_reasons = [
        f"{label}: {features['fit_failure_reason']}"
        for label, features in (
            ("input", input_features),
            ("output", output_features),
            ("reference", reference_features),
        )
        if features["fit_status"] != "ok"
    ]
    fwhm_errors = {
        axis: _relative_error(
            float(output_features[f"fwhm_{axis}"]),
            float(reference_features[f"fwhm_{axis}"]),
        )
        for axis in _AXES
    }
    finite_fwhm_errors = [value for value in fwhm_errors.values() if np.isfinite(value)]

    input_eV, _input_edc = _profile(input_da, "eV")
    output_eV, output_edc = _profile(output_da, "eV")
    reference_eV, reference_edc = _profile(reference_da, "eV")
    input_alpha, _input_mdc = _profile(input_da, "alpha")
    output_alpha, output_mdc = _profile(output_da, "alpha")
    reference_alpha, reference_mdc = _profile(reference_da, "alpha")
    assert np.array_equal(input_eV, output_eV) and np.array_equal(output_eV, reference_eV)
    assert np.array_equal(input_alpha, output_alpha) and np.array_equal(
        output_alpha, reference_alpha
    )

    background_mask = reference_values <= np.quantile(reference_values, 0.20)
    input_noise = float(np.std((input_values - reference_values)[background_mask]))
    output_noise = float(np.std((output_values - reference_values)[background_mask]))
    if input_noise == 0:
        noise_reduction = 0.0 if output_noise == 0 else np.nan
        noise_status = "no_input_noise" if output_noise == 0 else "undefined"
    else:
        noise_reduction = float((input_noise - output_noise) / input_noise)
        noise_status = "ok"

    peak_error_eV = float(
        abs(float(output_features["peak_eV"]) - float(reference_features["peak_eV"]))
    )
    peak_error_alpha = float(
        abs(float(output_features["peak_alpha"]) - float(reference_features["peak_alpha"]))
    )
    result: dict[str, float | str] = {
        "mae": float(np.mean(np.abs(difference))),
        "nrmse": float(nrmse),
        "psnr": float(psnr),
        "ssim": float(ssim),
        "edc_correlation": _correlation(output_edc, reference_edc),
        "mdc_correlation": _correlation(output_mdc, reference_mdc),
        "peak_position_error_eV": peak_error_eV,
        "peak_position_error_alpha": peak_error_alpha,
        "peak_position_error": float(max(peak_error_eV, peak_error_alpha)),
        "fwhm_relative_error_eV": fwhm_errors["eV"],
        "fwhm_relative_error_alpha": fwhm_errors["alpha"],
        "fwhm_relative_error": (
            float(max(finite_fwhm_errors)) if len(finite_fwhm_errors) == 2 else np.nan
        ),
        "integrated_intensity_relative_error": _relative_error(
            float(output_features["integrated_intensity"]),
            float(reference_features["integrated_intensity"]),
        ),
        "noise_region_reduction": noise_reduction,
        "noise_only_region_reduction": noise_reduction,
        "noise_region_status": noise_status,
        "fit_status": "failed" if failed_reasons else "ok",
        "fit_failure_reason": "; ".join(failed_reasons),
    }
    for label, features in (
        ("input", input_features),
        ("output", output_features),
        ("reference", reference_features),
    ):
        for feature in (
            "peak_eV",
            "peak_alpha",
            "fwhm_eV",
            "fwhm_alpha",
            "integrated_intensity",
        ):
            result[f"{label}_{feature}"] = float(features[feature])
    return result
