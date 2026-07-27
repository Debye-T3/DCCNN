"""Physical-fidelity report artifacts and scientific acceptance gating."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from scipy.signal import find_peaks

from .baselines import gaussian_baseline, median_baseline
from .metrics import evaluate_pair, physical_features

_METHODS = (
    "raw_input",
    "gaussian",
    "median",
    "LegacyCCNN",
    "ResidualDenoiser2D",
)
_METRIC_NAMES = (
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
_FEATURE_NAMES = (
    "peak_eV",
    "peak_alpha",
    "fwhm_eV",
    "fwhm_alpha",
    "integrated_intensity",
)
_RULE_TEXT = {
    "1_paired_test_improvement": (
        "At least 80% of paired test cuts have lower NRMSE than their raw input."
    ),
    "2_legacy_nrmse_improvement": ("Mean paired-test NRMSE is at least 10% lower than LegacyCCNN."),
    "3_peak_position_fidelity": (
        "Peak-position error is no larger than one eV or alpha sampling step."
    ),
    "4_fwhm_fidelity": "FWHM relative error is at most 10%.",
    "5_integrated_intensity_fidelity": (
        "Count-rate-normalized integrated-intensity relative error is at most 5%."
    ),
    "6_high_quality_identity": (
        "High-quality identity cuts contain no new peak above 5% prominence and "
        "no new stripe above five background standard deviations."
    ),
    "7_temperature_trends": (
        "Temperature trend direction is not reversed and adjacent output jumps stay "
        "within both the three-times-input-jump and measurement-uncertainty limits."
    ),
    "8_manifest_reconciliation": (
        "Evaluated, failed-fit and manually flagged samples reconcile exactly to the "
        "locked test manifest row count."
    ),
}


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One locked-manifest row and all available evaluation arrays."""

    record_id: str
    input_da: xr.DataArray | None
    output_da: xr.DataArray | None
    reference_da: xr.DataArray | None
    legacy_da: xr.DataArray | None = None
    pair_type: str = ""
    temperature_K: float | None = None
    temperature_group: str = ""
    measurement_uncertainty: float | None = None
    high_quality_identity: bool = False
    manually_flagged: bool = False
    manual_flag_reason: str = ""
    scientific_eligible: bool = False
    eligibility_reason: str = ""

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("record_id must be non-empty")
        if self.temperature_K is not None and not np.isfinite(self.temperature_K):
            raise ValueError("temperature_K must be finite when provided")
        if self.measurement_uncertainty is not None and (
            not np.isfinite(self.measurement_uncertainty) or self.measurement_uncertainty < 0
        ):
            raise ValueError("measurement_uncertainty must be finite and non-negative")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(value: object, path: Path) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _intent_flag(data: xr.DataArray, name: str) -> bool | None:
    value = data.attrs.get(name)
    if type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _scientific_gate(cases: list[EvaluationCase]) -> dict[str, object]:
    reasons: list[str] = []
    if not cases:
        reasons.append("the locked test manifest contains no rows")
    ineligible = [case.record_id for case in cases if not case.scientific_eligible]
    if ineligible:
        reasons.append(f"{len(ineligible)} manifest rows are not eligible for scientific testing")
    missing = [
        case.record_id
        for case in cases
        if any(
            array is None
            for array in (case.input_da, case.output_da, case.reference_da, case.legacy_da)
        )
    ]
    if missing:
        reasons.append(f"{len(missing)} rows lack a required input/output/reference/model artifact")
    non_scientific = [
        case.record_id
        for case in cases
        if case.output_da is None
        or _intent_flag(case.output_da, "scientific_use") is not True
        or _intent_flag(case.output_da, "smoke_test") is not False
    ]
    if non_scientific:
        reasons.append(f"{len(non_scientific)} residual outputs have non-scientific intent")
    return {
        "pass": not reasons,
        "reason": "; ".join(reasons)
        if reasons
        else "all output intent and population checks passed",
        "ineligible_record_ids": ineligible,
        "missing_artifact_record_ids": missing,
        "non_scientific_record_ids": non_scientific,
    }


def _sampling_step(data: xr.DataArray | None, axis: str) -> float:
    if data is None:
        return np.nan
    values = np.asarray(data.coords[axis].values, dtype=np.float64)
    return float(np.median(np.abs(np.diff(values))))


def _empty_metrics(reason: str) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {name: np.nan for name in _METRIC_NAMES}
    metrics.update(
        {
            "fit_status": "not_evaluated",
            "fit_failure_reason": reason,
            "noise_region_status": "not_evaluated",
        }
    )
    for label in ("input", "output", "reference"):
        for feature in _FEATURE_NAMES:
            metrics[f"{label}_{feature}"] = np.nan
    return metrics


def _method_arrays(case: EvaluationCase) -> dict[str, xr.DataArray | None]:
    return {
        "raw_input": case.input_da,
        "gaussian": (gaussian_baseline(case.input_da) if case.input_da is not None else None),
        "median": median_baseline(case.input_da) if case.input_da is not None else None,
        "LegacyCCNN": case.legacy_da,
        "ResidualDenoiser2D": case.output_da,
    }


def _evaluate_rows(
    cases: list[EvaluationCase],
) -> tuple[list[dict[str, object]], dict[str, dict[str, xr.DataArray | None]]]:
    rows: list[dict[str, object]] = []
    arrays_by_case: dict[str, dict[str, xr.DataArray | None]] = {}
    for case in cases:
        method_arrays = _method_arrays(case)
        arrays_by_case[case.record_id] = method_arrays
        for method in _METHODS:
            candidate = method_arrays[method]
            if case.input_da is None or candidate is None or case.reference_da is None:
                reason = "input, candidate and reference artifacts are required"
                metrics = _empty_metrics(reason)
                evaluation_status = "not_evaluated"
            else:
                try:
                    metrics = evaluate_pair(case.input_da, candidate, case.reference_da)
                except (TypeError, ValueError) as error:
                    reason = str(error)
                    metrics = _empty_metrics(reason)
                    metrics["fit_status"] = "failed"
                    metrics["fit_failure_reason"] = reason
                    evaluation_status = "failed"
                else:
                    evaluation_status = (
                        "failed_fit" if metrics["fit_status"] == "failed" else "evaluated"
                    )
            rows.append(
                {
                    "record_id": case.record_id,
                    "method": method,
                    "evaluation_status": evaluation_status,
                    "pair_type": case.pair_type,
                    "temperature_K": case.temperature_K,
                    "temperature_group": case.temperature_group,
                    "high_quality_identity": case.high_quality_identity,
                    "scientific_eligible": case.scientific_eligible,
                    "eV_sampling_step": _sampling_step(case.input_da, "eV"),
                    "alpha_sampling_step": _sampling_step(case.input_da, "alpha"),
                    **metrics,
                }
            )
    return rows, arrays_by_case


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    preferred = [
        "record_id",
        "method",
        "evaluation_status",
        "pair_type",
        "temperature_K",
        "temperature_group",
        "high_quality_identity",
        "scientific_eligible",
        "fit_status",
        "fit_failure_reason",
    ]
    keys = {key for row in rows for key in row}
    fieldnames = preferred + sorted(keys.difference(preferred))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, object]], manifest_row_count: int) -> dict[str, object]:
    methods: dict[str, object] = {}
    for method in _METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        metrics: dict[str, object] = {}
        for metric in _METRIC_NAMES:
            values = np.asarray([row[metric] for row in method_rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            metrics[metric] = {
                "count": int(finite.size),
                "mean": float(np.mean(finite)) if finite.size else None,
                "median": float(np.median(finite)) if finite.size else None,
                "minimum": float(np.min(finite)) if finite.size else None,
                "maximum": float(np.max(finite)) if finite.size else None,
            }
        methods[method] = {
            "row_count": len(method_rows),
            "status_counts": dict(
                sorted(Counter(str(row["evaluation_status"]) for row in method_rows).items())
            ),
            "metrics": metrics,
        }
    return {"manifest_row_count": manifest_row_count, "methods": methods}


def _worst_cases(rows: list[dict[str, object]]) -> dict[str, object]:
    residual = [
        row
        for row in rows
        if row["method"] == "ResidualDenoiser2D" and np.isfinite(float(row["nrmse"]))
    ]
    ranked = sorted(residual, key=lambda row: float(row["nrmse"]), reverse=True)
    return {
        "metric": "nrmse",
        "cases": [
            {
                "record_id": row["record_id"],
                "nrmse": row["nrmse"],
                "fit_status": row["fit_status"],
            }
            for row in ranked[:10]
        ],
    }


def _profile(data: xr.DataArray, axis: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(data.values, dtype=np.float64)
    if axis == "eV":
        coordinate = np.asarray(data.coords["eV"].values, dtype=np.float64)
        profile = np.trapezoid(
            values,
            x=np.asarray(data.coords["alpha"].values, dtype=np.float64),
            axis=1,
        )
    else:
        coordinate = np.asarray(data.coords["alpha"].values, dtype=np.float64)
        profile = np.trapezoid(
            values,
            x=np.asarray(data.coords["eV"].values, dtype=np.float64),
            axis=0,
        )
    return coordinate, np.abs(profile)


def _safe_stem(record_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", record_id).strip("._")
    return stem or "record"


def _aligned_for_preview(data: xr.DataArray, reference: xr.DataArray) -> bool:
    return (
        data.dims == reference.dims
        and data.shape == reference.shape
        and all(
            np.array_equal(data.coords[axis].values, reference.coords[axis].values)
            for axis in ("eV", "alpha")
        )
    )


def _write_preview(
    case: EvaluationCase,
    arrays: dict[str, xr.DataArray | None],
    destination: Path,
) -> None:
    reference = case.reference_da
    if reference is None:
        figure = Figure(figsize=(7.0, 3.0), constrained_layout=True)
        axis = figure.subplots()
        axis.text(0.5, 0.5, "not evaluated: missing reference", ha="center", va="center")
        axis.set_axis_off()
        figure.savefig(destination, dpi=120)
        return
    compatible = [
        data
        for data in arrays.values()
        if data is not None and _aligned_for_preview(data, reference)
    ]
    reference_values = np.asarray(reference.values, dtype=np.float64)
    all_values = np.concatenate(
        [np.asarray(data.values, dtype=np.float64).ravel() for data in compatible]
        + [reference_values.ravel()],
    )
    vmin, vmax = float(np.quantile(all_values, 0.01)), float(np.quantile(all_values, 0.99))
    difference_limit = max(
        (
            float(np.quantile(np.abs(np.asarray(data.values) - reference_values), 0.99))
            for data in compatible
        ),
        default=np.finfo(np.float64).eps,
    )
    difference_limit = max(difference_limit, np.finfo(np.float64).eps)
    figure = Figure(
        figsize=(3.0 * len(_METHODS), 5.5),
        constrained_layout=True,
    )
    axes = figure.subplots(2, len(_METHODS), squeeze=False, sharex=True, sharey=True)
    extent = [
        float(reference.alpha.values[0]),
        float(reference.alpha.values[-1]),
        float(reference.eV.values[-1]),
        float(reference.eV.values[0]),
    ]
    image = difference = None
    for column, method in enumerate(_METHODS):
        data = arrays[method]
        axes[0, column].set_title(method)
        axes[1, column].set_title(f"{method} - reference")
        axes[1, column].set_xlabel("alpha")
        if data is None or not _aligned_for_preview(data, reference):
            reason = "missing artifact" if data is None else "coordinate/shape mismatch"
            for row in range(2):
                axes[row, column].text(
                    0.5,
                    0.5,
                    f"not plotted:\n{reason}",
                    ha="center",
                    va="center",
                    transform=axes[row, column].transAxes,
                )
                axes[row, column].set_facecolor("0.95")
            continue
        image = axes[0, column].imshow(
            data.values,
            aspect="auto",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        difference = axes[1, column].imshow(
            np.asarray(data.values) - reference_values,
            aspect="auto",
            extent=extent,
            vmin=-difference_limit,
            vmax=difference_limit,
            cmap="coolwarm",
        )
    axes[0, 0].set_ylabel("eV")
    axes[1, 0].set_ylabel("eV")
    if image is not None and difference is not None:
        figure.colorbar(image, ax=axes[0, :].tolist(), label="intensity")
        figure.colorbar(difference, ax=axes[1, :].tolist(), label="difference")
    figure.savefig(destination, dpi=120)


def _write_profile_figure(
    case: EvaluationCase,
    arrays: dict[str, xr.DataArray | None],
    axis_name: str,
    destination: Path,
) -> None:
    figure = Figure(figsize=(7.0, 4.0), constrained_layout=True)
    axis = figure.subplots()
    plotted = False
    if case.reference_da is not None:
        coordinate, profile = _profile(case.reference_da, axis_name)
        axis.plot(coordinate, profile, color="black", linewidth=2.0, label="reference")
        plotted = True
    for method in _METHODS:
        data = arrays[method]
        if data is None:
            continue
        coordinate, profile = _profile(data, axis_name)
        axis.plot(coordinate, profile, linewidth=1.0, label=method)
        plotted = True
    if plotted:
        axis.set_xlabel(axis_name)
        axis.set_ylabel("integrated intensity")
        axis.legend(fontsize="small")
    else:
        axis.text(0.5, 0.5, "not evaluated: missing arrays", ha="center", va="center")
        axis.set_axis_off()
    figure.savefig(destination, dpi=120)


def _write_figures(
    cases: list[EvaluationCase],
    arrays_by_case: dict[str, dict[str, xr.DataArray | None]],
    output_dir: Path,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    used_stems: set[str] = set()
    for case in cases:
        stem = _safe_stem(case.record_id)
        if stem in used_stems:
            raise ValueError("record IDs must have unique filesystem-safe names")
        used_stems.add(stem)
        arrays = arrays_by_case[case.record_id]
        _write_preview(case, arrays, figure_dir / f"{stem}_preview.png")
        _write_profile_figure(case, arrays, "eV", figure_dir / f"{stem}_edc.png")
        _write_profile_figure(case, arrays, "alpha", figure_dir / f"{stem}_mdc.png")


def _boundary_is_local_maximum(profile: np.ndarray) -> bool:
    boundary = profile[0]
    for value in profile[1:]:
        if np.isclose(boundary, value):
            continue
        return bool(boundary > value)
    return False


def _peak_candidates(profile: np.ndarray, prominence: float) -> list[tuple[int, float]]:
    peaks, properties = find_peaks(profile, prominence=prominence)
    candidates = [
        (int(peak), float(value))
        for peak, value in zip(peaks, properties["prominences"], strict=True)
    ]
    left_prominence = float(profile[0] - np.min(profile[1:]))
    if _boundary_is_local_maximum(profile) and left_prominence > prominence:
        candidates.append((0, left_prominence))
    right_prominence = float(profile[-1] - np.min(profile[:-1]))
    if _boundary_is_local_maximum(profile[::-1]) and right_prominence > prominence:
        candidates.append((profile.size - 1, right_prominence))
    return candidates


def _new_peak_reasons(case: EvaluationCase) -> list[str]:
    if (
        case.output_da is None
        or case.reference_da is None
        or not _aligned_for_preview(case.output_da, case.reference_da)
    ):
        return []
    reasons: list[str] = []
    for axis in ("eV", "alpha"):
        coordinate, reference_profile = _profile(case.reference_da, axis)
        _, output_profile = _profile(case.output_da, axis)
        reference_prominence = 0.05 * float(np.max(reference_profile))
        reference_peaks = _peak_candidates(reference_profile, reference_prominence)
        for peak, prominence in _peak_candidates(output_profile, reference_prominence):
            if reference_peaks and np.min(
                np.abs(
                    coordinate[[reference_peak for reference_peak, _ in reference_peaks]]
                    - coordinate[peak]
                )
            ) <= (1.01 * _sampling_step(case.reference_da, axis)):
                continue
            reasons.append(
                f"new {axis} peak prominence {float(prominence):.6g} exceeds "
                "5% of the reference-profile maximum"
            )
    return reasons


def _stripe_reasons(case: EvaluationCase) -> list[str]:
    if (
        case.output_da is None
        or case.reference_da is None
        or not _aligned_for_preview(case.output_da, case.reference_da)
    ):
        return []
    reference = np.asarray(case.reference_da.values, dtype=np.float64)
    difference = np.asarray(case.output_da.values, dtype=np.float64) - reference
    background = reference[reference <= np.quantile(reference, 0.20)]
    background_std = float(np.std(background))
    threshold = 5.0 * max(background_std, np.finfo(np.float64).eps)
    row_signal = np.abs(np.mean(difference, axis=1) - np.median(difference))
    column_signal = np.abs(np.mean(difference, axis=0) - np.median(difference))
    sparse_amplitudes: list[float] = []
    for signal in (row_signal, column_signal):
        maximum = float(np.max(signal))
        sparse_limit = max(1, int(np.ceil(0.05 * signal.size)))
        if maximum > 0 and np.count_nonzero(signal >= 0.5 * maximum) <= sparse_limit:
            sparse_amplitudes.append(maximum)
    amplitude = max(sparse_amplitudes, default=0.0)
    if amplitude > threshold:
        return [
            (
                f"new stripe amplitude {amplitude:.6g} exceeds five background standard "
                f"deviations ({threshold:.6g})"
            )
        ]
    return []


def _high_quality_flags(
    cases: list[EvaluationCase],
    residual_rows: dict[str, dict[str, object]],
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    for case in cases:
        reasons: list[str] = []
        if case.manually_flagged:
            reasons.append(case.manual_flag_reason or "manually flagged in locked manifest")
        if case.high_quality_identity:
            reasons.extend(_new_peak_reasons(case))
            reasons.extend(_stripe_reasons(case))
            row = residual_rows[case.record_id]
            if (
                np.isfinite(float(row["peak_position_error_eV"]))
                and float(row["peak_position_error_eV"]) > float(row["eV_sampling_step"])
            ) or (
                np.isfinite(float(row["peak_position_error_alpha"]))
                and float(row["peak_position_error_alpha"]) > float(row["alpha_sampling_step"])
            ):
                reasons.append("peak-position error exceeds one sampling step")
            if (
                np.isfinite(float(row["fwhm_relative_error"]))
                and float(row["fwhm_relative_error"]) > 0.10
            ):
                reasons.append("FWHM relative error exceeds 10%")
            if (
                np.isfinite(float(row["integrated_intensity_relative_error"]))
                and float(row["integrated_intensity_relative_error"]) > 0.05
            ):
                reasons.append("integrated-intensity relative error exceeds 5%")
        if reasons:
            flags.append(
                {
                    "record_id": case.record_id,
                    "status": "review_required",
                    "reasons": "; ".join(dict.fromkeys(reasons)),
                }
            )
    return flags


def _write_flags(flags: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("record_id", "status", "reasons"))
        writer.writeheader()
        writer.writerows(flags)


def _feature_values(data: xr.DataArray | None) -> dict[str, float | str]:
    if data is None:
        return {
            "fit_status": "not_evaluated",
            "fit_failure_reason": "missing artifact",
            **{name: np.nan for name in _FEATURE_NAMES},
        }
    try:
        return physical_features(data)
    except (TypeError, ValueError) as error:
        return {
            "fit_status": "failed",
            "fit_failure_reason": str(error),
            **{name: np.nan for name in _FEATURE_NAMES},
        }


def _temperature_trends(cases: list[EvaluationCase]) -> dict[str, object]:
    groups: dict[str, list[EvaluationCase]] = {}
    for case in cases:
        if case.temperature_group:
            groups.setdefault(case.temperature_group, []).append(case)
    result: dict[str, object] = {}
    for group_name, unordered in sorted(groups.items()):
        ordered = sorted(
            unordered,
            key=lambda case: (
                case.temperature_K is None,
                float(case.temperature_K) if case.temperature_K is not None else np.inf,
                case.record_id,
            ),
        )
        missing_temperature_ids = sorted(
            case.record_id for case in ordered if case.temperature_K is None
        )
        missing_uncertainty_ids = sorted(
            case.record_id for case in ordered if case.measurement_uncertainty is None
        )
        samples: list[dict[str, object]] = []
        features_by_source = {
            "input": {name: [] for name in _FEATURE_NAMES},
            "output": {name: [] for name in _FEATURE_NAMES},
        }
        for case in ordered:
            input_features = _feature_values(case.input_da)
            output_features = _feature_values(case.output_da)
            reference_features = _feature_values(case.reference_da)
            sample: dict[str, object] = {
                "record_id": case.record_id,
                "reference_record_id": case.record_id,
                "temperature_K": case.temperature_K,
                "measurement_uncertainty": case.measurement_uncertainty,
                "input_fit_status": input_features["fit_status"],
                "output_fit_status": output_features["fit_status"],
                "reference_fit_status": reference_features["fit_status"],
            }
            for name in _FEATURE_NAMES:
                sample[f"input_{name}"] = input_features[name]
                sample[f"output_{name}"] = output_features[name]
                sample[f"reference_{name}"] = reference_features[name]
                features_by_source["input"][name].append(float(input_features[name]))
                features_by_source["output"][name].append(float(output_features[name]))
            samples.append(sample)

        temperatures = np.asarray(
            [
                float(case.temperature_K) if case.temperature_K is not None else np.nan
                for case in ordered
            ]
        )
        feature_evidence: dict[str, object] = {}
        for name in _FEATURE_NAMES:
            input_values = np.asarray(features_by_source["input"][name], dtype=np.float64)
            output_values = np.asarray(features_by_source["output"][name], dtype=np.float64)
            finite = (
                np.isfinite(temperatures) & np.isfinite(input_values) & np.isfinite(output_values)
            )
            valid_indices = np.flatnonzero(finite)
            evaluated_pair_count = int(np.count_nonzero(finite))
            unique_temperature_count = int(np.unique(temperatures[finite]).size)
            reversed_direction = False
            input_slope = output_slope = np.nan
            jump_violations: list[dict[str, object]] = []
            if evaluated_pair_count >= 2 and unique_temperature_count >= 2:
                valid_temperatures = temperatures[finite]
                valid_input = input_values[finite]
                valid_output = output_values[finite]
                input_slope = float(np.polyfit(valid_temperatures, valid_input, 1)[0])
                output_slope = float(np.polyfit(valid_temperatures, valid_output, 1)[0])
                scale = max(float(np.max(np.abs(valid_input))), 1.0)
                slope_tolerance = np.finfo(np.float64).eps * scale
                reversed_direction = (
                    abs(input_slope) > slope_tolerance
                    and abs(output_slope) > slope_tolerance
                    and np.sign(input_slope) != np.sign(output_slope)
                )
                input_jumps = np.abs(np.diff(valid_input))
                finite_input_jumps = input_jumps[np.isfinite(input_jumps)]
                median_input_jump = (
                    float(np.median(finite_input_jumps)) if finite_input_jumps.size else np.nan
                )
                for index, output_jump in enumerate(np.abs(np.diff(valid_output))):
                    left = ordered[int(valid_indices[index])]
                    right = ordered[int(valid_indices[index + 1])]
                    if (
                        left.measurement_uncertainty is None
                        or right.measurement_uncertainty is None
                    ):
                        continue
                    uncertainty = max(
                        left.measurement_uncertainty,
                        right.measurement_uncertainty,
                    )
                    if (
                        np.isfinite(output_jump)
                        and np.isfinite(median_input_jump)
                        and output_jump > 3.0 * median_input_jump
                        and output_jump > uncertainty
                    ):
                        jump_violations.append(
                            {
                                "left_record_id": left.record_id,
                                "right_record_id": right.record_id,
                                "output_jump": float(output_jump),
                                "three_times_input_median_jump": 3.0 * median_input_jump,
                                "measurement_uncertainty": uncertainty,
                            }
                        )
            feature_evidence[name] = {
                "required_sample_count": len(ordered),
                "evaluated_pair_count": evaluated_pair_count,
                "unique_temperature_count": unique_temperature_count,
                "input_slope": input_slope,
                "output_slope": output_slope,
                "direction_reversed": reversed_direction,
                "jump_violations": jump_violations,
            }
        result[group_name] = {
            "samples": samples,
            "features": feature_evidence,
            "missing_temperature_record_ids": missing_temperature_ids,
            "missing_measurement_uncertainty_record_ids": missing_uncertainty_ids,
        }
    return {"groups": result}


def _rule(
    status: str, passed: bool | None, evidence: dict[str, object], text: str
) -> dict[str, object]:
    return {"rule": text, "status": status, "pass": passed, "evidence": evidence}


def _pass_fail(passed: bool, evidence: dict[str, object], text: str) -> dict[str, object]:
    return _rule("pass" if passed else "fail", passed, evidence, text)


def _not_evaluated(evidence: dict[str, object], text: str) -> dict[str, object]:
    return _rule("not_evaluated", None, evidence, text)


def _acceptance(
    cases: list[EvaluationCase],
    rows: list[dict[str, object]],
    flags: list[dict[str, str]],
    trends: dict[str, object],
    manifest_row_count: int,
) -> dict[str, object]:
    residual = {str(row["record_id"]): row for row in rows if row["method"] == "ResidualDenoiser2D"}
    raw = {str(row["record_id"]): row for row in rows if row["method"] == "raw_input"}
    legacy = {str(row["record_id"]): row for row in rows if row["method"] == "LegacyCCNN"}
    paired_population = [
        case.record_id
        for case in cases
        if case.pair_type.strip().casefold() in {"a", "b", "pair", "paired"}
    ]
    paired_ids = [
        record_id
        for record_id in paired_population
        if np.isfinite(float(raw[record_id]["nrmse"]))
        and np.isfinite(float(residual[record_id]["nrmse"]))
    ]
    missing_paired = sorted(set(paired_population).difference(paired_ids))
    rules: dict[str, dict[str, object]] = {}
    if paired_population:
        improved = [
            record_id
            for record_id in paired_ids
            if float(residual[record_id]["nrmse"]) < float(raw[record_id]["nrmse"])
        ]
        rate = len(improved) / len(paired_population)
        evidence = {
            "paired_test_count": len(paired_population),
            "evaluated_count": len(paired_ids),
            "missing_record_ids": missing_paired,
            "improved_count": len(improved),
            "improved_fraction": rate,
            "threshold": 0.80,
        }
        if missing_paired:
            rules["1_paired_test_improvement"] = _not_evaluated(
                evidence,
                _RULE_TEXT["1_paired_test_improvement"],
            )
        else:
            rules["1_paired_test_improvement"] = _pass_fail(
                rate >= 0.80,
                evidence,
                _RULE_TEXT["1_paired_test_improvement"],
            )
    else:
        rules["1_paired_test_improvement"] = _not_evaluated(
            {
                "paired_test_count": 0,
                "evaluated_count": 0,
                "missing_record_ids": [],
                "threshold": 0.80,
            },
            _RULE_TEXT["1_paired_test_improvement"],
        )

    legacy_ids = [
        record_id
        for record_id in paired_population
        if np.isfinite(float(residual[record_id]["nrmse"]))
        and np.isfinite(float(legacy[record_id]["nrmse"]))
    ]
    missing_legacy = sorted(set(paired_population).difference(legacy_ids))
    if paired_population:
        residual_mean = (
            float(np.mean([float(residual[record_id]["nrmse"]) for record_id in legacy_ids]))
            if legacy_ids
            else np.nan
        )
        legacy_mean = (
            float(np.mean([float(legacy[record_id]["nrmse"]) for record_id in legacy_ids]))
            if legacy_ids
            else np.nan
        )
        reduction = (
            float((legacy_mean - residual_mean) / legacy_mean)
            if np.isfinite(legacy_mean) and legacy_mean > 0
            else (
                0.0
                if np.isfinite(residual_mean) and residual_mean == 0 and legacy_mean == 0
                else np.nan
            )
        )
        evidence = {
            "paired_test_count": len(paired_population),
            "evaluated_count": len(legacy_ids),
            "missing_record_ids": missing_legacy,
            "residual_mean_nrmse": residual_mean,
            "legacy_mean_nrmse": legacy_mean,
            "relative_reduction": reduction,
            "threshold": 0.10,
        }
        if missing_legacy:
            rules["2_legacy_nrmse_improvement"] = _not_evaluated(
                evidence,
                _RULE_TEXT["2_legacy_nrmse_improvement"],
            )
        else:
            rules["2_legacy_nrmse_improvement"] = _pass_fail(
                reduction >= 0.10,
                evidence,
                _RULE_TEXT["2_legacy_nrmse_improvement"],
            )
    else:
        rules["2_legacy_nrmse_improvement"] = _not_evaluated(
            {
                "paired_test_count": 0,
                "evaluated_count": 0,
                "missing_record_ids": [],
                "threshold": 0.10,
            },
            _RULE_TEXT["2_legacy_nrmse_improvement"],
        )

    required_ids = [case.record_id for case in cases]
    valid_peak_rows = [
        residual[record_id]
        for record_id in required_ids
        if all(
            np.isfinite(float(residual[record_id][name]))
            for name in (
                "peak_position_error_eV",
                "peak_position_error_alpha",
                "eV_sampling_step",
                "alpha_sampling_step",
            )
        )
    ]
    missing_peak = sorted(
        set(required_ids).difference(str(row["record_id"]) for row in valid_peak_rows)
    )
    if required_ids:
        violations = [
            str(row["record_id"])
            for row in valid_peak_rows
            if float(row["peak_position_error_eV"]) > float(row["eV_sampling_step"])
            or float(row["peak_position_error_alpha"]) > float(row["alpha_sampling_step"])
        ]
        evidence = {
            "required_count": len(required_ids),
            "evaluated_count": len(valid_peak_rows),
            "missing_record_ids": missing_peak,
            "violation_record_ids": violations,
            "maximum_eV_error": (
                max(float(row["peak_position_error_eV"]) for row in valid_peak_rows)
                if valid_peak_rows
                else np.nan
            ),
            "maximum_alpha_error": (
                max(float(row["peak_position_error_alpha"]) for row in valid_peak_rows)
                if valid_peak_rows
                else np.nan
            ),
            "threshold": "one per-sample eV or alpha sampling step",
        }
        if missing_peak:
            rules["3_peak_position_fidelity"] = _not_evaluated(
                evidence,
                _RULE_TEXT["3_peak_position_fidelity"],
            )
        else:
            rules["3_peak_position_fidelity"] = _pass_fail(
                not violations,
                evidence,
                _RULE_TEXT["3_peak_position_fidelity"],
            )
    else:
        rules["3_peak_position_fidelity"] = _not_evaluated(
            {"required_count": 0, "evaluated_count": 0, "missing_record_ids": []},
            _RULE_TEXT["3_peak_position_fidelity"],
        )

    valid_width_rows = [
        residual[record_id]
        for record_id in required_ids
        if np.isfinite(float(residual[record_id]["fwhm_relative_error"]))
    ]
    missing_width = sorted(
        set(required_ids).difference(str(row["record_id"]) for row in valid_width_rows)
    )
    if required_ids:
        maximum = (
            max(float(row["fwhm_relative_error"]) for row in valid_width_rows)
            if valid_width_rows
            else np.nan
        )
        evidence = {
            "required_count": len(required_ids),
            "evaluated_count": len(valid_width_rows),
            "missing_record_ids": missing_width,
            "maximum_relative_error": maximum,
            "threshold": 0.10,
        }
        if missing_width:
            rules["4_fwhm_fidelity"] = _not_evaluated(
                evidence,
                _RULE_TEXT["4_fwhm_fidelity"],
            )
        else:
            rules["4_fwhm_fidelity"] = _pass_fail(
                maximum <= 0.10,
                evidence,
                _RULE_TEXT["4_fwhm_fidelity"],
            )
    else:
        rules["4_fwhm_fidelity"] = _not_evaluated(
            {
                "required_count": 0,
                "evaluated_count": 0,
                "missing_record_ids": [],
                "threshold": 0.10,
            },
            _RULE_TEXT["4_fwhm_fidelity"],
        )

    valid_integral_rows = [
        residual[record_id]
        for record_id in required_ids
        if np.isfinite(float(residual[record_id]["integrated_intensity_relative_error"]))
    ]
    missing_integral = sorted(
        set(required_ids).difference(str(row["record_id"]) for row in valid_integral_rows)
    )
    if required_ids:
        maximum = (
            max(float(row["integrated_intensity_relative_error"]) for row in valid_integral_rows)
            if valid_integral_rows
            else np.nan
        )
        evidence = {
            "required_count": len(required_ids),
            "evaluated_count": len(valid_integral_rows),
            "missing_record_ids": missing_integral,
            "maximum_relative_error": maximum,
            "threshold": 0.05,
            "normalization": "acquisition_time_s, then sweep_count, otherwise unity",
        }
        if missing_integral:
            rules["5_integrated_intensity_fidelity"] = _not_evaluated(
                evidence,
                _RULE_TEXT["5_integrated_intensity_fidelity"],
            )
        else:
            rules["5_integrated_intensity_fidelity"] = _pass_fail(
                maximum <= 0.05,
                evidence,
                _RULE_TEXT["5_integrated_intensity_fidelity"],
            )
    else:
        rules["5_integrated_intensity_fidelity"] = _not_evaluated(
            {
                "required_count": 0,
                "evaluated_count": 0,
                "missing_record_ids": [],
                "threshold": 0.05,
            },
            _RULE_TEXT["5_integrated_intensity_fidelity"],
        )

    high_quality_ids = [case.record_id for case in cases if case.high_quality_identity]
    flagged_high_quality = [flag for flag in flags if flag["record_id"] in set(high_quality_ids)]
    complete_high_quality = [
        record_id
        for record_id in high_quality_ids
        if all(
            np.isfinite(float(residual[record_id][name]))
            for name in (
                "peak_position_error_eV",
                "peak_position_error_alpha",
                "fwhm_relative_error",
                "integrated_intensity_relative_error",
            )
        )
    ]
    missing_high_quality = sorted(set(high_quality_ids).difference(complete_high_quality))
    if high_quality_ids:
        evidence = {
            "required_count": len(high_quality_ids),
            "evaluated_count": len(complete_high_quality),
            "missing_record_ids": missing_high_quality,
            "flagged_cases": flagged_high_quality,
            "peak_prominence_threshold": 0.05,
            "stripe_standard_deviation_threshold": 5.0,
        }
        if missing_high_quality:
            rules["6_high_quality_identity"] = _not_evaluated(
                evidence,
                _RULE_TEXT["6_high_quality_identity"],
            )
        elif flagged_high_quality:
            rules["6_high_quality_identity"] = _rule(
                "review_required",
                False,
                evidence,
                _RULE_TEXT["6_high_quality_identity"],
            )
        else:
            rules["6_high_quality_identity"] = _pass_fail(
                True,
                evidence,
                _RULE_TEXT["6_high_quality_identity"],
            )
    else:
        rules["6_high_quality_identity"] = _not_evaluated(
            {
                "evaluated_count": 0,
                "peak_prominence_threshold": 0.05,
                "stripe_standard_deviation_threshold": 5.0,
            },
            _RULE_TEXT["6_high_quality_identity"],
        )

    trend_groups = trends["groups"]
    missing_temperature_ids = sorted(
        record_id
        for group in trend_groups.values()
        for record_id in group["missing_temperature_record_ids"]
    )
    missing_uncertainty_ids = sorted(
        record_id
        for group in trend_groups.values()
        for record_id in group["missing_measurement_uncertainty_record_ids"]
    )
    incomplete_features: list[dict[str, object]] = []
    for group_name, group in trend_groups.items():
        if len(group["samples"]) < 2:
            incomplete_features.append(
                {
                    "group": group_name,
                    "feature": "all",
                    "required_sample_count": 2,
                    "evaluated_pair_count": len(group["samples"]),
                    "unique_temperature_count": len(
                        {sample["temperature_K"] for sample in group["samples"]}
                    ),
                }
            )
            continue
        for feature, evidence in group["features"].items():
            if (
                evidence["evaluated_pair_count"] != evidence["required_sample_count"]
                or evidence["evaluated_pair_count"] < 2
                or evidence["unique_temperature_count"] < 2
            ):
                incomplete_features.append(
                    {
                        "group": group_name,
                        "feature": feature,
                        "required_sample_count": evidence["required_sample_count"],
                        "evaluated_pair_count": evidence["evaluated_pair_count"],
                        "unique_temperature_count": evidence["unique_temperature_count"],
                    }
                )
    if trend_groups:
        direction_reversals: list[dict[str, str]] = []
        jump_violations: list[dict[str, object]] = []
        for group_name, group in trend_groups.items():
            for feature, evidence in group["features"].items():
                if evidence["direction_reversed"]:
                    direction_reversals.append({"group": group_name, "feature": feature})
                for violation in evidence["jump_violations"]:
                    jump_violations.append({"group": group_name, "feature": feature, **violation})
        incomplete_groups = {item["group"] for item in incomplete_features}
        incomplete_groups.update(
            group_name
            for group_name, group in trend_groups.items()
            if group["missing_temperature_record_ids"]
            or group["missing_measurement_uncertainty_record_ids"]
        )
        evidence = {
            "required_group_count": len(trend_groups),
            "evaluated_group_count": len(trend_groups) - len(incomplete_groups),
            "incomplete_features": incomplete_features,
            "missing_temperature_record_ids": missing_temperature_ids,
            "missing_measurement_uncertainty_record_ids": missing_uncertainty_ids,
            "direction_reversals": direction_reversals,
            "jump_violations": jump_violations,
            "jump_factor_threshold": 3.0,
        }
        if incomplete_features or missing_temperature_ids or missing_uncertainty_ids:
            rules["7_temperature_trends"] = _not_evaluated(
                evidence,
                _RULE_TEXT["7_temperature_trends"],
            )
        else:
            rules["7_temperature_trends"] = _pass_fail(
                not direction_reversals and not jump_violations,
                evidence,
                _RULE_TEXT["7_temperature_trends"],
            )
    else:
        rules["7_temperature_trends"] = _not_evaluated(
            {
                "required_group_count": 0,
                "evaluated_group_count": 0,
                "incomplete_features": [],
                "missing_temperature_record_ids": [],
                "missing_measurement_uncertainty_record_ids": [],
                "jump_factor_threshold": 3.0,
            },
            _RULE_TEXT["7_temperature_trends"],
        )

    manual_flagged_ids = {case.record_id for case in cases if case.manually_flagged}
    automated_review_ids = {flag["record_id"] for flag in flags}.difference(manual_flagged_ids)
    failed_ids = {
        record_id
        for record_id, row in residual.items()
        if row["fit_status"] == "failed" and record_id not in manual_flagged_ids
    }
    evaluated_ids = {
        record_id
        for record_id, row in residual.items()
        if row["evaluation_status"] == "evaluated"
        and record_id not in manual_flagged_ids
        and record_id not in failed_ids
    }
    not_evaluated_ids = {case.record_id for case in cases}.difference(
        manual_flagged_ids, failed_ids, evaluated_ids
    )
    reconciled = (
        len(cases) == manifest_row_count
        and len(evaluated_ids) + len(failed_ids) + len(manual_flagged_ids) == manifest_row_count
        and not not_evaluated_ids
    )
    rules["8_manifest_reconciliation"] = _pass_fail(
        reconciled,
        {
            "manifest_row_count": manifest_row_count,
            "report_case_count": len(cases),
            "evaluated": len(evaluated_ids),
            "failed_fit": len(failed_ids),
            "manually_flagged": len(manual_flagged_ids),
            "automated_review_required": len(automated_review_ids),
            "automated_review_record_ids": sorted(automated_review_ids),
            "not_evaluated": len(not_evaluated_ids),
            "not_evaluated_record_ids": sorted(not_evaluated_ids),
        },
        _RULE_TEXT["8_manifest_reconciliation"],
    )

    gate = _scientific_gate(cases)
    if not gate["pass"]:
        rules = {
            name: _not_evaluated(rule["evidence"], rule["rule"]) for name, rule in rules.items()
        }
        status = "not_evaluated"
    else:
        statuses = {rule["status"] for rule in rules.values()}
        if "not_evaluated" in statuses:
            status = "not_evaluated"
        elif "review_required" in statuses:
            status = "review_required"
        elif "fail" in statuses:
            status = "fail"
        else:
            status = "pass"
    return {
        "status": status,
        "scientific_gate": gate,
        "rules": rules,
    }


def generate_evaluation_report(
    cases: list[EvaluationCase],
    output_dir: str | Path,
    *,
    manifest_row_count: int | None = None,
) -> dict[str, object]:
    """Write complete metrics, visual evidence, trends and gated acceptance."""
    case_list = list(cases)
    if len({case.record_id for case in case_list}) != len(case_list):
        raise ValueError("record IDs must be unique")
    locked_count = len(case_list) if manifest_row_count is None else manifest_row_count
    if type(locked_count) is not int or locked_count < 0:
        raise ValueError("manifest_row_count must be a non-negative integer")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows, arrays_by_case = _evaluate_rows(case_list)
    _write_rows(rows, destination / "per_file_metrics.csv")
    _write_json(_summary(rows, locked_count), destination / "summary.json")
    _write_json(_worst_cases(rows), destination / "worst_cases.json")
    _write_figures(case_list, arrays_by_case, destination)
    residual_rows = {
        str(row["record_id"]): row for row in rows if row["method"] == "ResidualDenoiser2D"
    }
    flags = _high_quality_flags(case_list, residual_rows)
    _write_flags(flags, destination / "high_quality_flags.csv")
    trends = _temperature_trends(case_list)
    _write_json(trends, destination / "temperature_trends.json")
    acceptance = _acceptance(case_list, rows, flags, trends, locked_count)
    _write_json(acceptance, destination / "acceptance.json")
    return _json_safe(acceptance)
