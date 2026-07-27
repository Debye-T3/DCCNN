"""Evaluation command-line interface."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml

from dccnn_arpes.evaluation import EvaluationCase, generate_evaluation_report
from dccnn_arpes.io import load_cut

_REPORT_ROOT = Path(r"D:\Projects\dccnn\outputs")
_SCIENTIFIC_REVIEWS = {"reviewed", "approved", "manually_approved"}


def _optional_float(row: Mapping[str, str], name: str) -> float | None:
    value = row.get(name, "").strip()
    if not value:
        return None
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite when provided")
    return converted


def _flag(row: Mapping[str, str], name: str) -> bool:
    return row.get(name, "").strip().casefold() in {"1", "true", "yes"}


def _path_value(row: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return value
    return ""


def _load_optional(path_text: str, split_directory: Path):
    if not path_text:
        return None, "path is absent"
    path = Path(path_text)
    if not path.is_absolute():
        path = split_directory / path
    try:
        return load_cut(path), ""
    except (OSError, TypeError, ValueError) as error:
        return None, f"{path}: {error}"


def _case_from_row(row: Mapping[str, str], split_directory: Path) -> EvaluationCase:
    record_id = row.get("record_id", "").strip()
    input_path = _path_value(row, "input_path", "converted_path")
    output_path = _path_value(row, "output_path", "denoised_path")
    reference_path = _path_value(row, "reference_path")
    legacy_path = _path_value(row, "legacy_output_path")
    input_da, input_issue = _load_optional(input_path, split_directory)
    output_da, output_issue = _load_optional(output_path, split_directory)
    reference_da, reference_issue = _load_optional(reference_path, split_directory)
    legacy_da, legacy_issue = _load_optional(legacy_path, split_directory)
    artifact_issues = [
        f"{name}: {issue}"
        for name, issue in (
            ("input", input_issue),
            ("output", output_issue),
            ("reference", reference_issue),
            ("legacy", legacy_issue),
        )
        if issue
    ]
    eligibility_issues = list(artifact_issues)
    if row.get("split", "").strip().casefold() != "test":
        eligibility_issues.append("row is not in the locked test split")
    if row.get("review_status", "").strip().casefold() not in _SCIENTIFIC_REVIEWS:
        eligibility_issues.append("row is not scientifically reviewed")
    if row.get("exclusion_reason", "").strip():
        eligibility_issues.append("row has an exclusion reason")
    quality_flag = row.get("quality_flag", "").strip().casefold()
    high_quality = _flag(row, "high_quality_identity") or quality_flag in {
        "high_quality",
        "high_quality_identity",
        "identity",
    }
    manually_flagged = _flag(row, "manually_flagged") or quality_flag in {
        "flagged",
        "needs_review",
    }
    temperature = _optional_float(row, "temperature_K")
    uncertainty = _optional_float(row, "measurement_uncertainty")
    temperature_group = row.get("temperature_group", "").strip()
    if not temperature_group and temperature is not None:
        temperature_group = (
            row.get("sample_id", "").strip() or row.get("acquisition_group", "").strip()
        )
    return EvaluationCase(
        record_id=record_id,
        input_da=input_da,
        output_da=output_da,
        reference_da=reference_da,
        legacy_da=legacy_da,
        pair_type=row.get("pair_type", "").strip(),
        temperature_K=temperature,
        temperature_group=temperature_group,
        measurement_uncertainty=uncertainty,
        high_quality_identity=high_quality,
        manually_flagged=manually_flagged,
        manual_flag_reason=row.get("manual_flag_reason", "").strip(),
        scientific_eligible=not eligibility_issues,
        eligibility_reason="; ".join(eligibility_issues),
    )


def _validate_output_path(output: Path) -> Path:
    destination = output.expanduser().resolve(strict=False)
    root = _REPORT_ROOT.expanduser().resolve(strict=False)
    if not destination.is_relative_to(root):
        raise ValueError(f"evaluation output must be under {root}")
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    """Run the evaluation command-line interface."""
    parser = argparse.ArgumentParser(description="Evaluate an ARPES denoising model.")
    parser.add_argument("--config", required=True, help="Training/evaluation YAML config.")
    parser.add_argument("--split", required=True, help="Locked test-manifest CSV.")
    parser.add_argument("--output", required=True, help="Report directory under DCCNN outputs.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError("config must contain a YAML mapping")

    split_path = Path(args.split)
    with split_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "record_id" not in reader.fieldnames:
            raise ValueError("split CSV must contain record_id")
        rows = list(reader)
    cases = [_case_from_row(row, split_path.parent) for row in rows]
    destination = _validate_output_path(Path(args.output))
    acceptance = generate_evaluation_report(
        cases,
        destination,
        manifest_row_count=len(rows),
    )
    print(f"report_path={destination}", flush=True)
    print(f"acceptance_status={acceptance['status']}", flush=True)


if __name__ == "__main__":
    main()
