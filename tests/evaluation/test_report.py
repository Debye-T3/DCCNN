"""Artifact and acceptance tests for physical-fidelity reports."""

import csv
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dccnn_arpes.cli import eval as eval_cli
from dccnn_arpes.evaluation.report import (
    EvaluationCase,
    _peak_candidates,
    generate_evaluation_report,
)
from dccnn_arpes.io import write_cut


def _cut(
    *,
    center_eV: float = 0.1,
    center_alpha: float = -0.2,
    extra_vertical_stripe: float = 0.0,
    scientific: bool = False,
) -> xr.DataArray:
    eV = np.linspace(-1.0, 1.0, 81)
    alpha = np.linspace(-2.0, 2.0, 71)
    values = 0.1 + 3.0 * np.exp(-0.5 * ((eV[:, None] - center_eV) / 0.14) ** 2) * np.exp(
        -0.5 * ((alpha[None, :] - center_alpha) / 0.32) ** 2
    )
    if extra_vertical_stripe:
        values[:, 5] += extra_vertical_stripe
    return xr.DataArray(
        values,
        dims=("eV", "alpha"),
        coords={"eV": eV, "alpha": alpha},
        name="intensity",
        attrs={
            "acquisition_time_s": 2.0,
            "scientific_use": str(scientific).lower(),
            "smoke_test": str(not scientific).lower(),
        },
    )


def _case(record_id: str, *, scientific: bool = False, **changes) -> EvaluationCase:
    reference = _cut(scientific=scientific)
    values = {
        "record_id": record_id,
        "input_da": _cut(center_eV=0.11, scientific=scientific),
        "output_da": reference.copy(deep=True),
        "reference_da": reference,
        "legacy_da": _cut(center_eV=0.105, scientific=scientific),
        "pair_type": "A",
        "scientific_eligible": True,
    }
    values.update(changes)
    return EvaluationCase(**values)


def test_non_scientific_output_writes_evidence_but_never_passes_acceptance(tmp_path):
    """Allowing a smoke output to produce a passing acceptance file must fail."""
    output_dir = tmp_path / "report"

    acceptance = generate_evaluation_report([_case("smoke")], output_dir)

    saved = json.loads((output_dir / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance == saved
    assert saved["status"] == "not_evaluated"
    assert saved["scientific_gate"]["pass"] is False
    assert "non-scientific" in saved["scientific_gate"]["reason"]
    assert list(saved["rules"]) == [
        "1_paired_test_improvement",
        "2_legacy_nrmse_improvement",
        "3_peak_position_fidelity",
        "4_fwhm_fidelity",
        "5_integrated_intensity_fidelity",
        "6_high_quality_identity",
        "7_temperature_trends",
        "8_manifest_reconciliation",
    ]
    for rule in saved["rules"].values():
        assert set(rule) == {"rule", "status", "pass", "evidence"}
        assert rule["status"] == "not_evaluated"
        assert rule["pass"] is None
    assert saved["rules"]["1_paired_test_improvement"]["evidence"]["threshold"] == 0.80
    assert saved["rules"]["2_legacy_nrmse_improvement"]["evidence"]["threshold"] == 0.10
    assert saved["rules"]["4_fwhm_fidelity"]["evidence"]["threshold"] == 0.10
    assert saved["rules"]["5_integrated_intensity_fidelity"]["evidence"]["threshold"] == 0.05
    high_quality_evidence = saved["rules"]["6_high_quality_identity"]["evidence"]
    assert high_quality_evidence["peak_prominence_threshold"] == 0.05
    assert high_quality_evidence["stripe_standard_deviation_threshold"] == 5.0
    assert saved["rules"]["7_temperature_trends"]["evidence"]["jump_factor_threshold"] == 3.0


def test_report_writes_all_methods_metrics_summaries_and_fixed_scale_figures(tmp_path):
    """Omitting a baseline, file, summary, difference panel, EDC or MDC must fail."""
    cases = [_case("cut-one"), _case("cut-two")]
    output_dir = tmp_path / "report"

    generate_evaluation_report(cases, output_dir)

    rows = pd.read_csv(output_dir / "per_file_metrics.csv")
    assert len(rows) == 10
    assert set(rows["record_id"]) == {"cut-one", "cut-two"}
    assert set(rows["method"]) == {
        "raw_input",
        "gaussian",
        "median",
        "LegacyCCNN",
        "ResidualDenoiser2D",
    }
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["methods"]) == set(rows["method"])
    assert summary["manifest_row_count"] == 2
    worst_cases = json.loads((output_dir / "worst_cases.json").read_text(encoding="utf-8"))
    assert worst_cases["metric"] == "nrmse"
    for record_id in ("cut-one", "cut-two"):
        for suffix in ("preview", "edc", "mdc"):
            figure = output_dir / "figures" / f"{record_id}_{suffix}.png"
            assert figure.is_file()
            assert figure.stat().st_size > 0


def test_failed_fits_and_high_quality_flags_remain_visible_and_reconcile(tmp_path):
    """Dropping failed or flagged samples from report accounting must fail."""
    template = _cut()
    flat = xr.full_like(template, 2.0)
    flat.attrs.update(template.attrs)
    failed = _case(
        "flat-fit",
        input_da=flat,
        output_da=flat.copy(deep=True),
        reference_da=flat.copy(deep=True),
        legacy_da=flat.copy(deep=True),
    )
    flagged = _case(
        "new-stripe",
        output_da=_cut(extra_vertical_stripe=0.8),
        high_quality_identity=True,
        manually_flagged=True,
        manual_flag_reason="controlled manual review fixture",
    )
    output_dir = tmp_path / "report"

    generate_evaluation_report([failed, flagged], output_dir)

    rows = pd.read_csv(output_dir / "per_file_metrics.csv")
    residual_rows = rows[rows["method"] == "ResidualDenoiser2D"].set_index("record_id")
    assert residual_rows.loc["flat-fit", "fit_status"] == "failed"
    assert np.isnan(residual_rows.loc["flat-fit", "fwhm_relative_error"])
    flags = list(
        csv.DictReader((output_dir / "high_quality_flags.csv").open(encoding="utf-8", newline=""))
    )
    assert {row["record_id"] for row in flags} == {"new-stripe"}
    assert "stripe" in flags[0]["reasons"]
    acceptance = json.loads((output_dir / "acceptance.json").read_text(encoding="utf-8"))
    evidence = acceptance["rules"]["8_manifest_reconciliation"]["evidence"]
    assert evidence["manifest_row_count"] == 2
    assert evidence["evaluated"] + evidence["failed_fit"] + evidence["manually_flagged"] == 2


def test_temperature_features_are_sorted_and_compared_as_series_not_targets(tmp_path):
    """Keeping manifest order or comparing adjacent temperatures as references must fail."""
    cases = [
        _case(
            "warm",
            temperature_K=80.0,
            temperature_group="series-a",
            input_da=_cut(center_eV=0.18),
            output_da=_cut(center_eV=0.17),
        ),
        _case(
            "cold",
            temperature_K=20.0,
            temperature_group="series-a",
            input_da=_cut(center_eV=0.12),
            output_da=_cut(center_eV=0.11),
        ),
        _case(
            "middle",
            temperature_K=50.0,
            temperature_group="series-a",
            input_da=_cut(center_eV=0.15),
            output_da=_cut(center_eV=0.14),
        ),
    ]
    output_dir = tmp_path / "report"

    generate_evaluation_report(cases, output_dir)

    trends = json.loads((output_dir / "temperature_trends.json").read_text(encoding="utf-8"))
    series = trends["groups"]["series-a"]
    assert [row["record_id"] for row in series["samples"]] == ["cold", "middle", "warm"]
    assert [row["temperature_K"] for row in series["samples"]] == [20.0, 50.0, 80.0]
    assert set(series["features"]) == {
        "peak_eV",
        "peak_alpha",
        "fwhm_eV",
        "fwhm_alpha",
        "integrated_intensity",
    }
    assert all(row["reference_record_id"] == row["record_id"] for row in series["samples"])


def test_high_quality_peak_shift_above_one_sampling_step_requires_review(tmp_path):
    """Ignoring the global peak-fidelity threshold on an identity cut must fail."""
    reference = _cut(center_eV=0.10)
    eV_step = float(reference.eV.values[1] - reference.eV.values[0])
    shifted = _case(
        "shifted-identity",
        output_da=_cut(center_eV=0.10 + 2.0 * eV_step),
        high_quality_identity=True,
    )
    output_dir = tmp_path / "report"

    generate_evaluation_report([shifted], output_dir)

    flags = list(
        csv.DictReader((output_dir / "high_quality_flags.csv").open(encoding="utf-8", newline=""))
    )
    assert len(flags) == 1
    assert "peak-position error exceeds one sampling step" in flags[0]["reasons"]
    assert "new stripe" not in flags[0]["reasons"]
    acceptance = json.loads((output_dir / "acceptance.json").read_text(encoding="utf-8"))
    reconciliation = acceptance["rules"]["8_manifest_reconciliation"]["evidence"]
    assert reconciliation["manually_flagged"] == 0
    assert reconciliation["automated_review_required"] == 1
    assert reconciliation["evaluated"] == 1
    assert reconciliation["not_evaluated"] == 0
    assert reconciliation["not_evaluated_record_ids"] == []


@pytest.mark.parametrize("endpoint_width", [1, 3])
def test_high_quality_endpoint_peak_above_five_percent_requires_review(tmp_path, endpoint_width):
    """Endpoint peaks must not escape detection because scipy find_peaks omits endpoints."""
    reference = _cut()
    output = reference.copy(deep=True)
    output.values[:endpoint_width, :] += 0.5
    case = _case(
        "endpoint-peak",
        output_da=output,
        high_quality_identity=True,
    )
    output_dir = tmp_path / "report"

    generate_evaluation_report([case], output_dir)

    flags = list(
        csv.DictReader((output_dir / "high_quality_flags.csv").open(encoding="utf-8", newline=""))
    )
    assert len(flags) == 1
    assert "new eV peak prominence" in flags[0]["reasons"]


def test_endpoint_peak_candidate_must_be_higher_than_adjacent_sample():
    """A high endpoint below its neighbor must not be promoted to a peak."""
    candidates = _peak_candidates(np.asarray([4.0, 5.0, 1.0, 1.0, 1.0]), prominence=0.1)

    assert [index for index, _ in candidates] == [1]


@pytest.mark.parametrize(
    "profile",
    [
        [4.0, 4.0, 5.0, 1.0, 1.0],
        [1.0, 1.0, 5.0, 4.0, 4.0],
    ],
)
def test_endpoint_plateau_must_be_higher_than_first_different_sample(profile):
    """A tied boundary shoulder that rises inward must not become an endpoint peak."""
    candidates = _peak_candidates(np.asarray(profile), prominence=0.1)

    assert [index for index, _ in candidates] == [2]


def test_incomplete_locked_pairs_never_shrink_acceptance_denominators(tmp_path):
    """Silently dropping a locked pair with missing evidence must fail this test."""
    complete = _case("complete-a", pair_type="A")
    incomplete = _case("incomplete-b", pair_type="B", output_da=None)
    output_dir = tmp_path / "report"

    generate_evaluation_report([complete, incomplete], output_dir)

    acceptance = json.loads((output_dir / "acceptance.json").read_text(encoding="utf-8"))
    first = acceptance["rules"]["1_paired_test_improvement"]["evidence"]
    second = acceptance["rules"]["2_legacy_nrmse_improvement"]["evidence"]
    peak = acceptance["rules"]["3_peak_position_fidelity"]["evidence"]
    width = acceptance["rules"]["4_fwhm_fidelity"]["evidence"]
    integral = acceptance["rules"]["5_integrated_intensity_fidelity"]["evidence"]
    assert first["paired_test_count"] == 2
    assert first["missing_record_ids"] == ["incomplete-b"]
    assert second["paired_test_count"] == 2
    assert second["missing_record_ids"] == ["incomplete-b"]
    assert peak["required_count"] == 2
    assert peak["missing_record_ids"] == ["incomplete-b"]
    assert width["required_count"] == 2
    assert width["missing_record_ids"] == ["incomplete-b"]
    assert integral["required_count"] == 2
    assert integral["missing_record_ids"] == ["incomplete-b"]


def test_failed_temperature_features_are_not_treated_as_no_violation(tmp_path):
    """A trend with unavailable fitted features must not be encoded as passing evidence."""
    valid = _case(
        "valid-temperature",
        temperature_K=20.0,
        temperature_group="series",
    )
    flat_template = _cut()
    flat = xr.full_like(flat_template, 2.0)
    flat.attrs.update(flat_template.attrs)
    failed = _case(
        "failed-temperature",
        temperature_K=50.0,
        temperature_group="series",
        input_da=flat,
        output_da=flat.copy(deep=True),
        reference_da=flat.copy(deep=True),
        legacy_da=flat.copy(deep=True),
    )
    output_dir = tmp_path / "report"

    generate_evaluation_report([valid, failed], output_dir)

    acceptance = json.loads((output_dir / "acceptance.json").read_text(encoding="utf-8"))
    trend = acceptance["rules"]["7_temperature_trends"]["evidence"]
    assert trend["incomplete_features"]
    assert any(item["group"] == "series" for item in trend["incomplete_features"])


def test_coordinate_mismatch_keeps_failed_row_and_writes_preview_placeholder(tmp_path):
    """Plot subtraction must not abort the report after metrics retain an unaligned output."""
    case = _case("mismatch", scientific=True)
    mismatched = case.output_da.isel(alpha=slice(None, -1))
    case = _case(
        "mismatch",
        scientific=True,
        output_da=mismatched,
        high_quality_identity=True,
    )
    output_dir = tmp_path / "report"

    generate_evaluation_report([case], output_dir)

    rows = pd.read_csv(output_dir / "per_file_metrics.csv")
    residual = rows[rows["method"] == "ResidualDenoiser2D"].iloc[0]
    assert residual["evaluation_status"] == "failed"
    preview = output_dir / "figures" / "mismatch_preview.png"
    assert preview.is_file()
    assert preview.stat().st_size > 0
    acceptance = json.loads((output_dir / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["scientific_gate"]["pass"] is True
    assert acceptance["status"] == "not_evaluated"
    peak_rule = acceptance["rules"]["3_peak_position_fidelity"]
    assert peak_rule["status"] == "not_evaluated"
    assert peak_rule["evidence"]["missing_record_ids"] == ["mismatch"]
    identity_rule = acceptance["rules"]["6_high_quality_identity"]
    assert identity_rule["status"] == "not_evaluated"
    assert identity_rule["evidence"]["missing_record_ids"] == ["mismatch"]


def test_repeated_samples_at_one_temperature_do_not_establish_a_trend(tmp_path):
    """Two rows at one temperature must not be mistaken for a temperature series."""
    cases = [
        _case("replicate-one", temperature_K=20.0, temperature_group="replicates"),
        _case("replicate-two", temperature_K=20.0, temperature_group="replicates"),
    ]
    output_dir = tmp_path / "report"

    generate_evaluation_report(cases, output_dir)

    acceptance = json.loads((output_dir / "acceptance.json").read_text(encoding="utf-8"))
    evidence = acceptance["rules"]["7_temperature_trends"]["evidence"]
    assert evidence["incomplete_features"]
    assert all(
        item["unique_temperature_count"] == 1
        for item in evidence["incomplete_features"]
        if item["group"] == "replicates"
    )


def test_temperature_group_keeps_members_with_missing_temperature_as_incomplete(tmp_path):
    """A missing temperature must not disappear before rule-7 population accounting."""
    cases = [
        _case(
            "known-temperature",
            temperature_K=20.0,
            temperature_group="series",
            measurement_uncertainty=0.01,
        ),
        _case(
            "missing-temperature",
            temperature_K=None,
            temperature_group="series",
            measurement_uncertainty=0.01,
        ),
    ]
    output_dir = tmp_path / "report"

    generate_evaluation_report(cases, output_dir)

    trends = json.loads((output_dir / "temperature_trends.json").read_text(encoding="utf-8"))
    assert [row["record_id"] for row in trends["groups"]["series"]["samples"]] == [
        "known-temperature",
        "missing-temperature",
    ]
    acceptance = json.loads((output_dir / "acceptance.json").read_text(encoding="utf-8"))
    rule = acceptance["rules"]["7_temperature_trends"]
    assert rule["status"] == "not_evaluated"
    assert rule["evidence"]["missing_temperature_record_ids"] == ["missing-temperature"]


def test_missing_measurement_uncertainty_makes_temperature_rule_incomplete(tmp_path):
    """Missing uncertainty must remain missing rather than becoming zero evidence."""
    cases = [
        _case(
            "known-uncertainty",
            temperature_K=20.0,
            temperature_group="series",
            measurement_uncertainty=0.01,
        ),
        _case(
            "missing-uncertainty",
            temperature_K=50.0,
            temperature_group="series",
            measurement_uncertainty=None,
        ),
    ]
    output_dir = tmp_path / "report"

    generate_evaluation_report(cases, output_dir)

    acceptance = json.loads((output_dir / "acceptance.json").read_text(encoding="utf-8"))
    rule = acceptance["rules"]["7_temperature_trends"]
    assert rule["status"] == "not_evaluated"
    assert rule["evidence"]["missing_measurement_uncertainty_record_ids"] == ["missing-uncertainty"]


def test_cli_writes_controlled_report_under_allowed_root_as_not_evaluated(
    tmp_path, monkeypatch, capsys
):
    """CLI smoke data must create evidence without being promotable to scientific acceptance."""
    allowed_root = tmp_path / "outputs"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    paths = {}
    for name, cut in {
        "input": _cut(center_eV=0.12),
        "output": _cut(center_eV=0.10),
        "reference": _cut(center_eV=0.10),
        "legacy": _cut(center_eV=0.11),
    }.items():
        path = artifact_dir / f"{name}.h5"
        write_cut(cut, path)
        paths[name] = path
    split_path = tmp_path / "controlled-test.csv"
    with split_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "record_id",
                "converted_path",
                "denoised_path",
                "reference_path",
                "legacy_output_path",
                "pair_type",
                "split",
                "review_status",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "record_id": "controlled",
                "converted_path": paths["input"],
                "denoised_path": paths["output"],
                "reference_path": paths["reference"],
                "legacy_output_path": paths["legacy"],
                "pair_type": "A",
                "split": "test",
                "review_status": "approved",
            }
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 20260727\n", encoding="utf-8")
    destination = allowed_root / "controlled-evaluation"
    monkeypatch.setattr(eval_cli, "_REPORT_ROOT", allowed_root)

    eval_cli.main(
        [
            "--config",
            str(config_path),
            "--split",
            str(split_path),
            "--output",
            str(destination),
        ]
    )

    acceptance = json.loads((destination / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["status"] == "not_evaluated"
    assert acceptance["scientific_gate"]["non_scientific_record_ids"] == ["controlled"]
    assert "acceptance_status=not_evaluated" in capsys.readouterr().out


def test_cli_rejects_report_destinations_outside_dccnn_outputs(tmp_path, monkeypatch):
    """Writing a report outside the configured output root must fail before creating it."""
    allowed_root = tmp_path / "outputs"
    monkeypatch.setattr(eval_cli, "_REPORT_ROOT", allowed_root)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 20260727\n", encoding="utf-8")
    split_path = tmp_path / "test.csv"
    split_path.write_text("record_id\none\n", encoding="utf-8")
    forbidden = tmp_path / "forbidden"

    with pytest.raises(ValueError, match="must be under"):
        eval_cli.main(
            [
                "--config",
                str(config_path),
                "--split",
                str(split_path),
                "--output",
                str(forbidden),
            ]
        )

    assert not forbidden.exists()


def test_cli_preserves_standard_split_rows_when_scientific_artifacts_do_not_exist(
    tmp_path, monkeypatch
):
    """A canonical ManifestRecord split must yield evidence, never invented comparisons."""
    allowed_root = tmp_path / "outputs"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    input_path = artifact_dir / "input.h5"
    write_cut(_cut(), input_path)
    split_path = tmp_path / "test.csv"
    with split_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "record_id",
                "converted_path",
                "pair_type",
                "split",
                "review_status",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "record_id": "locked-row",
                "converted_path": input_path,
                "pair_type": "B",
                "split": "test",
                "review_status": "approved",
            }
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 20260727\n", encoding="utf-8")
    destination = allowed_root / "standard-split-evaluation"
    monkeypatch.setattr(eval_cli, "_REPORT_ROOT", allowed_root)

    eval_cli.main(
        [
            "--config",
            str(config_path),
            "--split",
            str(split_path),
            "--output",
            str(destination),
        ]
    )

    acceptance = json.loads((destination / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["status"] == "not_evaluated"
    assert acceptance["scientific_gate"]["missing_artifact_record_ids"] == ["locked-row"]
    rows = pd.read_csv(destination / "per_file_metrics.csv")
    assert len(rows) == 5
    assert set(rows["record_id"]) == {"locked-row"}


def test_cli_fallback_temperature_group_keeps_missing_temperature_standard_split_row(
    tmp_path, monkeypatch
):
    """A blank temperature must not remove a same-sample row from fallback trend grouping."""
    allowed_root = tmp_path / "outputs"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    input_path = artifact_dir / "input.h5"
    write_cut(_cut(), input_path)
    split_path = tmp_path / "test.csv"
    with split_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "record_id",
                "converted_path",
                "sample_id",
                "temperature_K",
                "pair_type",
                "split",
                "review_status",
            ),
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "record_id": "known-temperature",
                    "converted_path": input_path,
                    "sample_id": "sample-series",
                    "temperature_K": "20.0",
                    "pair_type": "B",
                    "split": "test",
                    "review_status": "approved",
                },
                {
                    "record_id": "missing-temperature",
                    "converted_path": input_path,
                    "sample_id": "sample-series",
                    "temperature_K": "",
                    "pair_type": "B",
                    "split": "test",
                    "review_status": "approved",
                },
            ]
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 20260727\n", encoding="utf-8")
    destination = allowed_root / "fallback-temperature-group"
    monkeypatch.setattr(eval_cli, "_REPORT_ROOT", allowed_root)

    eval_cli.main(
        [
            "--config",
            str(config_path),
            "--split",
            str(split_path),
            "--output",
            str(destination),
        ]
    )

    trends = json.loads((destination / "temperature_trends.json").read_text(encoding="utf-8"))
    assert [row["record_id"] for row in trends["groups"]["sample-series"]["samples"]] == [
        "known-temperature",
        "missing-temperature",
    ]
    acceptance = json.loads((destination / "acceptance.json").read_text(encoding="utf-8"))
    rule = acceptance["rules"]["7_temperature_trends"]
    assert rule["status"] == "not_evaluated"
    assert rule["evidence"]["missing_temperature_record_ids"] == ["missing-temperature"]
