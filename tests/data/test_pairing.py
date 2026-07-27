"""Tests for conservative ARPES pairing decisions."""

from dataclasses import replace

import pytest

from dccnn_arpes.data.pairing import classify_pair, propose_pairs
from dccnn_arpes.data.schema import ManifestRecord


def _record(**changes) -> ManifestRecord:
    values = {
        "record_id": "left",
        "source_path": "D:/source/left.pxt",
        "sample_id": "sample-a",
        "acquisition_group": "run-1",
        "scan_type": "cut",
        "temperature_K": 20.0,
        "photon_energy_eV": 21.2,
        "polarization": "LH",
        "position_x": 1.0,
        "position_y": 2.0,
        "position_z": 3.0,
        "position_polar": 4.0,
        "position_tilt": 5.0,
        "position_azimuth": 6.0,
        "energy_axis": (-0.2, 0.0, 0.2),
        "angle_axis": (-10.0, 0.0, 10.0),
        "acquisition_time_s": 5.0,
        "sweep_count": 2,
        "review_status": "reviewed",
        "notes": "surface_state=fresh",
    }
    values.update(changes)
    return ManifestRecord(**values)


def test_identical_physical_settings_with_a_longer_acquisition_are_level_a():
    """Removing acquisition-scale comparison must make this test fail."""
    decision = classify_pair(_record(), _record(record_id="right", acquisition_time_s=20.0, sweep_count=8))

    assert decision.accepted is True
    assert decision.pair_type == "A"
    assert decision.exclusion_reason == ""


def test_independent_repeats_with_identical_settings_are_level_b():
    """Classifying equal-scale repeats as A must make this test fail."""
    decision = classify_pair(_record(), _record(record_id="right", source_path="D:/source/right.pxt"))

    assert decision.accepted is True
    assert decision.pair_type == "B"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("temperature_K", 21.0, "temperature_K"),
        ("photon_energy_eV", 22.0, "photon_energy_eV"),
        ("polarization", "LV", "polarization"),
        ("position_x", 1.1, "position_x"),
        ("notes", "surface_state=aged", "surface_state"),
        ("energy_axis", (-0.2, 0.0, 0.3), "energy_axis"),
        ("energy_axis", (-0.2, 0.2), "shape"),
        ("scan_type", "map", "scan_type"),
    ],
)
def test_prohibited_physical_differences_are_rejected(field, value, reason):
    """Dropping any physical comparison in this table must make a case fail."""
    decision = classify_pair(_record(), _record(record_id="right", **{field: value}))

    assert decision.accepted is False
    assert decision.exclusion_reason == reason


def test_missing_physical_metadata_is_not_assumed_equal():
    """Treating absent settings as equal must make this test fail."""
    decision = classify_pair(_record(), _record(record_id="right", polarization=""))

    assert decision.accepted is False
    assert decision.exclusion_reason == "polarization"


def test_inherited_excel_metadata_remains_needs_review():
    """Auto-accepting inherited workbook metadata must make this test fail."""
    decision = classify_pair(_record(), _record(record_id="right", review_status="needs_review"))

    assert decision.accepted is False
    assert decision.review_status == "needs_review"


def test_propose_pairs_assigns_stable_ids_only_to_accepted_candidates():
    """Including rejected candidates in pair records must make this test fail."""
    left = _record()
    accepted = _record(record_id="right", source_path="D:/source/right.pxt")
    rejected = replace(_record(record_id="other"), temperature_K=30.0)

    pairs, decisions = propose_pairs([left, accepted, rejected])

    assert len(pairs) == 1
    assert pairs[0].pair_type == "B"
    assert pairs[0].left_record_id == "left"
    assert any(decision.exclusion_reason == "temperature_K" for decision in decisions)
