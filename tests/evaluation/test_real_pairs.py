"""Tests for count-rate-normalized real A-pair evaluation."""

import numpy as np
import pytest
import xarray as xr

from dccnn_arpes.data.schema import ManifestRecord
from dccnn_arpes.evaluation.real_pairs import (
    build_pair_row,
    count_rate_normalize,
    effective_exposure,
    orient_pair,
)


def _record(record_id: str, **changes) -> ManifestRecord:
    values = {
        "record_id": record_id,
        "source_path": f"D:/source/{record_id}.pxt",
        "converted_path": f"D:/converted/{record_id}.h5",
        "review_status": "manually_approved",
    }
    values.update(changes)
    return ManifestRecord(**values)


def test_orient_pair_uses_acquisition_time_before_sweeps():
    short = _record("short", acquisition_time_s=2.0, sweep_count=20)
    long = _record("long", acquisition_time_s=8.0, sweep_count=1)

    assert orient_pair(long, short) == (short, long)


def test_orient_pair_falls_back_to_sweeps():
    one = _record("one", sweep_count=1)
    ten = _record("ten", sweep_count=10)

    assert orient_pair(ten, one) == (one, ten)


def test_orient_pair_rejects_missing_exposure():
    with pytest.raises(ValueError, match="exposure scale"):
        effective_exposure(_record("missing"))


def test_count_rate_normalize_scales_values_and_neutralizes_attrs():
    data = xr.DataArray(
        np.full((2, 2), 6.0),
        dims=("eV", "alpha"),
        coords={"eV": [1.0, 2.0], "alpha": [3.0, 4.0]},
        attrs={"acquisition_time_s": 3.0, "sample_name": "sample"},
    )

    normalized = count_rate_normalize(data, 3.0)

    np.testing.assert_allclose(normalized.values, 2.0)
    assert normalized.attrs["acquisition_time_s"] == 1.0
    assert normalized.attrs.get("sweep_count") in (None, "")
    assert normalized.attrs["sample_name"] == "sample"


def test_build_pair_row_preserves_split_and_reports_improvement():
    raw = {"nrmse": 0.5, "mae": 0.4}
    denoised = {"nrmse": 0.4, "mae": 0.3}

    row = build_pair_row(
        pair_id="pair-1",
        split="val",
        input_record=_record("short", acquisition_time_s=2.0),
        reference_record=_record("long", acquisition_time_s=8.0),
        metrics={"raw": raw, "denoised": denoised},
    )

    assert row["pair_id"] == "pair-1"
    assert row["split"] == "val"
    assert row["raw_nrmse"] == 0.5
    assert row["denoised_nrmse"] == 0.4
    assert row["nrmse_improvement"] == pytest.approx(0.2)
