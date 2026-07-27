"""Tests for conservative experiment-workbook candidate extraction."""

from pathlib import Path

import pandas as pd

from dccnn_arpes.data.metadata import read_workbook_candidates


def test_read_workbook_candidates_marks_forward_filled_values_for_review(tmp_path: Path):
    """Dropping the inheritance review flag must make this test fail."""
    workbook = tmp_path / "experiment.xlsx"
    pd.DataFrame(
        {
            "File ID": ["scan-001", "scan-002"],
            "Sample": ["FeSe", None],
            "Temperature (K)": [20.0, None],
            "Photon Energy (eV)": [21.2, 21.2],
            "Polarization": ["LH", "LV"],
            "Acquisition Time (s)": [4.0, 5.0],
            "Sweeps": [2, 3],
            "Operator note": ["first", "second"],
        }
    ).to_excel(workbook, index=False)

    rows = read_workbook_candidates(
        workbook,
        {
            "file_id": ["file id"],
            "sample_name": ["sample"],
            "temperature_K": ["temperature (k)"],
            "photon_energy_eV": ["photon energy (ev)"],
            "polarization": ["polarization"],
            "acquisition_time_s": ["acquisition time (s)"],
            "sweep_count": ["sweeps"],
        },
    )

    assert rows.loc[1, "sample_name"] == "FeSe"
    assert rows.loc[1, "temperature_K"] == 20.0
    assert rows.loc[1, "metadata_inherited"] is True
    assert rows.loc[1, "review_status"] == "needs_review"
    assert rows.attrs["unknown_columns"] == ["Operator note"]
