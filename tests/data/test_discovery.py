"""Tests for read-only ARPES source discovery and converted-file association."""

import csv
import hashlib
import json
from pathlib import Path

import h5py
import pytest

from dccnn_arpes.cli import data as data_cli
from dccnn_arpes.data.discovery import associate_converted, scan_archive, write_manifest_csv
from dccnn_arpes.data.schema import ManifestRecord


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scan_archive_indexes_supported_sources_without_changing_them(tmp_path: Path):
    """Removing a supported suffix or touching an input must break this contract."""
    archive = tmp_path / "archive"
    archive.mkdir()
    expected_sources = [
        archive / "cut.pxt",
        archive / "nested" / "spectrum.txt",
        archive / "nested" / "binary.bin",
        archive / "nested" / "wave.ibw",
        archive / "bundle.zip",
        archive / "log.xlsx",
    ]
    for source in expected_sources:
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"contents:{source.name}".encode())
    sidecar = archive / "nested" / "binary.ini"
    sidecar.write_text("detector=1", encoding="utf-8")
    ignored = archive / "preview.png"
    ignored.write_bytes(b"not source data")

    before = {path: (path.stat().st_mtime_ns, _sha256(path)) for path in archive.rglob("*") if path.is_file()}
    records = scan_archive(archive)
    after = {path: (path.stat().st_mtime_ns, _sha256(path)) for path in archive.rglob("*") if path.is_file()}

    assert {record.source_path for record in records} == {str(path.resolve()) for path in expected_sources}
    assert all(Path(record.source_path).is_absolute() for record in records)
    assert all(record.record_id for record in records)
    assert before == after
    binary_record = next(record for record in records if record.source_path.endswith("binary.bin"))
    assert binary_record.pair_type == "ini_sidecar"
    assert binary_record.pair_id == str(sidecar.resolve())


def test_associate_converted_uses_explicit_path_before_filename_fallback(tmp_path: Path):
    """Changing precedence to choose a same-stem file must break this contract."""
    archive = tmp_path / "archive"
    (archive / "left").mkdir(parents=True)
    (archive / "right").mkdir()
    explicit_source = archive / "left" / "cut.pxt"
    other_source = archive / "right" / "cut.pxt"
    explicit_source.write_text("left", encoding="utf-8")
    other_source.write_text("right", encoding="utf-8")
    converted = tmp_path / "converted"
    converted.mkdir()
    with h5py.File(converted / "cut.h5", "w") as output:
        output.attrs["source_path"] = str(explicit_source.resolve())

    associated = associate_converted(scan_archive(archive), converted)

    record_by_source = {record.source_path: record for record in associated}
    assert record_by_source[str(explicit_source.resolve())].converted_path == str(
        (converted / "cut.h5").resolve()
    )
    assert record_by_source[str(other_source.resolve())].converted_path == ""


def test_scan_command_writes_manifest_and_read_only_evidence(tmp_path: Path, monkeypatch):
    """Omitting scan audit artifacts or source hash comparison must break this contract."""
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "cut.pxt").write_text("source", encoding="utf-8")
    output = tmp_path / "workspace" / "manifests" / "records.csv"
    monkeypatch.setattr(data_cli, "_WORKSPACE_ROOT", output.parents[1])

    data_cli._scan_command(archive, tmp_path / "converted", output, None)

    assert output.exists()
    audit = json.loads((output.parent / "scan_audit.json").read_text(encoding="utf-8"))
    assert audit["source_file_count_before"] == audit["source_file_count_after"] == 1
    assert audit["sampled_source_hashes_before"] == audit["sampled_source_hashes_after"]
    assert (output.parent / "unknown_excel_columns.json").exists()
    assert (output.parent / "association_issues.json").exists()


def test_scan_command_rejects_output_inside_source_tree_without_changing_source(
    tmp_path: Path, monkeypatch
):
    """Removing output containment validation would write into the source archive."""
    workspace = tmp_path / "workspace"
    archive = workspace / "archive"
    archive.mkdir(parents=True)
    source = archive / "cut.pxt"
    source.write_text("source", encoding="utf-8")
    monkeypatch.setattr(data_cli, "_WORKSPACE_ROOT", workspace)
    before = _sha256(source)
    output = archive / "records.csv"

    with pytest.raises(ValueError, match="source root"):
        data_cli._scan_command(archive, tmp_path / "converted", output, None)

    assert _sha256(source) == before
    assert not output.exists()


def test_scan_command_rejects_output_outside_workspace(tmp_path: Path, monkeypatch):
    """Removing the workspace boundary would permit arbitrary manifest writes."""
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "cut.pxt").write_text("source", encoding="utf-8")
    monkeypatch.setattr(data_cli, "_WORKSPACE_ROOT", tmp_path / "workspace")
    output = tmp_path / "outside" / "records.csv"

    with pytest.raises(ValueError, match="workspace"):
        data_cli._scan_command(archive, tmp_path / "converted", output, None)

    assert not output.exists()


def test_file_id_token_association_normalizes_separator_containing_identifiers(tmp_path: Path):
    """Tokenizing only individual filename pieces would lose a unique run-001 association."""
    converted = tmp_path / "converted"
    converted.mkdir()
    converted_path = converted / "processed-run_001-output.h5"
    with h5py.File(converted_path, "w"):
        pass
    record = ManifestRecord(source_path=str((tmp_path / "raw.pxt").resolve()), file_id="run-001")

    associated = associate_converted([record], converted)

    assert associated[0].converted_path == str(converted_path.resolve())


def test_file_id_token_association_leaves_normalized_identifier_collisions_unassociated(tmp_path: Path):
    """Choosing the first normalized run-001 match would silently corrupt provenance."""
    converted = tmp_path / "converted"
    converted.mkdir()
    with h5py.File(converted / "processed-run_001-output.h5", "w"):
        pass
    records = [
        ManifestRecord(source_path=str((tmp_path / "first.pxt").resolve()), file_id="run-001"),
        ManifestRecord(source_path=str((tmp_path / "second.pxt").resolve()), file_id="run_001"),
    ]

    associated = associate_converted(records, converted)

    assert [record.converted_path for record in associated] == ["", ""]
    assert [record.review_status for record in associated] == ["needs_review", "needs_review"]


def test_manifest_csv_serializes_axes_as_compact_json_arrays(tmp_path: Path):
    """Writing tuples or empty strings would break the CSV axis contract."""
    output = tmp_path / "records.csv"
    write_manifest_csv(
        [
            ManifestRecord(record_id="empty"),
            ManifestRecord(record_id="populated", energy_axis=(1.0, 2.5), angle_axis=(-3.0, 0.0)),
        ],
        output,
    )

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["energy_axis"] == "[]"
    assert rows[0]["angle_axis"] == "[]"
    assert rows[1]["energy_axis"] == "[1.0,2.5]"
    assert rows[1]["angle_axis"] == "[-3.0,0.0]"


def test_manifest_record_copies_axis_lists_to_immutable_tuples():
    """Retaining caller-owned axis lists would violate the immutable manifest contract."""
    energy_axis = [1.0, 2.5]
    angle_axis = [-3.0, 0.0]
    record = ManifestRecord(energy_axis=energy_axis, angle_axis=angle_axis)
    energy_axis.append(4.0)
    angle_axis.clear()

    assert record.energy_axis == (1.0, 2.5)
    assert record.angle_axis == (-3.0, 0.0)
    assert isinstance(record.energy_axis, tuple)
    assert isinstance(record.angle_axis, tuple)
    with pytest.raises(AttributeError):
        record.energy_axis.append(4.0)
