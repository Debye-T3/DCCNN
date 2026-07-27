"""Tests for read-only ARPES source discovery and converted-file association."""

import hashlib
import json
from pathlib import Path

import h5py

from dccnn_arpes.cli import data as data_cli
from dccnn_arpes.data.discovery import associate_converted, scan_archive


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


def test_scan_command_writes_manifest_and_read_only_evidence(tmp_path: Path):
    """Omitting scan audit artifacts or source hash comparison must break this contract."""
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "cut.pxt").write_text("source", encoding="utf-8")
    output = tmp_path / "workspace" / "manifests" / "records.csv"

    data_cli._scan_command(archive, tmp_path / "converted", output, None)

    assert output.exists()
    audit = json.loads((output.parent / "scan_audit.json").read_text(encoding="utf-8"))
    assert audit["source_file_count_before"] == audit["source_file_count_after"] == 1
    assert audit["sampled_source_hashes_before"] == audit["sampled_source_hashes_after"]
    assert (output.parent / "unknown_excel_columns.json").exists()
    assert (output.parent / "association_issues.json").exists()
