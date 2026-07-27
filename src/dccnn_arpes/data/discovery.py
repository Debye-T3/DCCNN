"""Read-only source discovery, converted-output association, and manifest writing."""

import csv
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import h5py

from .schema import MANIFEST_FIELDNAMES, ManifestRecord

_SOURCE_SUFFIXES = {".pxt", ".txt", ".bin", ".ibw", ".zip", ".xlsx"}
_CONVERTED_SUFFIXES = {".h5", ".hdf5", ".nc"}


def _normalised_path(path: Path | str) -> str:
    return os.path.normcase(os.path.normpath(str(Path(path).expanduser().resolve(strict=False))))


def _record_id(path: Path) -> str:
    material = f"{_normalised_path(path)}\0{path.stat().st_size}".encode()
    return hashlib.sha256(material).hexdigest()


def _normalised_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _decode_attribute(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _h5_attributes(path: Path) -> dict[str, str]:
    """Collect HDF5 attributes recursively; malformed outputs have no usable attributes."""
    attributes: dict[str, str] = {}
    try:
        with h5py.File(path, "r") as handle:
            def visit(_: str, item: h5py.Group | h5py.Dataset) -> None:
                for name, value in item.attrs.items():
                    attributes.setdefault(name.casefold(), _decode_attribute(value))

            for name, value in handle.attrs.items():
                attributes.setdefault(name.casefold(), _decode_attribute(value))
            handle.visititems(visit)
    except (OSError, ValueError):
        return {}
    return attributes


def _converted_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() in _CONVERTED_SUFFIXES),
        key=lambda path: _normalised_path(path),
    )


def scan_archive(root: Path) -> list[ManifestRecord]:
    """Recursively index supported source files without reading or changing their bytes."""
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    records: list[ManifestRecord] = []
    for path in sorted(root.rglob("*"), key=lambda item: _normalised_path(item)):
        if not path.is_file() or path.suffix.casefold() not in _SOURCE_SUFFIXES:
            continue
        sidecar = path.with_suffix(".ini") if path.suffix.casefold() == ".bin" else None
        records.append(
            ManifestRecord(
                record_id=_record_id(path),
                source_path=str(path.resolve()),
                source_format=path.suffix.casefold().lstrip("."),
                file_id=path.stem,
                pair_type="ini_sidecar" if sidecar and sidecar.is_file() else "",
                pair_id=str(sidecar.resolve()) if sidecar and sidecar.is_file() else "",
            )
        )
    return records


def _match_converted(record_list: Sequence[ManifestRecord], converted_path: Path) -> tuple[list[int], str]:
    attributes = _h5_attributes(converted_path)
    explicit_path = attributes.get("source_path")
    if explicit_path:
        matches = [
            index
            for index, record in enumerate(record_list)
            if _normalised_path(record.source_path) == _normalised_path(explicit_path)
        ]
        if matches:
            return matches, "explicit_source_path"

    converted_stem = _normalised_token(converted_path.stem)
    stem_matches = [
        index
        for index, record in enumerate(record_list)
        if _normalised_token(Path(record.source_path).stem) == converted_stem
    ]
    if stem_matches:
        return stem_matches, "exact_stem"

    candidate_tokens = set(re.findall(r"[a-z0-9]+", converted_path.stem.casefold()))
    file_id_attribute = attributes.get("file_id")
    if file_id_attribute:
        candidate_tokens.update(re.findall(r"[a-z0-9]+", file_id_attribute.casefold()))
    token_matches = [
        index
        for index, record in enumerate(record_list)
        if _normalised_token(record.file_id) in candidate_tokens and _normalised_token(record.file_id)
    ]
    return token_matches, "file_id_token"


def associate_converted_with_issues(
    records: Iterable[ManifestRecord], converted_root: Path
) -> tuple[list[ManifestRecord], list[dict[str, object]]]:
    """Associate converted data conservatively and describe every ambiguous candidate set."""
    associated = list(records)
    issues: list[dict[str, object]] = []
    for converted_path in _converted_paths(Path(converted_root)):
        candidates, stage = _match_converted(associated, converted_path)
        if len(candidates) == 1:
            index = candidates[0]
            if associated[index].converted_path:
                issues.append(
                    {
                        "converted_path": str(converted_path.resolve()),
                        "stage": "multiple_converted_outputs",
                        "candidate_source_paths": [associated[index].source_path],
                    }
                )
                associated[index] = replace(associated[index], review_status="needs_review")
                continue
            associated[index] = replace(associated[index], converted_path=str(converted_path.resolve()))
        elif len(candidates) > 1:
            issues.append(
                {
                    "converted_path": str(converted_path.resolve()),
                    "stage": stage,
                    "candidate_source_paths": [associated[index].source_path for index in candidates],
                }
            )
            for index in candidates:
                associated[index] = replace(associated[index], review_status="needs_review")
        else:
            issues.append(
                {
                    "converted_path": str(converted_path.resolve()),
                    "stage": stage,
                    "candidate_source_paths": [],
                }
            )
    return associated, issues


def associate_converted(records: Iterable[ManifestRecord], converted_root: Path) -> list[ManifestRecord]:
    """Associate outputs by explicit path, exact stem, then a unique file-ID token."""
    associated, _ = associate_converted_with_issues(records, converted_root)
    return associated


def write_manifest_csv(records: Iterable[ManifestRecord], path: Path) -> None:
    """Write one UTF-8 CSV header in immutable dataclass field order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow({key: "" if value is None else value for key, value in asdict(record).items()})


def source_snapshot(root: Path, *, sample_size: int = 25) -> dict[str, object]:
    """Count all source files and hash a deterministic small sample without mutating them."""
    paths = sorted(
        (path for path in Path(root).rglob("*") if path.is_file()), key=lambda path: _normalised_path(path)
    )
    samples: dict[str, str] = {}
    for path in paths[:sample_size]:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        samples[str(path.resolve())] = digest.hexdigest()
    return {"file_count": len(paths), "hashes": samples}


def build_scan_audit(
    records: Sequence[ManifestRecord],
    *,
    source_root: Path,
    converted_root: Path,
    source_before: dict[str, object],
    source_after: dict[str, object],
    workbook_count: int,
) -> dict[str, object]:
    """Build the evidence report for a read-only source scan."""
    converted_paths = _converted_paths(converted_root)
    associated_paths = {record.converted_path for record in records if record.converted_path}
    return {
        "source_root": str(Path(source_root).resolve()),
        "converted_root": str(Path(converted_root).resolve()),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "counts_by_source_format": dict(sorted(Counter(record.source_format for record in records).items())),
        "workbook_count": workbook_count,
        "converted_associated": len(associated_paths),
        "converted_unassociated": len(converted_paths) - len(associated_paths),
        "source_file_count_before": source_before["file_count"],
        "source_file_count_after": source_after["file_count"],
        "sampled_source_hashes_before": source_before["hashes"],
        "sampled_source_hashes_after": source_after["hashes"],
    }


def write_json(value: object, path: Path) -> None:
    """Write an UTF-8 JSON audit artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
