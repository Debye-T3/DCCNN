"""Create a read-only inventory of legacy DCCNN assets.

The command intentionally reports proposed archive destinations only.  It never
creates the archive directory, moves files, or deletes files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

_CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".tar"}
_CONFIG_SUFFIXES = {".cfg", ".conf", ".ini", ".json", ".toml", ".yaml", ".yml"}
_H5_SUFFIXES = {".h5", ".hdf5"}
_FIELDNAMES = (
    "path",
    "type",
    "size_bytes",
    "modified_utc",
    "sha256",
    "duplicate_group",
    "proposed_destination",
)


@dataclass(frozen=True)
class InventoryEntry:
    """One legacy item and its proposed, but not performed, archive location."""

    path: str
    type: str
    size_bytes: int
    modified_utc: str
    sha256: str
    duplicate_group: str
    proposed_destination: str


def _resolved(path: Path, *, must_exist: bool) -> Path:
    """Expand and resolve a path without changing the filesystem."""
    candidate = path.expanduser()
    if must_exist and not candidate.exists():
        raise ValueError(f"path does not exist: {candidate}")
    return candidate.resolve(strict=must_exist)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_locations(repo: Path, source: Path, archive: Path, output: Path) -> tuple[Path, Path, Path, Path]:
    """Reject roots that could turn a later archive action into data loss."""
    repo_root = _resolved(repo, must_exist=True)
    source_root = _resolved(source, must_exist=True)
    archive_root = _resolved(archive, must_exist=False)
    output_path = _resolved(output, must_exist=False)
    if not repo_root.is_dir():
        raise ValueError(f"repository root must be a directory: {repo_root}")
    if not source_root.is_dir():
        raise ValueError(f"source root must be a directory: {source_root}")
    if not _is_within(source_root, repo_root):
        raise ValueError("source root must be inside the repository root")
    if archive_root == Path(archive_root.anchor):
        raise ValueError("archive root must not be a drive root")
    if archive_root == repo_root:
        raise ValueError("archive root must not equal the repository root")
    if archive_root == source_root:
        raise ValueError("archive root must not equal the source root")
    if _is_within(archive_root, source_root):
        raise ValueError("archive root must not be inside the source root")
    if archive_root.exists() and not archive_root.is_dir():
        raise ValueError("archive root must be a directory path")
    if _is_within(output_path, source_root):
        raise ValueError("inventory output must not be under the source root")
    return repo_root, source_root, archive_root, output_path


def _classify(path: Path) -> str:
    if path.is_dir():
        is_result_directory = any(part.casefold().startswith("result") for part in path.parts)
        return "result_directory" if is_result_directory else "directory"
    suffix = path.suffix.casefold()
    if suffix in _CHECKPOINT_SUFFIXES:
        return "checkpoint"
    if suffix in _H5_SUFFIXES:
        return "h5"
    if suffix == ".csv":
        return "csv"
    if suffix == ".png":
        return "png"
    if suffix in _CONFIG_SUFFIXES:
        return "config"
    return "other"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _modified_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _iter_legacy_paths(source_root: Path) -> Iterable[Path]:
    paths = sorted(source_root.rglob("*"), key=lambda path: path.relative_to(source_root).as_posix())
    for path in paths:
        if path.is_symlink():
            continue
        if path.is_file() or (path.is_dir() and _classify(path) == "result_directory"):
            yield path


def inventory_legacy(source_root: Path, archive_root: Path) -> list[InventoryEntry]:
    """Inspect legacy entries and return a stable report without writing to them."""
    entries: list[InventoryEntry] = []
    for path in _iter_legacy_paths(source_root):
        stat = path.stat()
        relative = path.relative_to(source_root).as_posix()
        checksum = _sha256(path) if path.is_file() else ""
        entries.append(
            InventoryEntry(
                path=relative,
                type=_classify(path),
                size_bytes=stat.st_size,
                modified_utc=_modified_utc(stat.st_mtime),
                sha256=checksum,
                duplicate_group="",
                proposed_destination=str((archive_root / relative).resolve(strict=False)),
            )
        )

    duplicate_counts = Counter(entry.sha256 for entry in entries if entry.sha256)
    duplicate_hashes = sorted(checksum for checksum, count in duplicate_counts.items() if count > 1)
    duplicate_groups = {checksum: f"duplicate-{index:04d}" for index, checksum in enumerate(duplicate_hashes, 1)}
    return [
        replace(entry, duplicate_group=duplicate_groups.get(entry.sha256, "")) for entry in entries
    ]


def write_inventory(entries: Sequence[InventoryEntry], output: Path) -> None:
    """Atomically write the report; this is the command's only filesystem mutation."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output.parent, delete=False
        ) as stream:
            temporary_name = stream.name
            writer = csv.DictWriter(stream, fieldnames=_FIELDNAMES, lineterminator="\n")
            writer.writeheader()
            for entry in entries:
                writer.writerow(
                    {
                        "path": entry.path,
                        "type": entry.type,
                        "size_bytes": entry.size_bytes,
                        "modified_utc": entry.modified_utc,
                        "sha256": entry.sha256,
                        "duplicate_group": entry.duplicate_group,
                        "proposed_destination": entry.proposed_destination,
                    }
                )
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the reporting-only legacy inventory command."""
    parser = argparse.ArgumentParser(description="Create a read-only legacy asset inventory.")
    parser.add_argument("--repo", required=True, type=Path, help="Repository root containing legacy assets.")
    parser.add_argument(
        "--source", type=Path, help="Optional legacy source subtree; defaults to --repo."
    )
    parser.add_argument("--archive", required=True, type=Path, help="Proposed archive root; never created.")
    parser.add_argument("--output", required=True, type=Path, help="CSV report written outside source data.")
    arguments = parser.parse_args(argv)
    source = arguments.source or arguments.repo
    try:
        _repo_root, source_root, archive_root, output = validate_locations(
            arguments.repo, source, arguments.archive, arguments.output
        )
        entries = inventory_legacy(source_root, archive_root)
        write_inventory(entries, output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"inventory: {output}")
    print(f"entries: {len(entries)}")
    print("mode: report-only (no legacy files were moved or deleted)")


if __name__ == "__main__":
    main()
