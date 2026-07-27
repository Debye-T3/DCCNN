"""Safety tests for the read-only legacy asset inventory command."""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inventory_legacy.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_inventory(repo: Path, archive: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            str(repo),
            "--archive",
            str(archive),
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )


def test_inventory_is_stable_groups_duplicates_and_never_mutates_sources(tmp_path: Path) -> None:
    """A missing hash, unstable traversal, or write to a legacy file must fail this test."""
    repo = tmp_path / "legacy-repository"
    archive = tmp_path / "proposed-archive"
    output = tmp_path / "manifests" / "legacy_inventory.csv"
    files = {
        "weights/model.pt": b"shared checkpoint bytes",
        "copies/model-copy.pth": b"shared checkpoint bytes",
        "converted/cut.h5": b"legacy hdf5 bytes",
        "reports/metrics.csv": b"loss,0.1\n",
        "previews/cut.png": b"png bytes",
        "results/run-001/summary.txt": b"result artifact",
        "config/train.yaml": b"training: {}\n",
    }
    for relative_path, contents in files.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    before = {
        path.relative_to(repo).as_posix(): (_sha256(path), path.stat().st_mtime_ns)
        for path in repo.rglob("*")
        if path.is_file()
    }

    _run_inventory(repo, archive, output)
    first_output = output.read_bytes()
    _run_inventory(repo, archive, output)

    assert output.read_bytes() == first_output
    assert not archive.exists()
    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert [row["path"] for row in rows] == sorted(row["path"] for row in rows)
    assert {
        "path",
        "type",
        "size_bytes",
        "modified_utc",
        "sha256",
        "duplicate_group",
        "proposed_destination",
    } <= set(rows[0])
    by_path = {row["path"]: row for row in rows}
    assert by_path["results/run-001"]["type"] == "result_directory"
    assert by_path["weights/model.pt"]["type"] == "checkpoint"
    assert by_path["converted/cut.h5"]["type"] == "h5"
    assert by_path["reports/metrics.csv"]["type"] == "csv"
    assert by_path["previews/cut.png"]["type"] == "png"
    assert by_path["config/train.yaml"]["type"] == "config"
    assert by_path["weights/model.pt"]["duplicate_group"] == by_path["copies/model-copy.pth"][
        "duplicate_group"
    ]
    assert by_path["weights/model.pt"]["duplicate_group"]
    assert by_path["converted/cut.h5"]["duplicate_group"] == ""
    for row in rows:
        assert Path(row["proposed_destination"]).resolve().is_relative_to(archive.resolve())

    after = {
        path.relative_to(repo).as_posix(): (_sha256(path), path.stat().st_mtime_ns)
        for path in repo.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    ("archive_name", "message"),
    [
        ("repo", "repository root"),
        ("source", "source root"),
        ("drive", "drive root"),
    ],
)
def test_inventory_rejects_unsafe_archive_roots(
    tmp_path: Path, archive_name: str, message: str
) -> None:
    """A future archiver must never be pointed at a root that can swallow its inputs."""
    repo = tmp_path / "repo"
    repo.mkdir()
    source_root = repo / "legacy-source"
    source_root.mkdir()
    source = source_root / "legacy.bin"
    source.write_bytes(b"input")
    output = tmp_path / "inventory.csv"
    archive = {
        "repo": repo,
        "source": source_root,
        "drive": Path(repo.anchor),
    }[archive_name]

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            str(repo),
            "--archive",
            str(archive),
            "--output",
            str(output),
            "--source",
            str(source_root),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert message in completed.stderr
    assert _sha256(source) == hashlib.sha256(b"input").hexdigest()
    assert not output.exists()
