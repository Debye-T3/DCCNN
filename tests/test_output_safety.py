"""Central output-boundary tests for read-only ARPES inputs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from dccnn_arpes.safety import guard_output_path


def test_guard_rejects_read_only_and_input_descendants_without_creating_them(tmp_path):
    """An output boundary must reject protected destinations before mkdir or write."""
    read_only_root = tmp_path / "read-only"
    input_root = tmp_path / "input-root"
    read_only_root.mkdir()
    input_root.mkdir()

    with pytest.raises(ValueError, match="read-only data root"):
        guard_output_path(
            read_only_root / "derived" / "run",
            protected_roots=(read_only_root,),
        )
    with pytest.raises(ValueError, match="input source"):
        guard_output_path(
            input_root / "derived" / "cut.h5",
            protected_roots=(),
            input_sources=(input_root,),
        )

    assert not (read_only_root / "derived").exists()
    assert not (input_root / "derived").exists()


def test_guard_allows_an_independent_output_root_without_creating_it(tmp_path):
    """A normal outputs tree must remain available to training and inference."""
    destination = tmp_path / "outputs" / "run"

    resolved = guard_output_path(destination, protected_roots=())

    assert resolved == destination.resolve(strict=False)
    assert not destination.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_guard_resolves_junctions_before_checking_read_only_roots(tmp_path):
    """A junction alias must not redirect a seemingly safe output into protected data."""
    read_only_root = tmp_path / "read-only"
    aliases = tmp_path / "aliases"
    read_only_root.mkdir()
    aliases.mkdir()
    junction = aliases / "redirect"
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(read_only_root)],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"cannot create Windows junction: {completed.stderr.strip()}")

    with pytest.raises(ValueError, match="read-only data root"):
        guard_output_path(
            junction / "derived" / "cut.h5",
            protected_roots=(read_only_root,),
        )

    assert not (read_only_root / "derived").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows extended-length path behavior")
def test_guard_normalizes_windows_extended_length_paths_before_comparison(tmp_path):
    """The \\?\\ namespace must not provide an alias around a protected root."""
    read_only_root = tmp_path / "read-only"
    read_only_root.mkdir()
    extended_destination = Path(rf"\\?\{read_only_root}\derived\cut_denoised.h5")

    with pytest.raises(ValueError, match="read-only data root"):
        guard_output_path(
            extended_destination,
            protected_roots=(read_only_root,),
        )

    assert not (read_only_root / "derived").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows extended-length path behavior")
@pytest.mark.parametrize(
    "read_only_root",
    (Path(r"D:\Data\ARPES"), Path(r"D:\Projects\convert")),
)
def test_guard_rejects_extended_length_aliases_for_configured_read_only_roots(
    read_only_root,
):
    """Both fixed read-only roots must reject their extended-length aliases."""
    destination = Path(rf"\\?\{read_only_root}\final-fix-probe")

    with pytest.raises(ValueError, match="read-only data root"):
        guard_output_path(destination)


@pytest.mark.skipif(os.name != "nt", reason="Windows extended-length UNC path behavior")
def test_guard_normalizes_extended_length_unc_paths_before_comparison():
    """The \\?\\UNC namespace must compare as the equivalent ordinary UNC path."""
    protected_root = Path(r"\\server\share\read-only")
    destination = Path(r"\\?\UNC\server\share\read-only\derived\cut.h5")

    with pytest.raises(ValueError, match="read-only data root"):
        guard_output_path(destination, protected_roots=(protected_root,))
