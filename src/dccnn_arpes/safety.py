"""Shared filesystem guards for outputs derived from read-only ARPES data."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

READ_ONLY_DATA_ROOTS = (
    Path(r"D:\Data\ARPES"),
    Path(r"D:\Projects\convert"),
)


def _without_windows_device_prefix(value: str) -> str:
    lowered = value.casefold()
    unc_prefix = "\\\\?\\unc\\"
    device_prefix = "\\\\?\\"
    if lowered.startswith(unc_prefix):
        return "\\\\" + value[len(unc_prefix) :]
    if lowered.startswith(device_prefix):
        return value[len(device_prefix) :]
    return value


def _resolved(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    return Path(_without_windows_device_prefix(str(resolved)))


def _is_at_or_below(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def guard_output_path(
    destination: str | Path,
    *,
    protected_roots: Iterable[str | Path] | None = None,
    input_sources: Iterable[str | Path] = (),
) -> Path:
    """Resolve redirects and reject output paths inside any read-only input tree."""
    resolved_destination = _resolved(destination)
    roots = READ_ONLY_DATA_ROOTS if protected_roots is None else tuple(protected_roots)
    for root in roots:
        resolved_root = _resolved(root)
        if _is_at_or_below(resolved_destination, resolved_root):
            raise ValueError(
                f"output path must not be inside read-only data root {resolved_root}: "
                f"{resolved_destination}"
            )
    for source in input_sources:
        resolved_source = _resolved(source)
        if _is_at_or_below(resolved_destination, resolved_source):
            raise ValueError(
                f"output path must not be the input source or its descendant "
                f"{resolved_source}: {resolved_destination}"
            )
    return resolved_destination
