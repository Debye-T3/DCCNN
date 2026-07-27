"""Data-preparation command-line interface."""

import argparse
from pathlib import Path

from dccnn_arpes.io import load_cut


def _format_range(values) -> str:
    """Format the inclusive range of one validated coordinate."""
    return f"{values.min():g} to {values.max():g}"


def _validate_command(path: Path, allow_legacy: bool) -> None:
    """Load a cut and print the properties needed to verify its data boundary."""
    cut = load_cut(path, allow_legacy=allow_legacy)
    source = "legacy-adapted" if "legacy_source" in cut.attrs else "standard"
    print(f"path: {path}")
    print(f"object name: {cut.name}")
    print(f"shape: {cut.shape}")
    print(f"dimensions: {', '.join(cut.dims)}")
    print(f"eV range: {_format_range(cut.coords['eV'].values)}")
    print(f"alpha range: {_format_range(cut.coords['alpha'].values)}")
    print(f"format: {source}")


def main() -> None:
    """Run the data-preparation command-line interface."""
    parser = argparse.ArgumentParser(description="Prepare ARPES denoising data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate", help="validate a 2D ARPES cut")
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument("--allow-legacy", action="store_true")
    arguments = parser.parse_args()

    if arguments.command == "validate":
        try:
            _validate_command(arguments.path, arguments.allow_legacy)
        except (OSError, ValueError, TypeError) as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
