"""Denoising command-line interface."""

import argparse
from collections.abc import Sequence

from dccnn_arpes.inference import denoise_file


def main(argv: Sequence[str] | None = None) -> None:
    """Run the denoising command-line interface."""
    parser = argparse.ArgumentParser(description="Denoise an ARPES 2D cut.")
    parser.add_argument("--input", required=True, help="Canonical input HDF5 cut.")
    parser.add_argument("--checkpoint", required=True, help="Version 2 model checkpoint.")
    parser.add_argument("--output", required=True, help="Directory for the new denoised HDF5 file.")
    args = parser.parse_args(argv)

    output_path = denoise_file(args.input, args.checkpoint, args.output)
    print(f"output_path={output_path}", flush=True)


if __name__ == "__main__":
    main()
