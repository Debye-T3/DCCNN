"""Denoising command-line interface."""

import argparse


def main() -> None:
    """Run the denoising command-line interface."""
    parser = argparse.ArgumentParser(description="Denoise an ARPES 2D cut.")
    parser.parse_args()


if __name__ == "__main__":
    main()
