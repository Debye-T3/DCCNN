"""Data-preparation command-line interface."""

import argparse


def main() -> None:
    """Run the data-preparation command-line interface."""
    parser = argparse.ArgumentParser(description="Prepare ARPES denoising data.")
    parser.parse_args()


if __name__ == "__main__":
    main()
