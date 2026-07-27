"""Evaluation command-line interface."""

import argparse


def main() -> None:
    """Run the evaluation command-line interface."""
    parser = argparse.ArgumentParser(description="Evaluate an ARPES denoising model.")
    parser.parse_args()


if __name__ == "__main__":
    main()
