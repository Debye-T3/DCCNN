"""Training command-line interface."""

import argparse


def main() -> None:
    """Run the training command-line interface."""
    parser = argparse.ArgumentParser(description="Train an ARPES denoising model.")
    parser.parse_args()


if __name__ == "__main__":
    main()
