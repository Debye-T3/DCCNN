"""Training command-line interface."""

import argparse
from collections.abc import Sequence
from dataclasses import replace

import torch

from dccnn_arpes.training.config import load_train_config
from dccnn_arpes.training.trainer import run_training


def main(argv: Sequence[str] | None = None) -> None:
    """Run the training command-line interface."""
    parser = argparse.ArgumentParser(description="Train an ARPES denoising model.")
    parser.add_argument("--config", required=True, help="Strict training YAML.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use bounded in-memory smoke settings without modifying YAML.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Explicit device override.")
    args = parser.parse_args(argv)

    config = load_train_config(args.config)
    if args.smoke_test:
        config = config.for_smoke_test(device=args.device)
    elif args.device is not None:
        config = replace(
            config,
            training=replace(
                config.training,
                device=args.device,
                amp=config.training.amp and args.device == "cuda",
            ),
        )

    print(f"requested_device={config.training.device}", flush=True)
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)
    print(f"torch.version.cuda={torch.version.cuda}", flush=True)
    if config.training.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        print(f"cuda_device_name={torch.cuda.get_device_name(0)}", flush=True)
    result = run_training(config)
    print(f"output_dir={result.output_dir}", flush=True)


if __name__ == "__main__":
    main()
