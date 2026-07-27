"""Checkpoint-compatible implementation of the original CCNN model."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn


class LegacyCCNN(nn.Module):
    """The original CCNN architecture with its historical state-dict layout."""

    def __init__(self, kernel_size: int = 3, num_layers: int = 7) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(1, 64, kernel_size, padding=kernel_size // 2))
        self.layers.append(nn.PReLU())

        for _ in range(num_layers - 2):
            self.layers.append(nn.Conv2d(64, 64, kernel_size, padding=kernel_size // 2))
            self.layers.append(nn.PReLU())

        self.final = nn.Conv2d(64, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the legacy stack without changing the cut's spatial shape."""
        for layer in self.layers:
            inputs = layer(inputs)
        return self.final(inputs)


def load_legacy_checkpoint(model: LegacyCCNN, path: str | Path) -> None:
    """Load a raw or wrapped legacy checkpoint, reporting incompatible keys explicitly."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Legacy checkpoint must be a state-dict mapping or contain one.")

    state_dict = checkpoint
    for key in ("state_dict", "model_state_dict"):
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    if not isinstance(state_dict, Mapping):
        raise TypeError("Legacy checkpoint state dict must be a mapping.")

    normalized = {
        key.removeprefix("module.") if isinstance(key, str) else key: value
        for key, value in state_dict.items()
    }
    incompatible = model.load_state_dict(normalized, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        missing = ", ".join(incompatible.missing_keys) or "none"
        unexpected = ", ".join(incompatible.unexpected_keys) or "none"
        raise RuntimeError(
            f"Legacy checkpoint incompatible. Missing keys: {missing}. Unexpected keys: {unexpected}."
        )
