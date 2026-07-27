"""Residual noise-prediction architecture and common denoising adapter."""

from __future__ import annotations

from torch import Tensor, nn


class _ResidualBlock(nn.Module):
    """A local residual correction made from two same-resolution convolutions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.conv2(self.activation(self.conv1(inputs)))


class ResidualDenoiser2D(nn.Module):
    """Predict additive noise from a 2D cut and subtract it from the input."""

    def __init__(self, channels: int = 64, blocks: int = 8) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(_ResidualBlock(channels) for _ in range(blocks))
        self.noise_output = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.noise_output.weight)
        nn.init.zeros_(self.noise_output.bias)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Return the denoised cut followed by the predicted additive noise."""
        features = self.input_conv(inputs)
        for block in self.blocks:
            features = block(features)
        predicted_noise = self.noise_output(features)
        return inputs - predicted_noise, predicted_noise


def denoise_forward(model: nn.Module, input_tensor: Tensor) -> tuple[Tensor, Tensor | None]:
    """Normalize legacy and residual model predictions to a common tuple contract."""
    prediction = model(input_tensor)
    if isinstance(prediction, tuple):
        return prediction
    return prediction, None
