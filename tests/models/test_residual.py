"""Tests for the residual ARPES 2D-cut denoiser."""

from __future__ import annotations

import torch
from torch import nn

from dccnn_arpes.models.legacy_ccnn import LegacyCCNN
from dccnn_arpes.models.residual import ResidualDenoiser2D, denoise_forward


def test_residual_denoiser_preserves_odd_spatial_shape_without_unsupported_layers():
    """A spatially destructive layer must not alter an odd-size ARPES cut."""
    model = ResidualDenoiser2D(channels=64, blocks=8)
    inputs = torch.randn(2, 1, 17, 19)

    denoised, predicted_noise = model(inputs)

    assert denoised.shape == inputs.shape
    assert predicted_noise.shape == inputs.shape
    torch.testing.assert_close(denoised, inputs - predicted_noise)
    forbidden_types = (nn.modules.batchnorm._BatchNorm, nn.ConvTranspose2d, nn.MaxPool2d, nn.AvgPool2d)
    assert not any(isinstance(module, forbidden_types) for module in model.modules())


def test_residual_denoiser_starts_as_identity():
    """A nonzero initial noise head would alter inputs before training."""
    model = ResidualDenoiser2D(channels=64, blocks=8)
    inputs = torch.randn(1, 1, 15, 21)

    denoised, predicted_noise = model(inputs)

    torch.testing.assert_close(predicted_noise, torch.zeros_like(inputs))
    torch.testing.assert_close(denoised, inputs)


def test_denoise_forward_normalizes_legacy_and_residual_predictions():
    """Returning a model-specific prediction shape would force caller branches."""
    inputs = torch.randn(1, 1, 11, 13)
    legacy = LegacyCCNN(kernel_size=3, num_layers=7)
    residual = ResidualDenoiser2D(channels=64, blocks=8)

    legacy_denoised, legacy_noise = denoise_forward(legacy, inputs)
    residual_denoised, residual_noise = denoise_forward(residual, inputs)

    torch.testing.assert_close(legacy_denoised, legacy(inputs))
    assert legacy_noise is None
    torch.testing.assert_close(residual_denoised, inputs - residual_noise)
