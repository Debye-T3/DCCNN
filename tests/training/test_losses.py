"""Tests for the structure-preserving ARPES denoising loss."""

from __future__ import annotations

import importlib
import sys

import pytest
import torch
from pytorch_msssim import ms_ssim


def _loss_type():
    from dccnn_arpes.training.losses import CompositeDenoisingLoss

    return CompositeDenoisingLoss


def _gradient_charbonnier(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate the specified first-difference loss independently."""
    alpha_error = torch.sqrt(
        (
            prediction[..., :, 1:]
            - prediction[..., :, :-1]
            - (target[..., :, 1:] - target[..., :, :-1])
        )
        ** 2
        + 1.0e-6
    ).mean()
    ev_error = torch.sqrt(
        (
            prediction[..., 1:, :]
            - prediction[..., :-1, :]
            - (target[..., 1:, :] - target[..., :-1, :])
        )
        ** 2
        + 1.0e-6
    ).mean()
    return (alpha_error + ev_error) / 2


def test_loss_reports_unweighted_components_and_weighted_total():
    """Changing a component's weight must change the observable total loss."""
    torch.manual_seed(7)
    target = torch.randn(2, 1, 256, 256)
    prediction = target + 0.15 * torch.randn_like(target)

    total, parts = _loss_type()()(prediction, target)

    expected = 0.80 * parts["charbonnier"] + 0.15 * parts["ms_ssim"] + 0.05 * parts["gradient"]
    assert torch.isfinite(total)
    assert set(parts) == {"charbonnier", "ms_ssim", "gradient"}
    assert all(not component.requires_grad for component in parts.values())
    torch.testing.assert_close(total.detach(), expected)


def test_loss_uses_unclipped_charbonnier_and_gradient_components():
    """Clamping raw tensors would hide valid normalized values outside [0, 1]."""
    target = torch.linspace(-0.4, 1.4, 256 * 256).reshape(1, 1, 256, 256)
    prediction = target + 0.1 * torch.sin(torch.linspace(0, 12, 256 * 256)).reshape_as(target)

    _, parts = _loss_type()()(prediction, target)

    expected_charbonnier = torch.sqrt((prediction - target) ** 2 + 1.0e-6).mean()
    expected_ms_ssim = 1 - ms_ssim(
        prediction.clamp(0, 1), target.clamp(0, 1), data_range=1.0, size_average=True
    )
    expected_gradient = _gradient_charbonnier(prediction, target)
    torch.testing.assert_close(parts["charbonnier"], expected_charbonnier)
    torch.testing.assert_close(parts["ms_ssim"], expected_ms_ssim)
    torch.testing.assert_close(parts["gradient"], expected_gradient)


def test_loss_backpropagates_finite_gradients():
    """A nonfinite composite loss would prevent denoiser training."""
    torch.manual_seed(11)
    target = torch.randn(2, 1, 256, 256)
    prediction = torch.randn(2, 1, 256, 256, requires_grad=True)

    total, _ = _loss_type()()(prediction, target)
    total.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_identical_tensors_have_lower_loss_than_shifted_tensors():
    """A uniform intensity shift must be penalized even when gradients match."""
    torch.manual_seed(13)
    target = torch.randn(2, 1, 256, 256)
    loss = _loss_type()()

    identical, _ = loss(target, target)
    shifted, _ = loss(target + 0.25, target)

    assert identical < shifted


@pytest.mark.parametrize(
    ("weights",),
    [((-0.01, 0.96, 0.05),), ((0.80, 0.15, 0.04),)],
)
def test_loss_rejects_invalid_component_weights(weights: tuple[float, float, float]):
    """Invalid weights would make the configured loss physically meaningless."""
    with pytest.raises(ValueError):
        _loss_type()(*weights)


def test_missing_ms_ssim_dependency_raises_explicit_runtime_error(monkeypatch: pytest.MonkeyPatch):
    """A silent approximation would change the configured experiment objective."""
    with monkeypatch.context() as import_patch:
        import_patch.setitem(sys.modules, "pytorch_msssim", None)
        import_patch.delitem(sys.modules, "dccnn_arpes.training.losses", raising=False)

        with pytest.raises(RuntimeError, match="pytorch-msssim"):
            importlib.import_module("dccnn_arpes.training.losses")

    importlib.import_module("dccnn_arpes.training.losses")
