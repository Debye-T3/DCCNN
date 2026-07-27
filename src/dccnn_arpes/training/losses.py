"""Structure-preserving losses for ARPES 2D-cut denoising."""

from __future__ import annotations

import math

import torch
from torch import nn

try:
    from pytorch_msssim import ms_ssim
except ImportError as error:
    raise RuntimeError("CompositeDenoisingLoss requires the pytorch-msssim dependency.") from error


class CompositeDenoisingLoss(nn.Module):
    """Combine pixel, structural, and directional-gradient denoising losses."""

    epsilon = 1.0e-6

    def __init__(
        self,
        charbonnier: float = 0.80,
        ms_ssim: float = 0.15,
        gradient: float = 0.05,
    ) -> None:
        super().__init__()
        weights = (charbonnier, ms_ssim, gradient)
        if any(weight < 0 for weight in weights) or not math.isclose(
            sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-8
        ):
            raise ValueError("loss weights must be nonnegative and sum to one within 1e-8")
        self.charbonnier_weight = charbonnier
        self.ms_ssim_weight = ms_ssim
        self.gradient_weight = gradient

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted loss and detached, unweighted component metrics."""
        charbonnier = self._charbonnier(prediction, target)
        structure = 1 - ms_ssim(
            prediction.clamp(0.0, 1.0),
            target.clamp(0.0, 1.0),
            data_range=1.0,
            size_average=True,
        )
        gradient = self._gradient(prediction, target)
        total = (
            self.charbonnier_weight * charbonnier
            + self.ms_ssim_weight * structure
            + self.gradient_weight * gradient
        )
        components = {
            "charbonnier": charbonnier.detach(),
            "ms_ssim": structure.detach(),
            "gradient": gradient.detach(),
        }
        return total, components

    @classmethod
    def _charbonnier(cls, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((prediction - target).square() + cls.epsilon).mean()

    @classmethod
    def _gradient(cls, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        alpha_prediction = prediction[..., :, 1:] - prediction[..., :, :-1]
        alpha_target = target[..., :, 1:] - target[..., :, :-1]
        ev_prediction = prediction[..., 1:, :] - prediction[..., :-1, :]
        ev_target = target[..., 1:, :] - target[..., :-1, :]
        return (
            cls._charbonnier(alpha_prediction, alpha_target)
            + cls._charbonnier(ev_prediction, ev_target)
        ) / 2
