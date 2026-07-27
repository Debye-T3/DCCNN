"""Neural-network architectures used by the ARPES denoising workflows."""

from dccnn_arpes.models.legacy_ccnn import LegacyCCNN, load_legacy_checkpoint
from dccnn_arpes.models.residual import ResidualDenoiser2D, denoise_forward

__all__ = [
    "LegacyCCNN",
    "ResidualDenoiser2D",
    "denoise_forward",
    "load_legacy_checkpoint",
]
