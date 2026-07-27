"""Calibrated dynamic noise synthesis for measured ARPES cuts."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True, slots=True)
class NoiseParameters:
    """Fixed parameters for one synthetic ARPES noise realization."""

    poisson_peak_counts: float = 500.0
    background_fraction: float = 0.04
    stripe_probability: float = 0.30
    stripe_fraction: float = 0.025

    def __post_init__(self) -> None:
        if self.poisson_peak_counts <= 0:
            raise ValueError("poisson_peak_counts must be positive")
        if self.background_fraction < 0:
            raise ValueError("background_fraction must be non-negative")
        if not 0 <= self.stripe_probability <= 1:
            raise ValueError("stripe_probability must be between zero and one")
        if self.stripe_fraction < 0:
            raise ValueError("stripe_fraction must be non-negative")


def _low_frequency_background(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    field = gaussian_filter(
        rng.standard_normal(shape),
        sigma=(max(shape[0] / 10.0, 1.0), max(shape[1] / 10.0, 1.0)),
        mode="reflect",
    )
    field -= field.min()
    maximum = field.max()
    return field / maximum if maximum > 0 else np.zeros(shape, dtype=np.float64)


def synthesize_noisy(
    clean: np.ndarray, params: NoiseParameters, rng: np.random.Generator
) -> np.ndarray:
    """Add Poisson, low-frequency background, and row/column stripe noise."""
    clean_values = np.asarray(clean, dtype=np.float64)
    if clean_values.ndim != 2 or clean_values.size == 0:
        raise ValueError("clean cut must be a non-empty two-dimensional array")
    if not np.isfinite(clean_values).all():
        raise ValueError("clean cut must contain only finite values")
    clean_values = np.clip(clean_values, 0.0, None)
    peak = float(clean_values.max())
    if peak == 0:
        peak = 1.0

    noisy = clean_values.copy()
    if np.isfinite(params.poisson_peak_counts):
        expected = clean_values * (params.poisson_peak_counts / peak)
        noisy = rng.poisson(expected).astype(np.float64) * (peak / params.poisson_peak_counts)

    if params.background_fraction:
        noisy += (
            _low_frequency_background(clean_values.shape, rng)
            * params.background_fraction
            * peak
        )

    if params.stripe_fraction and rng.random() < params.stripe_probability:
        if rng.random() < 0.5:
            offsets = rng.normal(0.0, params.stripe_fraction * peak, size=(clean_values.shape[0], 1))
        else:
            offsets = rng.normal(0.0, params.stripe_fraction * peak, size=(1, clean_values.shape[1]))
        noisy += offsets

    return np.clip(noisy, 0.0, None)
