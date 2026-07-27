"""Reversible shared intensity preprocessing for ARPES cuts."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TransformStats:
    """Input-derived robust limits retained for inverse transformation."""

    lower: float
    scale: float


class IntensityTransform:
    """Apply negative clipping, ``log1p``, and shared robust normalization."""

    lower_quantile = 0.01
    upper_quantile = 0.995
    minimum_scale = 1.0e-6

    @staticmethod
    def _logged(array: np.ndarray) -> np.ndarray:
        values = np.asarray(array, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("intensity array must contain only finite values")
        return np.log1p(np.clip(values, 0.0, None))

    def fit(self, input_array: np.ndarray) -> TransformStats:
        """Fit robust limits from the input only."""
        logged = self._logged(input_array)
        if logged.size == 0:
            raise ValueError("intensity array must not be empty")
        lower, upper = np.quantile(logged, (self.lower_quantile, self.upper_quantile))
        return TransformStats(lower=float(lower), scale=max(float(upper - lower), self.minimum_scale))

    def forward(self, array: np.ndarray, stats: TransformStats) -> np.ndarray:
        """Normalize an array with previously fitted shared statistics."""
        if not np.isfinite(stats.lower) or not np.isfinite(stats.scale) or stats.scale <= 0:
            raise ValueError("transform statistics must have a finite positive scale")
        return (self._logged(array) - stats.lower) / stats.scale

    def inverse(self, array: np.ndarray, stats: TransformStats) -> np.ndarray:
        """Return normalized log intensities to nonnegative count space."""
        values = np.asarray(array, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("normalized array must contain only finite values")
        if not np.isfinite(stats.lower) or not np.isfinite(stats.scale) or stats.scale <= 0:
            raise ValueError("transform statistics must have a finite positive scale")
        return np.expm1(values * stats.scale + stats.lower)
