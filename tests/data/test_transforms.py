"""Numerical tests for reversible, shared ARPES intensity preprocessing."""

import numpy as np

from dccnn_arpes.data.transforms import IntensityTransform


def test_forward_inverse_round_trip_recovers_nonnegative_counts():
    """Dropping log inversion or robust scaling must break count recovery."""
    values = np.geomspace(1.0e-3, 1.0e5, 4096).reshape(64, 64)
    transform = IntensityTransform()

    stats = transform.fit(values)
    recovered = transform.inverse(transform.forward(values, stats), stats)

    np.testing.assert_allclose(recovered, values, rtol=1.0e-5, atol=1.0e-7)


def test_negative_count_artifacts_are_the_only_values_clipped():
    """Removing negative clipping or upper-clipping bright pixels must fail."""
    values = np.array([[-4.0, 0.0, 2.0, 1.0e8]], dtype=np.float64)
    transform = IntensityTransform()

    stats = transform.fit(values)
    recovered = transform.inverse(transform.forward(values, stats), stats)

    np.testing.assert_allclose(recovered, [[0.0, 0.0, 2.0, 1.0e8]], rtol=1.0e-5)


def test_input_fitted_statistics_are_shared_with_target():
    """Fitting target statistics independently must change this target encoding."""
    input_values = np.arange(1, 101, dtype=np.float64).reshape(10, 10)
    target_values = input_values * 40.0 + 7.0
    transform = IntensityTransform()

    input_stats = transform.fit(input_values)
    target_encoded = transform.forward(target_values, input_stats)
    target_recovered = transform.inverse(target_encoded, input_stats)

    assert input_stats != transform.fit(target_values)
    np.testing.assert_allclose(target_recovered, target_values, rtol=1.0e-5)


def test_constant_input_uses_minimum_nonzero_scale():
    """Allowing a zero scale must produce non-finite normalized values."""
    transform = IntensityTransform()

    stats = transform.fit(np.full((4, 5), 3.0))

    assert stats.scale == 1.0e-6
    assert np.isfinite(transform.forward(np.full((4, 5), 3.0), stats)).all()
