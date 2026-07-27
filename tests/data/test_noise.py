"""Tests for calibrated reproducible ARPES noise synthesis."""

import numpy as np

from dccnn_arpes.data.noise import NoiseParameters, synthesize_noisy


def test_poisson_background_and_stripes_are_seed_reproducible():
    """Ignoring the supplied generator must make repeated synthesis differ."""
    clean = np.linspace(0.2, 10.0, 48 * 64, dtype=np.float64).reshape(48, 64)
    params = NoiseParameters(
        poisson_peak_counts=80.0,
        background_fraction=0.08,
        stripe_probability=1.0,
        stripe_fraction=0.05,
    )

    first = synthesize_noisy(clean, params, np.random.default_rng(314))
    second = synthesize_noisy(clean, params, np.random.default_rng(314))
    changed = synthesize_noisy(clean, params, np.random.default_rng(315))

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, changed)
    assert first.shape == clean.shape
    assert np.isfinite(first).all()
    assert (first >= 0).all()


def test_each_noise_component_changes_a_nonconstant_clean_cut():
    """Silently omitting any calibrated component must fail its isolated case."""
    clean = np.linspace(1.0, 9.0, 32 * 40, dtype=np.float64).reshape(32, 40)
    cases = (
        NoiseParameters(
            poisson_peak_counts=30.0,
            background_fraction=0.0,
            stripe_probability=0.0,
            stripe_fraction=0.0,
        ),
        NoiseParameters(
            poisson_peak_counts=np.inf,
            background_fraction=0.1,
            stripe_probability=0.0,
            stripe_fraction=0.0,
        ),
        NoiseParameters(
            poisson_peak_counts=np.inf,
            background_fraction=0.0,
            stripe_probability=1.0,
            stripe_fraction=0.1,
        ),
    )

    for index, params in enumerate(cases):
        noisy = synthesize_noisy(clean, params, np.random.default_rng(index))
        assert not np.array_equal(noisy, clean)
