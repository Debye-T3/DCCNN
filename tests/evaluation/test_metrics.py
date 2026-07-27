"""Analytical tests for coordinate-aware ARPES evaluation metrics."""

import numpy as np
import pytest
import xarray as xr

from dccnn_arpes.evaluation import (
    evaluate_pair,
    gaussian_baseline,
    median_baseline,
)


def _gaussian_cut(
    *,
    center_eV: float = 0.17,
    center_alpha: float = -0.28,
    sigma_eV: float = 0.12,
    sigma_alpha: float = 0.31,
) -> xr.DataArray:
    eV = np.linspace(-1.0, 1.0, 201)
    alpha = np.linspace(-2.0, 2.0, 161)
    energy_peak = np.exp(-0.5 * ((eV - center_eV) / sigma_eV) ** 2)
    angle_peak = np.exp(-0.5 * ((alpha - center_alpha) / sigma_alpha) ** 2)
    values = 0.08 + 4.2 * energy_peak[:, None] * angle_peak[None, :]
    return xr.DataArray(
        values,
        dims=("eV", "alpha"),
        coords={"eV": eV, "alpha": alpha, "photon_energy_eV": 21.2},
        name="intensity",
        attrs={"sample_id": "analytic", "acquisition_time_s": 2.0},
    )


def test_identical_arrays_have_exact_zero_errors_and_unit_ssim():
    """Changing an identity comparison into a nonzero error must fail this test."""
    cut = _gaussian_cut()

    metrics = evaluate_pair(cut, cut.copy(deep=True), cut.copy(deep=True))

    assert metrics["mae"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["nrmse"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["ssim"] == pytest.approx(1.0, abs=1e-14)
    assert metrics["integrated_intensity_relative_error"] == pytest.approx(0.0, abs=1e-14)


def test_one_bin_energy_shift_is_reported_in_physical_coordinate_units():
    """Returning an index displacement instead of an eV displacement must fail."""
    reference = _gaussian_cut(center_eV=0.17)
    eV_step = float(abs(reference.eV.values[1] - reference.eV.values[0]))
    shifted = _gaussian_cut(center_eV=0.17 + eV_step)

    metrics = evaluate_pair(reference, shifted, reference)

    assert metrics["peak_position_error_eV"] == pytest.approx(eV_step, rel=0.02)
    assert metrics["peak_position_error"] == pytest.approx(eV_step, rel=0.02)


def test_gaussian_fwhm_fit_matches_analytic_width_within_two_percent():
    """Using pixels or a half-width in place of physical FWHM must fail."""
    sigma_eV = 0.12
    sigma_alpha = 0.31
    cut = _gaussian_cut(sigma_eV=sigma_eV, sigma_alpha=sigma_alpha)
    expected_eV = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_eV
    expected_alpha = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_alpha

    metrics = evaluate_pair(cut, cut, cut)

    assert metrics["fit_status"] == "ok"
    assert metrics["reference_fwhm_eV"] == pytest.approx(expected_eV, rel=0.02)
    assert metrics["reference_fwhm_alpha"] == pytest.approx(expected_alpha, rel=0.02)
    assert metrics["fwhm_relative_error"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("mismatched_argument", ["input", "output", "reference"])
def test_every_coordinate_mismatch_raises_before_metric_calculation(mismatched_argument):
    """Silently aligning any mismatched physical grid must fail this test."""
    input_cut = _gaussian_cut()
    output_cut = input_cut.copy(deep=True)
    reference_cut = input_cut.copy(deep=True)
    changed = input_cut.assign_coords(
        eV=input_cut.eV.values + 0.25 * float(input_cut.eV.values[1] - input_cut.eV.values[0])
    )
    arrays = {
        "input": input_cut,
        "output": output_cut,
        "reference": reference_cut,
    }
    arrays[mismatched_argument] = changed

    with pytest.raises(ValueError, match="coordinates must match exactly"):
        evaluate_pair(arrays["input"], arrays["output"], arrays["reference"])


@pytest.mark.parametrize(
    ("baseline", "kwargs"),
    [
        (gaussian_baseline, {"sigma": 1.25}),
        (median_baseline, {"size": 3}),
    ],
)
def test_traditional_baselines_preserve_every_coordinate_name_value_and_attribute(baseline, kwargs):
    """Dropping scalar coordinates or source attributes must fail this test."""
    cut = _gaussian_cut()

    result = baseline(cut, **kwargs)

    assert result.dims == cut.dims
    assert result.name == cut.name
    assert result.attrs == cut.attrs
    assert set(result.coords) == set(cut.coords)
    for coordinate in cut.coords:
        xr.testing.assert_identical(result.coords[coordinate], cut.coords[coordinate])


def test_invalid_flat_peak_keeps_named_failure_status_and_nan_width():
    """Dropping or fabricating a width for a failed fit must fail this test."""
    cut = _gaussian_cut()
    flat = xr.full_like(cut, 3.0)

    metrics = evaluate_pair(flat, flat, flat)

    assert metrics["fit_status"] == "failed"
    assert isinstance(metrics["fit_failure_reason"], str)
    assert np.isnan(metrics["fwhm_relative_error"])


def test_noise_only_region_reduction_uses_input_and_output_residuals():
    """Returning an image-wide smoothing score instead of background-noise reduction must fail."""
    reference = _gaussian_cut()
    background = reference.values <= np.quantile(reference.values, 0.20)
    pattern = np.zeros_like(reference.values)
    pattern[background] = np.resize(np.array([-0.04, 0.04]), np.count_nonzero(background))
    noisy = reference.copy(data=reference.values + pattern)
    improved = reference.copy(data=reference.values + 0.25 * pattern)

    metrics = evaluate_pair(noisy, improved, reference)

    assert metrics["noise_only_region_reduction"] == pytest.approx(0.75, rel=1e-6)
