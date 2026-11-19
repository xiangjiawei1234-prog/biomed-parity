import numpy as np
import pytest

from biomed_parity.competing_risks import cmprsk_residuals_by_time, cmprsk_timepoints


def test_cumulative_incidence_uses_post_jump_values_at_ties():
    grid, estimate, variance = cmprsk_timepoints(
        [0, 1, 1, 3, 3, 4],
        [0, 0, 0.2, 0.2, 0.46, 0.46],
        [0, 0, 0.04, 0.04, 0.08, 0.08],
        [4.001, -1, 4, 3, 1, 1, 0.5, 2, np.nan],
    )
    np.testing.assert_array_equal(grid, [-1, 0.5, 1, 2, 3, 4, 4.001])
    np.testing.assert_allclose(estimate, [np.nan, 0, 0.2, 0.2, 0.46, 0.46, np.nan])
    np.testing.assert_allclose(variance, [np.nan, 0, 0.04, 0.04, 0.08, 0.08, np.nan])


def test_cumulative_incidence_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="sorted"):
        cmprsk_timepoints([1, 0], [0, 1], [0, 1], [0])
    with pytest.raises(ValueError, match="infinities"):
        cmprsk_timepoints([0], [0], [0], [np.inf])


def test_residuals_aggregate_only_rows_with_identical_times():
    times, residuals = cmprsk_residuals_by_time(
        [5, 2, 5, 3], [[1, 2], [-1, 1], [2, -2], [1.5, 3]],
    )
    np.testing.assert_array_equal(times, [2, 3, 5])
    np.testing.assert_allclose(residuals, [[-1, 1], [1.5, 3], [3, 0]])


def test_residuals_support_empty_inputs():
    times, residuals = cmprsk_residuals_by_time([], np.empty((0, 2)))
    assert times.shape == (0,)
    assert residuals.shape == (0, 2)
