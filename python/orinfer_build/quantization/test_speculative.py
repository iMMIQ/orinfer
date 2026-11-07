import unittest

import numpy as np

from orinfer_build.quantization.speculative import (
    calibrated_survival,
    calibration_metrics,
    fit_temperatures,
    survival_labels,
)


class SpeculativeCalibrationTests(unittest.TestCase):
    def test_temperature_fits_prefix_survival_and_keeps_monotonicity(self):
        confidence = np.full((128, 3), 0.9)
        accepted = np.tile([0, 0, 1, 3], 32)
        fitted = fit_temperatures(confidence, accepted)
        before = calibration_metrics(confidence, accepted, np.ones(3))
        after = calibration_metrics(confidence, accepted, fitted)
        self.assertLess(after["cumulative_brier"], before["cumulative_brier"])
        self.assertTrue(np.all(np.diff(calibrated_survival(confidence, fitted), axis=1) <= 0))
        np.testing.assert_array_equal(
            survival_labels(np.array([0, 1, 3]), 3),
            [[False, False, False], [True, False, False], [True, True, True]],
        )

    def test_invalid_probability_and_temperature_are_rejected(self):
        with self.assertRaises(ValueError):
            calibrated_survival([[1.1]], [1.0])
        with self.assertRaises(ValueError):
            calibrated_survival([[0.5]], [0.0])
        with self.assertRaises(ValueError):
            fit_temperatures([[0.5]], np.array([2]))
        with self.assertRaises(ValueError):
            fit_temperatures([], np.array([], dtype=int))
