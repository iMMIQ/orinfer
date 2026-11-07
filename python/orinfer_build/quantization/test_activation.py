import unittest

import numpy as np

from orinfer_build.quantization.activation import (
    fit_output_scales,
    quantize_rows,
    refine_integer_codes,
    relative_mse,
    screen_linear,
    statistics,
)


class ActivationCalibrationTests(unittest.TestCase):
    def test_coordinate_rounding_uses_activations_and_respects_int4_limits(self):
        x = np.eye(2, dtype="f4")
        codes = np.array([[0, 0], [7, -8]], dtype="i1")
        target = np.array([[1, 20], [-1, -20]], dtype="f4")
        rounded = refine_integer_codes(x, target, codes, np.ones(2), np.ones_like(target), steps=2)
        np.testing.assert_array_equal(rounded, [[1, -1], [7, -8]])
        np.testing.assert_array_equal(codes, [[0, 0], [7, -8]])
        unchanged = refine_integer_codes(x, target, codes, np.ones(2), np.zeros_like(target))
        np.testing.assert_array_equal(unchanged, codes)

    def test_weighted_scale_fit_keeps_unobserved_channels_and_bounds_changes(self):
        products = np.array([[1, 0], [2, 0]], dtype="f4")
        reference = np.array([[2, 4], [2, 8]], dtype="f4")
        scales = np.ones(2, dtype="f4")
        weights = np.array([[1, 1], [0, 1]], dtype="f4")
        fitted = fit_output_scales(products, reference, scales, weights, shrink=0.5, limit=0.04)
        np.testing.assert_allclose(fitted, [1.02, 1.0])
        np.testing.assert_array_equal(
            fit_output_scales(products, reference, scales, shrink=0), scales
        )
        for invalid in (weights[:, :1], -weights, weights * np.nan):
            with self.assertRaises(ValueError):
                fit_output_scales(products, reference, scales, invalid)

    def test_activation_energy_changes_the_importance_of_equal_weight_errors(self):
        x = np.array([[100, 1], [-100, 1]], dtype=np.float32)
        reference = np.array([[1], [1]], dtype=np.float32)
        important = reference + x @ np.array([[1], [0]], dtype=np.float32)
        small = reference + x @ np.array([[0], [1]], dtype=np.float32)
        self.assertGreater(
            relative_mse(important, reference), 1000 * relative_mse(small, reference)
        )
        self.assertGreater(statistics(x)["max_channel_energy_fraction"], 0.999)

    def test_joint_a8_loss_is_present_even_when_weights_are_exact(self):
        weight = np.array([[0, 1]], dtype=np.float32)
        x = np.array([[127, 0.1], [127, -0.1]], dtype=np.float32)
        weight_only = screen_linear(weight, x, x, bits=8, clips=(1.0,))
        joint = screen_linear(weight, x, x, bits=8, quantize_activation=True, clips=(1.0,))
        self.assertEqual(weight_only["holdout_nmse"], 0)
        self.assertEqual(joint["weight_only_nmse"], 0)
        self.assertGreater(joint["activation_only_nmse"], 0.9)

    def test_holdout_does_not_select_training_clipping(self):
        rng = np.random.default_rng(20261002)
        weight = rng.normal(size=(7, 16))
        train = rng.normal(size=(32, 16))
        a = screen_linear(weight, train, rng.normal(size=(16, 16)), bits=4)
        b = screen_linear(weight, train, rng.normal(size=(16, 16)) * 50, bits=4)
        self.assertEqual((a["clip"], a["train_nmse"]), (b["clip"], b["train_nmse"]))
        np.testing.assert_array_equal(quantize_rows(np.zeros((2, 16))), np.zeros((2, 16)))

    def test_rejects_invalid_numeric_contracts(self):
        for x in (np.empty((0, 2)), np.full((2, 2), np.nan), np.zeros(2)):
            with self.assertRaises(ValueError):
                quantize_rows(x)
        with self.assertRaises(ValueError):
            screen_linear(np.ones((2, 3)), np.ones((2, 4)), np.ones((2, 3)), bits=4)
        with self.assertRaises(ValueError):
            quantize_rows(np.ones((2, 3)), bits=2)
