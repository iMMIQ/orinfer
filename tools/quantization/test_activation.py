import unittest

import numpy as np

from tools.quantization.activation import quantize_rows, relative_mse, screen_linear, statistics


class ActivationCalibrationTests(unittest.TestCase):
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
