"""CPU compiler checks for the affine-shape host ABI regression."""

import unittest

from kernels.model.rows import explicit_rows
from tools.model.flash_next.dynamic_batch import factory


class DynamicRowsTests(unittest.TestCase):
    def test_affine_greedy_output_has_an_explicit_scalar_parameter(self):
        fn = explicit_rows(factory("greedy-merge-4-248320", 4))
        (scalar,) = [p for p in fn.params if str(p.dtype) == "int32"]
        self.assertEqual(scalar.name, "m")
        self.assertTrue(explicit_rows(fn).same_as(fn))

    def test_expert_rotation_uses_assignment_rows_without_implicit_affine_inference(self):
        fn = explicit_rows(factory("rotate-a8-ffn-8", 8))
        (scalar,) = [p for p in fn.params if str(p.dtype) == "int32"]
        for buffer in fn.buffer_map.values():
            if buffer.name in ("X", "Q", "S"):
                self.assertTrue(buffer.shape[0].same_as(scalar))

    def test_unknown_shared_geometry_is_not_silently_reused(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            factory("unknown-4", 4)


if __name__ == "__main__":
    unittest.main()
