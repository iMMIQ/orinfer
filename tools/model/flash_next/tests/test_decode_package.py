"""Rebuilt native launch order must preserve buffer slices and reject scalars."""

import unittest

from tools.model.flash_next.optimize_decode import pointer_bindings, replace_binding


class DecodePackageTests(unittest.TestCase):
    def test_reordered_launch_preserves_slice_without_mutating_original(self):
        original = dict(
            name="verify_m3/layer0/k30",
            args=[
                dict(kind="buffer_slice", name="Input", offset=256),
                dict(kind="buffer", name="Output"),
            ],
        )
        host = dict(
            ordered_arguments=[
                dict(ctype="ctypes.c_void_p", value="A.data_ptr()"),
                dict(ctype="ctypes.c_void_p", value="C.data_ptr()"),
            ]
        )
        pointers = pointer_bindings(original, host)
        rebuilt = dict(
            ordered_arguments=list(reversed(host["ordered_arguments"])),
            symbol="routed_kernel",
            launch_expressions=dict(
                gridDimX="32",
                gridDimY="10",
                gridDimZ="1",
                blockDimX="128",
                blockDimY="1",
                blockDimZ="1",
                sharedMemBytes="16384",
            ),
        )
        binding = replace_binding(original, rebuilt, {}, pointers)
        self.assertEqual(binding["args"], list(reversed(original["args"])))
        self.assertEqual(binding["grid"], [32, 10, 1])
        binding["args"][1]["offset"] = 512
        self.assertEqual(original["args"][0]["offset"], 256)
        self.assertEqual(pointers["A"]["offset"], 256)

    def test_symbolic_or_incomplete_abi_is_rejected(self):
        original = dict(args=[dict(kind="buffer", name="Input")])
        for host in [
            dict(ordered_arguments=[]),
            dict(ordered_arguments=[dict(ctype="ctypes.c_int32", value="m")]),
        ]:
            with self.assertRaises(ValueError):
                pointer_bindings(original, host)


if __name__ == "__main__":
    unittest.main()
