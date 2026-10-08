import unittest

from tools.model.resize_context import recipe, rebind, trim_prefill


HOST = """
def call(kernels, K, Out, tokens, stream=0):
    config = CUlaunchConfig()
    config.gridDimX = (tokens * 512 + 2047) // 2048
    config.gridDimY = 1
    config.gridDimZ = 1
    config.blockDimX = 256
    config.blockDimY = 1
    config.blockDimZ = 1
    config.sharedMemBytes = 0
    arg_values = K.data_ptr(), Out.data_ptr(), tokens
    arg_types = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32
    res = cuLaunchKernelEx(config, kernels['kernel_kernel'], (arg_values, arg_types), 0)[0]
"""


class ResizeTests(unittest.TestCase):
    def test_smaller_prefill_preserves_vision_and_gdn_head_dimensions(self):
        buffers = [
            dict(name=n, shape=s, data=None)
            for n, s in [
                ("Input", [8192]),
                ("Hidden", [8192, 5120]),
                ("Q", [16, 8192, 128]),
                ("Senter", [48, 128, 128, 128]),
                ("TemporaryA8", [8192 * 17408]),
                ("VMerged", [8192, 4608]),
                ("MtpTargetHidden", [8192, 5120]),
            ]
        ]
        wrapper = dict(
            buffer_scopes={b["name"]: "workspace" for b in buffers},
            metadata=dict(
                chunk_tokens=8192,
                input="Input",
                buffers=buffers,
                prefill_plans=[dict(chunk_tokens=n) for n in [512, 2048, 8192]],
                mtp=dict(capture_plans=[dict(tokens=n) for n in [1, 2048, 8192]]),
            ),
        )
        package = dict(
            prefill_profiles=[dict(tokens=n) for n in [512, 2048, 8192]],
            kernels=[
                dict(name="prefill_m8192/layer3/k5"),
                dict(name="prefill_m2048/layer3/k5"),
                dict(name="mtp_warm_m16/body/k0"),
                dict(name="head_m8192/body/k0"),
            ],
        )
        trim_prefill(wrapper, package, 2048)
        shapes = {b["name"]: b["shape"] for b in buffers}
        self.assertEqual(shapes["Q"], [16, 2048, 128])
        self.assertEqual(shapes["Senter"], [48, 32, 128, 128])
        self.assertEqual(shapes["TemporaryA8"], [2048 * 17408])
        self.assertEqual(shapes["VMerged"], [8192, 4608])
        self.assertEqual(shapes["MtpTargetHidden"], [8192, 5120])
        self.assertEqual(
            [k["name"] for k in package["kernels"]],
            ["prefill_m2048/layer3/k5", "mtp_warm_m16/body/k0"],
        )
        with self.assertRaises(ValueError):
            trim_prefill(wrapper, package, 4096)

    def test_capacity_rebinding_uses_new_pointer_order_and_grid(self):
        binding = dict(
            name="prefill_m512/layer3/k4",
            args=[
                dict(kind="buffer", name="L3_KPages"),
                dict(kind="buffer", name="Kcontig"),
                dict(kind="i32", value=8704),
            ],
        )
        reordered = HOST.replace("K.data_ptr(), Out.data_ptr()", "Out.data_ptr(), K.data_ptr()")
        result = rebind(binding, HOST, reordered, dict(tokens=262144))
        self.assertEqual(result["grid"], [65536, 1, 1])
        self.assertEqual(
            result["args"], [binding["args"][1], binding["args"][0], dict(kind="i32", value=262144)]
        )
        self.assertEqual(binding["args"][-1]["value"], 8704)

    def test_capacity_selection_does_not_recompile_projection_8704_width(self):
        self.assertIsNone(
            recipe(
                dict(
                    name="prefill_m8192/layer0/k4",
                    args=[
                        dict(kind="buffer", name="Activated"),
                        dict(kind="buffer", name="TemporaryW8"),
                    ],
                )
            )
        )
        for name, expected in [
            ("decode/layer3/k3", ("attention", None)),
            ("mtp_warm_m16/body/k9", ("attention", 16)),
        ]:
            self.assertEqual(
                recipe(
                    dict(
                        name=name,
                        args=[
                            dict(kind="buffer", name="Pages"),
                            dict(kind="buffer", name="MtpAttO"),
                        ],
                    )
                ),
                expected,
            )
        with self.assertRaises(ValueError):
            recipe(dict(name="unknown", args=[dict(kind="buffer", name="Pages")]))


if __name__ == "__main__":
    unittest.main()
