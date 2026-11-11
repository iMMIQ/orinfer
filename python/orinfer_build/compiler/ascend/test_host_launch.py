import unittest

from orinfer_build.compiler.ascend.host_launch import lower_host_arguments


SOURCE = """extern "C" __global__ __aicore__ void kernel(GM_ADDR x, int64_t m) {
  device_body();
}
extern "C" void call(uint8_t* x, int64_t m, aclrtStream stream) {
  kernel_tiling(m);
  kernel<<<(2 * (m < 64 ? m : 64)), nullptr, stream>>>(x, m);
}
"""


class HostLaunchTests(unittest.TestCase):
    def test_preserves_device_tiling_grid_arguments_and_returns_api_errors(self):
        result = lower_host_arguments(SOURCE)
        self.assertEqual(
            result.split('extern "C" int32_t call')[0], SOURCE.split('extern "C" void call')[0]
        )
        self.assertIn("kernel_tiling(m);", result)
        self.assertIn("decltype(x) p0; decltype(m) p1;", result)
        self.assertIn("parameters{x, m}", result)
        self.assertIn("static_cast<uint32_t>((2 * (m < 64 ? m : 64)))", result)
        self.assertIn("if (status) return status;", result)
        self.assertIn("return aclrtLaunchKernelWithHostArgs", result)
        self.assertNotIn("<<<", result)

    def test_unknown_shared_memory_or_argument_expression_is_rejected(self):
        for source in (
            SOURCE.replace("nullptr, stream", "workspace, stream"),
            SOURCE.replace(">>>(x, m)", ">>>(x, m + 1)"),
            SOURCE.replace("  kernel_tiling(m);", "  kernel<<<1, nullptr, stream>>>(x, m);"),
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                lower_host_arguments(source)
