import unittest
from orinfer_build.compiler.ascend.scalar_cache import check_scalar_strides, lower_scalar_cache

SOURCE = """
extern "C" __global__ __aicore__ void main_kernel(GM_ADDR a, GM_ADDR b) {
  AscendC::GlobalTensor<int> input;
  input.SetGlobalBuffer((__gm__ int*)a);
  AscendC::GlobalTensor<int> output;
  output.SetGlobalBuffer((__gm__ int*)b);
  if ASCEND_IS_AIC {
    output.SetValue(0, input.GetValue(0));
  }
}
extern "C" void call(aclrtStream stream) {
  main_kernel<<<1, nullptr, stream>>>(nullptr, nullptr);
}
"""


class ScalarCacheTests(unittest.TestCase):
    def test_single_owner_writes_are_visible_to_following_dma_consumers(self):
        result = lower_scalar_cache(SOURCE)
        self.assertEqual(result.count("DataCacheCleanAndInvalid<"), 2)
        self.assertLess(
            result.index("DataCacheCleanAndInvalid<"), result.index("input.GetValue(0)")
        )
        self.assertGreater(
            result.rindex("DataCacheCleanAndInvalid<"), result.index("output.SetValue")
        )
        self.assertIn("main_kernel<<<1, nullptr, stream>>>", result)

    def test_same_dtype_declarations_and_scalar_reads_are_all_tracked(self):
        source = SOURCE.replace("output.SetValue(0, input.GetValue(0));", "use(input.GetValue(0));")
        result = lower_scalar_cache(source)
        self.assertEqual(result.count("DataCacheCleanAndInvalid<"), 1)
        self.assertIn("CACHELINE_OUT>(input);", result)

    def test_parallel_or_unscoped_scalar_writers_are_rejected(self):
        for source in (
            SOURCE.replace("<<<1,", "<<<2,"),
            SOURCE.replace("if ASCEND_IS_AIC", "if ASCEND_IS_AIV"),
        ):
            with self.assertRaises(ValueError):
                lower_scalar_cache(source)

    def test_dma_and_local_scalar_access_need_no_global_cache_maintenance(self):
        source = SOURCE.replace(
            "output.SetValue(0, input.GetValue(0));", "local.SetValue(0, local.GetValue(0));"
        )
        self.assertEqual(lower_scalar_cache(source), source)

    def test_multidimensional_global_scalar_tile_lowering_is_rejected(self):
        broken = """
        AscendC::GlobalTensor<float> normalizers;
        auto normalizers_scalar = -(float)normalizers.GetValue(row);
        AscendC::Adds(p[0], p[0], normalizers_scalar, 1024);
        """
        with self.assertRaisesRegex(ValueError, "normalizers"):
            check_scalar_strides(broken, [{"name": "normalizers", "shape": [544, 4]}])
        # 1D and trailing-unit tensors have no dropped stride.
        check_scalar_strides(broken, [{"name": "normalizers", "shape": [544]}])
        check_scalar_strides(broken, [{"name": "normalizers", "shape": [544, 1]}])
        # Explicit scalar local loads keep full 2D offsets and pass.
        fixed = """
        AscendC::GlobalTensor<float> normalizers;
        float log_z_p = normalizers.GetValue(row * 4);
        float log_z_q = normalizers.GetValue(row * 4 + 1);
        AscendC::Adds(p[0], p[0], -log_z_p, 1024);
        """
        check_scalar_strides(fixed, [{"name": "normalizers", "shape": [544, 4]}])


if __name__ == "__main__":
    unittest.main()
