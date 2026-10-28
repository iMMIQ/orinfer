import unittest
from tools.operators.ascend.scalar_cache import lower_scalar_cache

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


if __name__ == "__main__":
    unittest.main()
