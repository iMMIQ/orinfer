import unittest

from tools.operators.ascend.vector_launch import lower_independent_cube, lower_independent_vector


SOURCE = """
void main_kernel(void* input, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    output[cid * 2 + vid] = input[cid * 2 + vid];
  }
}
extern "C" void call(void* input, aclrtStream stream) {
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel<<<20, nullptr, stream>>>(input, fftsAddr);
}
"""


class VectorLaunchTests(unittest.TestCase):
    def test_independent_launch_keeps_arithmetic_and_is_idempotent(self):
        result = lower_independent_vector(SOURCE)
        self.assertIn("KERNEL_TYPE_AIV_ONLY", result)
        self.assertIn("main_kernel<<<40, nullptr, stream>>>", result)
        self.assertIn("(AscendC::GetBlockIdx() % 2)", result)
        self.assertIn("output[cid * 2 + vid] = input[cid * 2 + vid];", result)
        self.assertNotIn("ascend_l1", result)
        self.assertEqual(lower_independent_vector(result), result)

    def test_cube_work_and_used_cube_storage_keep_the_original_launch(self):
        for statement in ("if ASCEND_IS_AIC {}", "auto x = ascend_l1.GetWithOffset<float>(8, 0);"):
            source = SOURCE.replace("  auto vid", f"  {statement}\n  auto vid")
            self.assertEqual(lower_independent_vector(source), source)

    def test_communication_and_global_barriers_keep_the_original_launch(self):
        for statement in (
            "CrossCoreSetFlag();",
            "SyncAll();",
            "GlobalBarrier();",
            "use(fftsAddr);",
        ):
            source = SOURCE.replace("  auto vid", f"  {statement}\n  auto vid")
            self.assertEqual(lower_independent_vector(source), source)

    def test_unknown_grid_or_pair_mapping_keeps_the_original_launch(self):
        for before, after in (
            ("<<<20,", "<<<capacity,"),
            ("<<<20,", "<<<32768,"),
            ("cid / 2", "cid / 4"),
        ):
            source = SOURCE.replace(before, after)
            self.assertEqual(lower_independent_vector(source), source)

    def test_bounded_dynamic_grid_preserves_pair_rows_for_every_extent(self):
        grid = "(((m + 1) / 2) < 64 ? ((m + 1) / 2) : 64)"
        source = SOURCE.replace("<<<20,", f"<<<{grid},")
        result = lower_independent_vector(source, grid_limit=64)
        self.assertIn("KERNEL_TYPE_AIV_ONLY", result)
        self.assertIn(f"<<<(2 * ({grid})), nullptr, stream>>>", result)
        self.assertNotIn("ascend_l1", result)
        for rows in range(1, 4097):
            pairs = min((rows + 1) // 2, 64)
            expected = [(cid, lane) for cid in range(pairs) for lane in range(2)]
            self.assertEqual([(block // 2, block % 2) for block in range(pairs * 2)], expected)
        self.assertEqual(lower_independent_vector(source), source)

    def test_dynamic_grid_requires_a_valid_bound_and_known_scalar(self):
        for grid in ("capacity", "m + other", "min(m, 64)"):
            source = SOURCE.replace("<<<20,", f"<<<{grid},")
            self.assertEqual(lower_independent_vector(source, grid_limit=64), source)
        source = SOURCE.replace("<<<20,", "<<<m < 64 ? m : 64,")
        for limit in (None, 0, -1, 32768, True):
            self.assertEqual(lower_independent_vector(source, grid_limit=limit), source)

    def test_cube_stage_keeps_its_grid_and_arithmetic(self):
        source = SOURCE.replace(
            "  auto vid",
            "  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub;\n"
            "  pipe.InitBuffer(ascend_ub, 196352);\n  auto vid",
        ).replace("  if ASCEND_IS_AIV {\n    output", "  if ASCEND_IS_AIC {\n    output")
        source = source.replace("input[cid * 2 + vid]", "input[cid]").replace(
            "output[cid * 2 + vid]", "output[cid]"
        )
        result = lower_independent_cube(source)
        self.assertIn("KERNEL_TYPE_AIC_ONLY", result)
        self.assertIn("main_kernel<<<20, nullptr, stream>>>", result)
        self.assertIn("output[cid] = input[cid];", result)
        self.assertNotIn("ascend_ub", result)
        self.assertIn("ascend_l1", result)
        self.assertEqual(lower_independent_cube(result), result)
        self.assertEqual(lower_independent_cube(SOURCE), SOURCE)


if __name__ == "__main__":
    unittest.main()
