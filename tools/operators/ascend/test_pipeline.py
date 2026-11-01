"""The gather schedule only relaxes fences inside a declared DMA-only loop."""

import unittest

from tools.operators.ascend.pipeline import lower_gather_pipeline


class GatherPipelineTests(unittest.TestCase):
    def region(self, body):
        return (
            "AscendC::PipeBarrier<PIPE_ALL>();\n"
            "AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(7);\n"
            + body
            + "\nAscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(7);\n"
            "AscendC::PipeBarrier<PIPE_ALL>();"
        )

    def test_outer_fences_and_other_events_survive(self):
        body = (
            "for (int lane=0; lane<16; ++lane) {\n"
            "AscendC::PipeBarrier<PIPE_ALL>();\n"
            "tl::ascend::copy_gm_to_ub<int8_t,64,8>(dst[lane*512], src[lane*512],64);\n"
            "AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);\n}"
        )
        result = lower_gather_pipeline(self.region(body))
        self.assertEqual(result.count("PipeBarrier<PIPE_ALL>"), 2)
        self.assertIn("SetFlag<AscendC::HardEvent::MTE2_S>", result)
        self.assertIn("copy_gm_to_ub<int8_t,64,8>", result)
        self.assertNotIn("HardEvent::S_MTE2>(7)", result)

    def test_unknown_arithmetic_global_writes_and_unpaired_regions_fail(self):
        for body in ["AscendC::Cast(x,y);", "tl::ascend::copy_ub_to_gm(x,y);", "gm.SetValue(0,1);"]:
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, "unsupported"):
                lower_gather_pipeline("AscendC::GlobalTensor<int32_t> gm;\n" + self.region(body))
        for marker in ["SetFlag", "WaitFlag"]:
            with self.assertRaisesRegex(ValueError, "paired"):
                lower_gather_pipeline(f"AscendC::{marker}<AscendC::HardEvent::S_MTE2>(7);")

    def test_unmarked_source_keeps_all_synchronization(self):
        source = "AscendC::PipeBarrier<PIPE_ALL>(); AscendC::Cast(x,y);"
        self.assertEqual(lower_gather_pipeline(source), source)
        with self.assertRaisesRegex(ValueError, "discarded"):
            lower_gather_pipeline(source, required=True)
