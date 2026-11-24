"""CPU unit tests for CUDA target discovery and Orin vs generic-GPU dispatch."""

import os
import unittest
from unittest.mock import patch

from orinfer_build.compiler.cuda.target import (
    DEFAULT_CUDA_ARCH,
    SUPPORTED_CUDA_ARCHES,
    cuda_arch,
    cuda_target,
    host_arch,
    is_generic_cuda_arch,
    is_orin_arch,
    package_target,
    parse_cuda_sm,
    validate_cuda_arch,
)
from orinfer_build.tuning.expert_tuning import (
    EXACT_EXPERT_PASSES,
    expert_configs,
    expert_passes,
)


class CudaTargetTests(unittest.TestCase):
    def test_supported_arches_roundtrip(self):
        for arch in SUPPORTED_CUDA_ARCHES:
            sm = parse_cuda_sm(arch)
            self.assertIsNotNone(sm)
            self.assertGreaterEqual(sm[0], 8)
            self.assertEqual(validate_cuda_arch(arch), arch)
            self.assertEqual(cuda_target(arch), {"kind": "cuda", "arch": arch})
        for bad in ("sm_70", "sm_75", "ascend910b", "", "sm_999"):
            self.assertIsNone(parse_cuda_sm(bad))
            with self.assertRaises(ValueError):
                validate_cuda_arch(bad)

    def test_env_override_and_default_fallback(self):
        with patch.dict(os.environ, {"ORINFER_CUDA_ARCH": "sm_80"}):
            self.assertEqual(cuda_arch(), "sm_80")
            self.assertFalse(is_orin_arch())
            self.assertTrue(is_generic_cuda_arch())
            self.assertEqual(cuda_target(), {"kind": "cuda", "arch": "sm_80"})
        with patch.dict(os.environ, {"ORINFER_CUDA_ARCH": "sm_87"}):
            self.assertEqual(cuda_arch(), "sm_87")
            self.assertTrue(is_orin_arch())
            self.assertFalse(is_generic_cuda_arch())
        with patch.dict(os.environ, {"ORINFER_CUDA_ARCH": ""}):
            # CPU environment without Torch falls back cleanly to sm_87.
            resolved = cuda_arch()
            self.assertIn(resolved, SUPPORTED_CUDA_ARCHES)
            self.assertEqual(DEFAULT_CUDA_ARCH, "sm_87")

    def test_package_target_formatting_matches_rust_sdk(self):
        self.assertEqual(package_target("sm_87", "aarch64"), "sm_87")
        self.assertEqual(package_target("sm_87", "x86_64"), "sm_87-x86_64")
        self.assertEqual(package_target("sm_80", "x86_64"), "sm_80")
        self.assertEqual(package_target("sm_90", "x86_64"), "sm_90")
        self.assertEqual(package_target("sm_90", "aarch64"), "sm_90-aarch64")
        self.assertEqual(package_target("sm_120", "x86_64"), "sm_120")
        self.assertEqual(host_arch("amd64"), "x86_64")
        self.assertEqual(host_arch("arm64"), "aarch64")
        with self.assertRaises(ValueError):
            host_arch("riscv64")

    def test_expert_tuning_preserves_orin_and_uses_portable_path_on_generic_gpus(self):
        orin_cfgs = expert_configs("e8p_q2", arch="sm_87")
        self.assertEqual(orin_cfgs[0], {"block_m": 8, "stages": 2})
        self.assertEqual(expert_passes("e8p_q2", orin_cfgs[0], arch="sm_87"), EXACT_EXPERT_PASSES)

        for arch in ("sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120"):
            gen_cfgs = expert_configs("e8p_q2", arch=arch)
            self.assertTrue(all(cfg.get("portable") is True for cfg in gen_cfgs))
            self.assertEqual(expert_passes("e8p_q2", gen_cfgs[0], arch=arch), {})


if __name__ == "__main__":
    unittest.main()
