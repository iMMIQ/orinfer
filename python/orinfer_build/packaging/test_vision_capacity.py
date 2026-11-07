"""Capacity changes preserve per-image geometry and all text execution contracts."""

import copy
import unittest

from orinfer_build.packaging.vision_capacity import capacity_metadata


class CapacityTests(unittest.TestCase):
    def test_only_total_request_feature_extent_changes(self):
        data = dict(
            architecture="qwen3_5",
            buffer_scopes={"Features": "sequence"},
            metadata=dict(
                max_context=262144,
                chunk_tokens=4096,
                vision=dict(
                    max_features=32768, max_patches=32768, hidden=5120, features="Features"
                ),
                buffers=[
                    dict(name="Features", dtype="f16", shape=[32768, 5120]),
                    dict(name="Hidden", dtype="f16", shape=[4096, 5120]),
                ],
            ),
        )
        package = dict(
            buffer_contracts=copy.deepcopy(data["metadata"]["buffers"]),
            dynamic_batch_kernels=[dict(name="prefill", capacity=4096)],
        )
        original_data, original_package = copy.deepcopy(data), copy.deepcopy(package)
        capacity_metadata(data, package, 262144)
        self.assertEqual(data["metadata"]["vision"]["max_features"], 262144)
        self.assertEqual(data["metadata"]["buffers"][0]["shape"], [262144, 5120])
        self.assertEqual(package["buffer_contracts"][0]["shape"], [262144, 5120])
        data["metadata"]["vision"]["max_features"] = 32768
        data["metadata"]["buffers"][0]["shape"][0] = 32768
        package["buffer_contracts"][0]["shape"][0] = 32768
        self.assertEqual(data, original_data)
        self.assertEqual(package, original_package)
        for invalid in (32768, 262145, 0, True):
            with self.assertRaises(ValueError):
                capacity_metadata(data, package, invalid)
        data["buffer_scopes"]["Features"] = "workspace"
        with self.assertRaisesRegex(ValueError, "request-owned"):
            capacity_metadata(data, package, 65536)
        data["buffer_scopes"]["Features"] = "sequence"
        package["buffer_contracts"][0]["shape"][0] = 8192
        with self.assertRaisesRegex(ValueError, "buffer contract"):
            capacity_metadata(data, package, 65536)
        self.assertEqual(data["metadata"]["vision"]["max_features"], 32768)
