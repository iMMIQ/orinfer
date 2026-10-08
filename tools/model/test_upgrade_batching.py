import unittest
import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.model.optimize_batch_gdn import upgrade as upgrade_gdn
from tools.model.upgrade_batching import bind


class BatchAbiTests(unittest.TestCase):
    def test_gdn_upgrade_uses_checkpoint_layers_and_requires_private_fp32_state(self):
        data = dict(
            metadata=dict(
                buffers=[
                    dict(name="L0_State", dtype="f32", shape=[1, 48, 128, 128]),
                    dict(name="L0_History", dtype="f16", shape=[1, 3, 10240]),
                ],
                weight_bytes=123,
            ),
            buffer_scopes={"L0_State": "sequence", "L0_History": "sequence"},
        )
        package = dict(batch_profiles=[2], kernels=[])
        text = dict(num_hidden_layers=2, layer_types=["linear_attention", "full_attention"])
        kernels = SimpleNamespace(
            batch_gdn_conv=lambda rows: None, batch_gdn_recurrent=lambda rows: None
        )

        def export_kernel(kernel, output):
            output.mkdir(parents=True, exist_ok=True)
            (output / "host.txt").write_text("fixture host")

        compiler = SimpleNamespace(configure=lambda: None, export_kernel=export_kernel)
        prefix = "tools.model.optimize_batch_gdn."
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            operator = root / "operators"
            with (
                patch.dict(
                    sys.modules,
                    {"kernels.model.gdn_batch": kernels, "tools.operators.common": compiler},
                ),
                patch(prefix + "clone_model", return_value=operator) as clone,
                patch(prefix + "parse_host", return_value=[{}]),
                patch(prefix + "file_hash", return_value="digest"),
                patch(
                    prefix + "bind",
                    side_effect=lambda export, name, names, scalars: dict(
                        name=name, bindings=names
                    ),
                ),
                patch(prefix + "commit_package", return_value="digest") as commit,
            ):
                for config in (text, {"text_config": text}):
                    (root / "config.json").write_text(json.dumps(config))
                    candidate, execution = copy.deepcopy(data), copy.deepcopy(package)
                    with patch(
                        prefix + "load_model", return_value=(candidate, operator, execution)
                    ):
                        upgrade_gdn(root, root / "result", root)
                    self.assertEqual(
                        [k["name"] for k in execution["kernels"]],
                        [
                            "batch_gdn_m2/layer0/k0",
                            "batch_gdn_m2/layer0/k1",
                        ],
                    )
                    self.assertEqual(candidate["metadata"]["buffers"][-1]["shape"], [2, 128, 3])
                    self.assertEqual(candidate["buffer_scopes"]["BatchGdnPointers"], "workspace")
                    self.assertTrue(execution["batch_gdn"])
                    commit.assert_called()
                clone.reset_mock()
                data["metadata"]["buffers"][0]["dtype"] = "f16"
                with patch(prefix + "load_model", return_value=(data, operator, package)):
                    with self.assertRaisesRegex(ValueError, "Unsupported private mixer layout"):
                        upgrade_gdn(root, root / "invalid", root)
                clone.assert_not_called()

    def test_dynamic_shape_is_bound_by_host_abi_order(self):
        export = dict(
            module={"file": "kernel.cubin"},
            source={"file": "kernel.cu"},
            host_abi={"file": "host.txt"},
            symbol="kernel_kernel",
            ordered_arguments=[
                dict(ctype="ctypes.c_int32", value="M"),
                dict(ctype="ctypes.c_void_p", value="State.data_ptr()"),
                dict(ctype="ctypes.c_int32", value="batch"),
            ],
            launch_expressions=dict(
                gridDimX="(M + 15) // 16",
                gridDimY="batch",
                gridDimZ="1",
                blockDimX="128",
                blockDimY="1",
                blockDimZ="1",
                sharedMemBytes="0",
            ),
        )
        actual = bind(export, "batch_m4/layer0/k0", {"State": "L0_State"}, dict(M=4, batch=1))
        self.assertEqual(
            actual["args"],
            [
                dict(kind="i32", value=4),
                dict(kind="buffer", name="L0_State"),
                dict(kind="i32", value=1),
            ],
        )
        self.assertEqual(actual["grid"], [1, 1, 1])
        self.assertFalse(actual["cooperative"])
        export["ordered_arguments"][0]["ctype"] = "ctypes.c_double"
        with self.assertRaises(ValueError):
            bind(export, "invalid", {"State": "L0_State"}, dict(M=4, batch=1))


if __name__ == "__main__":
    unittest.main()
