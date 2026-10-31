"""Container and byte-identity checks for the native release export."""

import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from tools.model.deepseek_v41.export_release import Export, verify_release
from tools.model.safetensors_source import local_header
from tools.build.files import file_hash


def write_tensors(path, tensors):
    header, payload = {}, bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = dict(
            dtype=dtype, shape=shape, data_offsets=[len(payload), len(payload) + len(data)]
        )
        payload.extend(data)
    raw = json.dumps(header).encode()
    raw += b" " * ((-len(raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def read_tensor(root, index, name):
    path = root / index["weight_map"][name]
    start, header = local_header(path)
    lo, hi = header[name]["data_offsets"]
    with path.open("rb") as f:
        f.seek(start + lo)
        return f.read(hi - lo)


class ReleaseExportTests(unittest.TestCase):
    def test_roundtrip_keeps_physical_banks_and_global_host_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "release"
            package = source / "cache/packages/pinned"
            package.mkdir(parents=True)
            for name in [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "generation_config.json",
            ]:
                (source / name).write_text("{}")
            (source / "chat_template.jinja").write_text("chat")
            (source / "LICENSE").write_text("fixture license")
            (source / "cache/model.json").write_text(
                json.dumps(dict(schema_version=1, execution_package="pinned"))
            )
            bits = np.array([0x3F80, 0x4000, 0xBF80, 0x0000] * 4, dtype="<u2").reshape(4, 4)
            weights = bytes(range(128)) * 3
            scales = np.arange(12, dtype="<f4").tobytes()
            write_tensors(
                source / "original.safetensors",
                [
                    ("embed.weight", "BF16", [4, 4], bits.tobytes()),
                    ("matrix", "BF16", [4, 4], bits.tobytes()),
                    ("pad", "F32", [2], np.array([1.25, -0.0], dtype="<f4").tobytes()),
                    ("layers.1.engram.embed.weight", "I8", [12, 32], weights),
                    ("layers.1.engram.embed.scale", "F32", [12, 1], scales),
                ],
            )
            names = [
                "embed.weight",
                "matrix",
                "layers.1.engram.embed.weight",
                "layers.1.engram.embed.scale",
                "pad",
            ]
            (source / "model.safetensors.index.json").write_text(
                json.dumps(dict(weight_map=dict.fromkeys(names, "original.safetensors")))
            )
            (package / "engram.json").write_text(
                json.dumps(dict(layers=[dict(layer=1, rows=12, primes=[5, 7])]))
            )
            buffers = []
            for name, size in [("packed", 16), ("scale", 64), ("assist", 32)]:
                (package / (name + ".bin")).write_bytes(bytes(range(size)))
                buffers.append(dict(name="L0." + name, bytes=size, prepared_weight=name + ".bin"))
            buffers.append(
                dict(
                    name="L0.matrix",
                    bytes=16,
                    weight=dict(
                        kind="matrix",
                        name="matrix",
                        dtype="bf16",
                        first_row=1,
                        rows=2,
                        first_column=1,
                        columns=2,
                        float32=True,
                    ),
                )
            )
            buffers.extend(
                [
                    dict(name="L0.pad", bytes=16, weight=dict(kind="pad", name="pad", elements=4)),
                    dict(
                        name="L0.product",
                        bytes=64,
                        weight=dict(kind="channel_product", left="matrix", right="matrix"),
                    ),
                ]
            )
            model = dict(
                schema_version=1,
                index=dict(file="model.safetensors.index.json"),
                engram=dict(file="engram.json"),
                rope={},
                ranks=[
                    dict(
                        rank=0,
                        buffers=buffers,
                        packets=[
                            dict(
                                body=[
                                    dict(
                                        op="w4",
                                        experts=2,
                                        input=4,
                                        output=4,
                                        buffers=[
                                            "a",
                                            "L0.packed",
                                            "L0.scale",
                                            "L0.assist",
                                            "ts",
                                            "out",
                                        ],
                                    )
                                ],
                                head=[],
                            )
                        ],
                    )
                ],
            )
            (package / "model.json").write_text(json.dumps(model))
            exporter = Export(source, output, execution_output=root / "execution").plan()
            exporter.export(2)
            index = json.loads((output / "model.safetensors.index.json").read_text())
            descriptor = json.loads((output / "cache/model.json").read_text())
            plan = json.loads(
                (root / "execution" / descriptor["execution_package"] / "model.json").read_text()
            )
            for b in plan["ranks"][0]["buffers"]:
                name = b["prepared_tensor"]["name"]
                actual = read_tensor(output, index, name)
                if b["name"] == "L0.matrix":
                    expected = (bits[1:3, 1:3].astype("<u4") << 16).tobytes()
                elif b["name"] == "L0.pad":
                    expected = np.array([1.25, -0.0, 0.0, 0.0], dtype="<f4").tobytes()
                elif b["name"] == "L0.product":
                    values = (bits.astype("<u4") << 16).view("<f4")
                    expected = (values * values).tobytes()
                else:
                    expected = (package / (b["name"].split(".")[-1] + ".bin")).read_bytes()
                self.assertEqual(actual, expected)
                self.assertNotIn("weight", b)
                self.assertNotIn("prepared_weight", b)
            for logical, expected in [(names[2], weights), (names[3], scales)]:
                parts = index["row_shards"][logical]
                self.assertEqual(b"".join(read_tensor(output, index, n) for n in parts), expected)
                self.assertEqual(len(parts), 4)
            self.assertEqual(read_tensor(output, index, "embed.weight"), bits.tobytes())
            for name, record in exporter.manifest["files"].items():
                self.assertEqual(file_hash(output / name), record["sha256"])
            self.assertEqual(verify_release(output, 2, root / "execution")["prepared_bindings"], 6)
            self.assertIsNone(verify_release(output, 2)["prepared_bindings"])
            self.assertFalse((output / "cache/packages").exists())
            self.assertFalse(any(output.rglob("*.so")))
            self.assertFalse((output / "original.safetensors").exists())
            self.assertEqual((output / "LICENSE").read_text(), "fixture license")


if __name__ == "__main__":
    unittest.main()
