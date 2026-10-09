import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from tools.model.compact import compact, header, write_shard
from tools.model.publication import AssetReader, file_hash, write_json
from tools.model.package import archive, install


class CompactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "source"
        self.weights = self.model / "cache/weights"
        self.weights.mkdir(parents=True)
        buffers, mapping = [], {}
        self.values = {
            "packed": np.arange(8, dtype=np.uint8),
            "scales": np.array([0x8000, 0x7E01], dtype=np.uint16),
        }
        for i, (name, value) in enumerate(self.values.items()):
            filename = f"weight-{i}.safetensors"
            save_file(
                {name: value},
                str(self.weights / filename),
                metadata={f"orin.layout.{name}": "physical"},
            )
            mapping[name] = filename
            buffers.append(
                dict(
                    name=name,
                    layout="physical",
                    data=dict(tensor=name, sha256=hashlib.sha256(value.tobytes()).hexdigest()),
                )
            )
        write_json(self.weights / "model.safetensors.index.json", dict(weight_map=mapping))
        cpu = self.model / "cache/cpu"
        cpu.mkdir()
        assets = dict(version=1, books=[], embedding=[], ple=[])
        for i in range(2):
            for kind in ("embedding", "ple"):
                name = f"{kind}-{i}.safetensors"
                data = (
                    {
                        "rows": np.full((1, 2560), i, dtype=np.int8),
                        "scales": np.array([i + 1], dtype=np.float16),
                    }
                    if kind == "embedding"
                    else {"rows": np.full((2, 42), i, dtype=np.uint8)}
                )
                save_file(data, str(cpu / name))
                begin, spec = header(cpu / name)
                part = dict(
                    file=f"cache/cpu/{name}",
                    sha256=file_hash(cpu / name),
                    first=i if kind == "embedding" else 2 * i,
                    rows=1 if kind == "embedding" else 2,
                    offset=begin + spec["rows"]["data_offsets"][0],
                )
                if kind == "embedding":
                    part["scale_offset"] = begin + spec["scales"]["data_offsets"][0]
                else:
                    part["book"] = 0
                assets[kind].append(part)
        write_json(cpu / "inputs.json", assets)
        package = self.root / "package"
        package.mkdir()
        kernels = []
        for i, raw in enumerate((b"\x7fELFexact-cubin", b"exact-source", b"exact-host")):
            filename = f"asset-{i}"
            (package / filename).write_bytes(raw)
            kernels.append(dict(file=filename, sha256=hashlib.sha256(raw).hexdigest()))
        library = bytearray(20)
        library[:6] = b"\x7fELF\x02\x01"
        library[16:20] = bytes((3, 0, 183, 0))
        (package / "lib").mkdir()
        (package / "lib/model.so").write_bytes(library)
        spec = dict(
            schema_version=1,
            runtime_abi=1,
            execution=dict(
                abi_version=1,
                package="fixture",
                version="1",
                library=dict(file="lib/model.so", sha256=hashlib.sha256(library).hexdigest()),
            ),
            kernels=[dict(module=kernels[0], source=kernels[1], host_abi=kernels[2])] * 2,
        )
        raw = json.dumps(spec).encode()
        digest = hashlib.sha256(raw).hexdigest()
        directory = self.model / "cache/packages" / digest
        directory.mkdir(parents=True)
        for path in package.iterdir():
            if path.is_file():
                (directory / path.name).write_bytes(path.read_bytes())
        (directory / "lib").mkdir()
        (directory / "lib/model.so").write_bytes(library)
        (directory / "package.json").write_bytes(raw)
        write_json(
            self.model / "cache/model.json",
            dict(
                schema_version=1,
                execution_package=digest,
                metadata=dict(
                    buffers=buffers,
                    input_assets=dict(
                        file="cache/cpu/inputs.json", sha256=file_hash(cpu / "inputs.json")
                    ),
                ),
            ),
        )
        write_json(self.model / "source.json", dict(execution_package=digest))

    def test_roundtrip_preserves_weights_cpu_rows_and_separate_kernel_assets(self):
        output, cache = self.root / "output", self.root / "operators"
        with contextlib.redirect_stdout(io.StringIO()):
            report = compact(self.model, output, cache, shard_bytes=10000)
        self.assertEqual((report["gpu_shards"], report["cpu_shards"]), (1, 2))
        self.assertFalse((output / "cache/packages").exists())
        index = json.loads((output / "cache/weights/model.safetensors.index.json").read_text())
        for name, value in self.values.items():
            with safe_open(
                output / "cache/weights" / index["weight_map"][name], framework="numpy"
            ) as sf:
                self.assertEqual(sf.get_tensor(name).tobytes(), value.tobytes())
                self.assertEqual(sf.metadata()[f"orin.layout.{name}"], "physical")
        assets = json.loads((output / "cache/cpu/inputs.json").read_text())
        for kind, width in (("embedding", 2560), ("ple", 42)):
            self.assertEqual(len(assets[kind]), 1)
            p = assets[kind][0]
            self.assertEqual(file_hash(output / p["file"]), p["sha256"])
            with (output / p["file"]).open("rb") as stream:
                stream.seek(p["offset"])
                raw = stream.read(p["rows"] * width)
                rows = 1 if kind == "embedding" else 2
                self.assertEqual(raw, bytes(rows * width) + bytes([1]) * rows * width)
                if kind == "embedding":
                    stream.seek(p["scale_offset"])
                    self.assertEqual(stream.read(4), np.array([1, 2], dtype=np.float16).tobytes())
        package = cache / report["execution_package"]
        manifest = json.loads((package / "package.json").read_text())
        reader = AssetReader(package)
        for field, expected in (
            ("module", b"\x7fELFexact-cubin"),
            ("source", b"exact-source"),
            ("host_abi", b"exact-host"),
        ):
            self.assertEqual(reader.read(manifest["kernels"][0][field]), expected)
        self.assertEqual(len(reader.headers), 1)
        with safe_open(package / "assets.safetensors", framework="numpy") as sf:
            self.assertEqual(len(sf.keys()), 3)
        packed = self.root / "operators.tar.gz"
        archive(package, packed)
        installed = self.root / "installed"
        self.assertEqual(install(packed, installed), report["execution_package"])
        self.assertEqual(
            AssetReader(installed / report["execution_package"]).read(
                manifest["kernels"][0]["module"]
            ),
            b"\x7fELFexact-cubin",
        )

    def test_corrupt_tensor_aborts_without_publication(self):
        path = self.weights / "weight-0.safetensors"
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            compact(self.model, self.root / "output", self.root / "operators")
        self.assertFalse((self.root / "output").exists())

    def test_invalid_extent_never_publishes_shard(self):
        source = self.root / "data"
        source.write_bytes(b"abcd")
        output = self.root / "invalid.safetensors"
        with self.assertRaises(ValueError):
            write_shard(output, [dict(name="x", dtype="F16", shape=[4], segments=[(source, 0, 4)])])
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".tmp").exists())

    def test_bundled_assets_reject_missing_wrong_shape_and_corrupt_payload(self):
        bundle = self.root / "bundle.safetensors"
        save_file(
            {"a": np.arange(4, dtype=np.uint8).reshape(2, 2), "b": np.arange(4, dtype=np.uint8)},
            str(bundle),
        )
        reader = AssetReader(self.root)
        for name in ("missing", "a", "b"):
            with self.assertRaises(ValueError):
                reader.read(dict(file=bundle.name, tensor=name, sha256="0" * 64))


if __name__ == "__main__":
    unittest.main()
