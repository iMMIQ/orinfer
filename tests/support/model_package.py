"""Small data execution package for CPU packaging and release tests."""

import hashlib

from orinfer_build.packaging.compact import compact_package


def data_package(root, payload=b"fixture cubin bytes", **metadata):
    source = root / "compiler"
    source.mkdir(parents=True, exist_ok=True)
    (source / "kernel.cubin").write_bytes(payload)
    identity = dict(file="kernel.cubin", sha256=hashlib.sha256(payload).hexdigest())
    package = dict(
        architecture="qwen3_5",
        compute_policy="int8_quality",
        target="sm_87",
        config_signature={},
        prefill_profiles=[],
        buffer_contracts=[],
        toolchain={},
        kernels=[
            dict(
                name="test",
                symbol="main",
                grid=[1, 1, 1],
                block=[32, 1, 1],
                shared_memory_bytes=0,
                cooperative=False,
                args=[],
                module=identity,
                source=identity,
                host_abi=identity,
            )
        ],
    )
    package.update(metadata)
    cache = root / "packages"
    return cache / compact_package(source, package, cache)
