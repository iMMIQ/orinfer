"""Shared Ascend Graph measurement for offline tuning and validation."""

import ctypes
import json

from orinfer_build.common.files import file_hash
from orinfer_build.common.sources import component_identity
from orinfer_build.packaging.publication import source_path


def measurement_identity():
    return component_identity(
        [
            "orinfer_build.tuning.ascend",
            "orinfer_build.tuning.tuning",
            "orinfer_build.contracts.row_selection",
        ]
    )


def kernel_entry(manifest, library="kernel.so"):
    entry = ctypes.CDLL(str((manifest.parent / library).resolve())).orinfer_ascend_kernel_v2
    entry.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]
    entry.restype = ctypes.c_int32
    return entry


def verified_manifest(base, name, *, expected=None, compiled_like=None):
    """Verify a cached manifest and the native library identified by it."""
    path = source_path(base.resolve(), name)
    spec = json.loads(path.read_text())
    if expected is not None and spec != expected:
        return None
    if compiled_like is not None and any(
        spec[field] != compiled_like[field] for field in ("compiler", "source_sha256")
    ):
        return None
    library = spec["library"]
    if file_hash(source_path(path.parent, library["file"])) == library["sha256"]:
        return spec
    return None


def npu_benchmark(run, repetitions, warmup, *, calls_per_replay=1):
    """Time each call, amortizing host submission for independent projections."""
    import statistics
    import torch

    if type(calls_per_replay) is not int or not 1 <= calls_per_replay <= 64:
        raise ValueError("NPU replay call count must be in 1..64")
    for _ in range(warmup):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(calls_per_replay):
            run()
    for _ in range(2):
        graph.replay()
    torch.npu.synchronize()
    values = []
    for _ in range(repetitions):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / calls_per_replay)
    return dict(median_ms=statistics.median(values)), graph
