"""Offline SM87 operator helpers; runtime kernels remain independent of Torch."""
import hashlib
import json
import os
import shutil
import statistics
import time
from functools import wraps
from pathlib import Path

import torch
import tilelang

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261002


def configure():
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_capability() == (8, 7)


def orin_jit(function):
    compiled = tilelang.jit(out_idx=[], execution_backend="nvrtc",
                            target={"kind": "cuda", "arch": "sm_87"})(function)

    @wraps(function)
    def build(*args, **kwargs):
        kernel = compiled(*args, **kwargs)
        kernel.adapter.kernels = dict(kernel.adapter.kernels)

        def launch(*inputs, stream=None):
            if stream is None:
                stream = torch.cuda.current_stream(inputs[0].device).cuda_stream
            return kernel.adapter.func(*inputs, stream=stream)

        kernel.torch_function = launch
        return kernel

    return build


def identity(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8)
                          .numpy().tobytes()).hexdigest()


def error(actual, reference):
    actual, reference = actual.float(), reference.float()
    delta = actual - reference
    norm = float(reference.norm())
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    return {"finite": finite, "relative_l2": float(delta.norm()) / max(norm, 1e-30),
            "reference_l2": norm, "max_abs": float(delta.abs().max()),
            "rms_abs": float(delta.square().mean().sqrt())}


def export_kernel(kernel, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "kernel.cu").write_text(kernel.get_kernel_source())
    (output / "host.txt").write_text(Path(kernel.adapter.lib_generator.pypath).read_text())
    shutil.copyfile(kernel.adapter.lib_generator.libpath, output / "kernel.cubin")
    return {"files": [identity(output / name) for name in
                      ("kernel.cu", "host.txt", "kernel.cubin")],
            "symbols": list(kernel.adapter.function_names),
            "abi_note": "Use actual host.txt argument order/config, never PrimFunc order"}


def benchmark(run, repetitions=20, warmup=3, calls_per_replay=1):
    """For stateful operations run must reset state or have separate immutable input."""
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    with torch.cuda.graph(graph):
        for _ in range(calls_per_replay):
            run()
    capture_s = time.perf_counter() - started
    samples = []
    for _ in range(3):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / (repetitions * calls_per_replay))
    return {"median_ms": statistics.median(samples), "trials_ms": samples,
            "repetitions": repetitions, "graph_capture_s": capture_s,
            "calls_per_replay": calls_per_replay,
            "timing_scope": "CUDA-event graph replay; reset/work within run included"}, graph


def environment():
    return {"torch": torch.__version__, "tilelang": tilelang.__version__,
            "cuda": torch.version.cuda, "device": str(torch.cuda.get_device_properties(0)),
            "sm": list(torch.cuda.get_device_capability()), "seed": SEED,
            "output_root": os.environ.get("ORIN_OPERATOR_OUTPUT")}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
