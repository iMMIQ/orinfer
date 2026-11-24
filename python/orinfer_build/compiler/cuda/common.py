"""Offline CUDA operator helpers for Orin (`sm_87`) and generic discrete GPUs (`sm_80`+).

Runtime kernels remain independent of Torch. On Jetson AGX Orin (`sm_87`),
behavior is unchanged; on non-Orin SM80+ GPUs (`sm_80`, `sm_86`, `sm_89`,
`sm_90`, `sm_100`, `sm_120`), compilation automatically targets the live GPU or
`ORINFER_CUDA_ARCH`.
"""

from functools import partial, wraps
import os
from pathlib import Path
import shutil
import statistics
import time

import tilelang
import torch

from orinfer_build.common.files import file_hash, write_json as write_json
from orinfer_build.common.paths import ROOT
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

__all__ = [
    "DEFAULT_CUDA_ARCH",
    "SEED",
    "SUPPORTED_CUDA_ARCHES",
    "benchmark",
    "configure",
    "cuda_arch",
    "cuda_jit",
    "cuda_target",
    "environment",
    "error",
    "export_kernel",
    "host_arch",
    "identity",
    "is_generic_cuda_arch",
    "is_orin_arch",
    "orin_jit",
    "package_target",
    "parse_cuda_sm",
    "validate_cuda_arch",
    "write_json",
]

SEED = 20261002


def configure(*, require_sm87: bool | None = None):
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cap = torch.cuda.get_device_capability()
    strict_orin = (
        require_sm87 if require_sm87 is not None else os.environ.get("ORINFER_REQUIRE_SM87") == "1"
    )
    if strict_orin:
        assert cap == (8, 7), f"Expected SM87, got {cap}"
    else:
        arch = f"sm_{cap[0]}{cap[1]}"
        if parse_cuda_sm(arch) is None:
            raise AssertionError(
                f"Unsupported CUDA capability {cap}; expected one of {SUPPORTED_CUDA_ARCHES}"
            )


def cuda_jit(function=None, *, pass_configs=None, arch=None, out_idx=()):
    """Compile a TileLang kernel for the active CUDA target (`sm_87` on Orin, live SM otherwise)."""
    if function is None:
        return partial(cuda_jit, pass_configs=pass_configs, arch=arch, out_idx=out_idx)
    compiled_by_arch = {}

    @wraps(function)
    def build(*args, **kwargs):
        resolved_arch = cuda_arch(arch)
        if resolved_arch not in compiled_by_arch:
            compiled_by_arch[resolved_arch] = tilelang.jit(
                out_idx=list(out_idx),
                execution_backend="nvrtc",
                target={"kind": "cuda", "arch": resolved_arch},
                pass_configs=pass_configs,
            )(function)
        kernel = compiled_by_arch[resolved_arch](*args, **kwargs)
        kernel.adapter.kernels = dict(kernel.adapter.kernels)

        def launch(*inputs, stream=None):
            if stream is None:
                stream = torch.cuda.current_stream(inputs[0].device).cuda_stream
            return kernel.adapter.func(*inputs, stream=stream)

        kernel.torch_function = launch
        return kernel

    return build


def orin_jit(function=None, *, pass_configs=None):
    """Backward-compatible JIT decorator resolving the active CUDA target dynamically."""
    return cuda_jit(function, pass_configs=pass_configs)


def identity(path):
    path = Path(path)
    return {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size}


def error(actual, reference):
    actual, reference = actual.float(), reference.float()
    delta = actual - reference
    norm = float(reference.norm())
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    return {
        "finite": finite,
        "relative_l2": float(delta.norm()) / max(norm, 1e-30),
        "reference_l2": norm,
        "max_abs": float(delta.abs().max()),
        "rms_abs": float(delta.square().mean().sqrt()),
    }


def export_kernel(kernel, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "kernel.cu").write_text(kernel.get_kernel_source())
    (output / "host.txt").write_text(Path(kernel.adapter.lib_generator.pypath).read_text())
    shutil.copyfile(kernel.adapter.lib_generator.libpath, output / "kernel.cubin")
    return {
        "files": [identity(output / name) for name in ("kernel.cu", "host.txt", "kernel.cubin")],
        "symbols": list(kernel.adapter.function_names),
        "abi_note": "Use actual host.txt argument order/config, never PrimFunc order",
    }


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
    return {
        "median_ms": statistics.median(samples),
        "trials_ms": samples,
        "repetitions": repetitions,
        "graph_capture_s": capture_s,
        "calls_per_replay": calls_per_replay,
        "timing_scope": "CUDA-event graph replay; reset/work within run included",
    }, graph


def environment():
    cap = torch.cuda.get_device_capability()
    return {
        "torch": torch.__version__,
        "tilelang": tilelang.__version__,
        "cuda": torch.version.cuda,
        "device": str(torch.cuda.get_device_properties(0)),
        "sm": list(cap),
        "target": package_target(),
        "generic_cuda": is_generic_cuda_arch(),
        "seed": SEED,
        "python_dependencies_sha256": file_hash(ROOT / "tools/environments/cuda/uv.lock"),
        "output_root": os.environ.get("ORINFER_OPERATOR_OUTPUT"),
    }
