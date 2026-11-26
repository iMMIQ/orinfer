"""Validate persistent engram.cast_fp32 across non-aligned dynamic rows and changed-input Graphs."""

import argparse
import ctypes
from pathlib import Path

from orinfer_build.common.files import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=7)
    args = p.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from kernels.ascend import engram
    from kernels.shared.rows import row_count
    from orinfer_build.compiler.ascend.build import compile_kernel, operand
    from orinfer_build.tuning.ascend import npu_benchmark

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)

    width = 5120
    reports = []
    libraries = []
    for capacity, row_cases in (
        (32, (1, 7, 17, 32)),
        (256, (1, 17, 63, 129, 256)),
        (2048, (1, 17, 257, 513, 1025, 2048)),
    ):
        params = dict(rows=capacity, width=width, tile=2048)
        path = compile_kernel(
            engram.cast_fp32(row_count(capacity, True), width, tile=2048),
            "engram.cast_fp32",
            [
                operand("hidden", "bf16", [capacity * 4 * width]),
                operand("out", "f32", [capacity * 4 * width], "write"),
            ],
            params,
            args.output / "cache",
            row_capacity=capacity,
            grid_limit=20,
        )
        lib = ctypes.CDLL(str(path.parent / "kernel.so"))
        libraries.append(lib)
        entry = lib.orinfer_ascend_kernel_v2
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        entry.restype = ctypes.c_int32

        source = torch.randn(capacity, 4, width).bfloat16()
        device_x = source.npu()
        device_y = torch.empty(capacity, 4, width, dtype=torch.float32, device="npu")
        ptrs = (ctypes.c_void_p * 2)(device_x.data_ptr(), device_y.data_ptr())

        for rows in row_cases:
            device_y.fill_(17.0)

            def run(n=rows):
                status = entry(ptrs, 2, n, torch.npu.current_stream()._as_parameter_)
                if status:
                    raise RuntimeError(f"engram.cast_fp32 launch failed: {status}")

            def check(n=rows):
                torch.npu.synchronize()
                assert torch.equal(device_y[:n].cpu(), source[:n].float()), (capacity, n)
                if n < capacity:
                    assert bool((device_y[n:] == 17.0).all()), f"Tail overwritten: {(capacity, n)}"

            run()
            check()
            timing, graph = npu_benchmark(run, 3, 1)
            source.neg_()
            device_x.copy_(source)
            device_y.fill_(17.0)
            graph.replay()
            check()
            graph.reset()
            reports.append(
                dict(
                    capacity=capacity,
                    rows=rows,
                    median_ms=timing["median_ms"],
                    exact=True,
                    changed_graph=True,
                )
            )
            print(reports[-1], flush=True)
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
