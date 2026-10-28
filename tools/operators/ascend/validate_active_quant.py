"""Check routed-prefix SwiGLU/quantization against the full-capacity arithmetic."""

import argparse
import ctypes
from pathlib import Path

from tools.build.files import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=7)
    p.add_argument("--compile-only", action="store_true")
    args = p.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.tune import npu_benchmark
    from tools.operators.ascend import model_ops
    from tools.operators.ascend.build import compile_kernel, operand

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    if not args.compile_only:
        torch.npu.set_device(args.device)
    capacity, width = 2048, 2304
    slots = [
        operand("packed", "bf16", [capacity * 6, width * 2]),
        operand("out", "i8", [capacity * 6, width], "write"),
        operand("scales", "f32", [capacity * 6], "write"),
    ]
    paths = []
    for active in (False, True):
        factory = model_ops.swiglu_packed_quant_active if active else model_ops.swiglu_packed_quant
        paths.append(
            compile_kernel(
                factory(row_count(capacity, True) * 6, width),
                "model_ops." + factory.__name__,
                slots + ([operand("active_counts", "i32", [capacity + 1])] if active else []),
                dict(rows=capacity * 6, width=width),
                args.output / "cache",
                row_capacity=capacity,
            )
        )
    if args.compile_only:
        return
    libraries = [ctypes.CDLL(str(path.parent / "kernel.so")) for path in paths]
    entries = [lib.orinfer_ascend_kernel_v2 for lib in libraries]
    for entry in entries:
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        entry.restype = ctypes.c_int32
    packed = (torch.randn(capacity * 6, width * 2, device="npu") * 3).bfloat16()
    outputs = [
        torch.empty(capacity * 6 + 1, width, dtype=torch.int8, device="npu") for _ in range(2)
    ]
    scales = [torch.empty(capacity * 6 + 1, device="npu") for _ in range(2)]
    counts = torch.empty(capacity + 1, dtype=torch.int32, device="npu")
    pointers = [
        (ctypes.c_void_p * (3 + i))(
            *(
                [packed.data_ptr(), outputs[i].data_ptr(), scales[i].data_ptr()]
                + ([counts.data_ptr()] if i else [])
            )
        )
        for i in range(2)
    ]
    reports = []
    for tokens in (1, 2, 17, 31, 129, 2048):
        for active_rows in sorted({0, 1, min(tokens * 6, 17), tokens * 6 // 8, tokens * 6}):
            for t in outputs + scales:
                t.fill_(17)
            counts.fill_(-1)
            counts[tokens] = active_rows

            def run(which):
                status = entries[which](
                    pointers[which],
                    len(pointers[which]),
                    tokens,
                    torch.npu.current_stream()._as_parameter_,
                )
                if status:
                    raise RuntimeError(f"Active quantization launch: {status}")

            def check():
                torch.npu.synchronize()
                torch.testing.assert_close(
                    outputs[1][:active_rows], outputs[0][:active_rows], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    scales[1][:active_rows], scales[0][:active_rows], atol=0, rtol=0
                )
                assert bool((outputs[1][active_rows:] == 17).all()), "Active quant wrote the tail"
                assert bool((scales[1][active_rows:] == 17).all()), (
                    "Active quant scale wrote the tail"
                )

            run(0)
            run(1)
            check()
            timings, graphs = [], []
            for i in (0, 1):
                timing, graph = npu_benchmark(lambda: run(i), 2, 1)
                timings.append(timing["median_ms"])
                graphs.append(graph)
            packed.neg_()
            for graph in graphs:
                graph.replay()
            check()
            timed_active_rows = active_rows
            # Replay observes a changed live bound without recapture.
            smaller = active_rows // 2
            outputs[1].fill_(17)
            scales[1].fill_(17)
            counts[tokens] = smaller
            graphs[1].replay()
            active_rows = smaller
            check()
            counts[tokens] = timed_active_rows
            graphs[1].replay()
            active_rows = timed_active_rows
            check()
            reports.append(
                dict(
                    tokens=tokens,
                    timed_active_rows=timed_active_rows,
                    replayed_active_rows=[smaller, timed_active_rows],
                    milliseconds=timings,
                )
            )
            print(reports[-1], flush=True)
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
