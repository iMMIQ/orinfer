"""Verify vector-generated RoPE gather indices, tails and changed-input Graphs."""

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
    from tools.operators.ascend.build import compile_kernel, operand
    from tools.operators.ascend.model_ops import partial_rope

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    if not args.compile_only:
        torch.npu.set_device(args.device)
    device = "cpu" if args.compile_only else "npu"
    capacity = 2049
    reports = []
    for heads, dimension, inverse in ((1, 512, False), (8, 512, True), (32, 128, False)):
        x = torch.randn(capacity, heads, dimension).bfloat16().to(device)
        phase = torch.randn(capacity, 32)
        cosine, sine = phase.cos().to(device), phase.sin().to(device)
        outputs = [
            torch.empty(capacity + 1, heads, dimension, dtype=torch.bfloat16, device=device)
            for _ in range(3)
        ]
        entries, libraries = [], []
        for index, (vector, batched) in enumerate(((False, False), (True, False), (True, True))):
            params = dict(
                rows=capacity,
                heads=heads,
                dimension=dimension,
                inverse=inverse,
                vector_indices=vector,
                batch_heads=batched,
            )
            path = compile_kernel(
                partial_rope(**dict(params, rows=row_count(capacity, True))),
                "model_ops.partial_rope",
                [
                    operand("x", "bf16", [capacity, heads, dimension]),
                    operand("cosine", "f32", [capacity, 32]),
                    operand("sine", "f32", [capacity, 32]),
                    operand("out", "bf16", [capacity, heads, dimension], "write"),
                ],
                params,
                args.output / "cache",
                row_capacity=capacity,
            )
            if args.compile_only:
                continue
            lib = ctypes.CDLL(str(path.parent / "kernel.so"))
            libraries.append(lib)
            call = lib.orinfer_ascend_kernel_v2
            call.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            call.restype = ctypes.c_int32
            ptrs = (ctypes.c_void_p * 4)(
                x.data_ptr(), cosine.data_ptr(), sine.data_ptr(), outputs[index].data_ptr()
            )
            entries.append((call, ptrs))
        if args.compile_only:
            continue
        for rows in (1, 2, 17, 31, 129, 2048, 2049):
            for out in outputs:
                out.fill_(17)

            def run(which):
                call, ptrs = entries[which]
                status = call(ptrs, 4, rows, torch.npu.current_stream()._as_parameter_)
                if status:
                    raise RuntimeError(f"RoPE launch failed: {status}")

            def check():
                torch.npu.synchronize()
                for candidate in outputs[1:]:
                    torch.testing.assert_close(candidate[:rows], outputs[0][:rows], atol=0, rtol=0)
                for out in outputs:
                    assert bool((out[rows:] == 17).all()), "RoPE wrote tail"
                    torch.testing.assert_close(
                        out[:rows, :, :-64], x[:rows, :, :-64], atol=0, rtol=0
                    )

            run(0)
            run(1)
            run(2)
            check()
            times, graphs = [], []
            for i in (0, 1, 2):
                timing, graph = npu_benchmark(lambda: run(i), 3, 1)
                times.append(timing["median_ms"])
                graphs.append(graph)
            x.neg_()
            cosine.neg_()
            for graph in graphs:
                graph.replay()
            check()
            reports.append(
                dict(
                    rows=rows, heads=heads, dimension=dimension, inverse=inverse, milliseconds=times
                )
            )
            print(reports[-1], flush=True)
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
