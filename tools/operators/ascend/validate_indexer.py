"""Check causal INT8 Cube indexing against the independent FP32 reference."""

import argparse
import ctypes
from pathlib import Path

from tools.build.files import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=7)
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--tile-rows", type=int, default=1)
    p.add_argument("--tile-s", type=int, default=64)
    p.add_argument("--vector-mask", action="store_true")
    p.add_argument("--direct-sum", action="store_true")
    p.add_argument("--prefetch-inputs", action="store_true")
    p.add_argument("--lengths", type=int, nargs="+", default=[35, 128])
    args = p.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.tune import npu_benchmark
    from tools.model.deepseek_v41.validate import reference
    from tools.operators.ascend.build import compile_kernel, operand
    from tools.operators.ascend.indexer_cube import causal_logits

    torch.manual_seed(20261002)
    torch.set_num_threads(4)
    if not args.compile_only:
        torch.npu.set_device(args.device)
    capacity = 129
    reports = []
    for length in args.lengths:
        parameters = dict(
            rows=capacity,
            length=length,
            tile_s=args.tile_s,
            tile_rows=args.tile_rows,
            vector_mask=args.vector_mask,
            direct_sum=args.direct_sum,
            prefetch_inputs=args.prefetch_inputs,
        )
        shapes = [
            ("query", "i8", [capacity, 32, 128], "read"),
            ("query_scale", "f16", [capacity, 32], "read"),
            ("keys", "i8", [length, 128], "read"),
            ("key_scale", "f16", [length], "read"),
            ("weights", "bf16", [capacity, 32], "read"),
            ("indices", "i32", [capacity, length], "read"),
            ("out", "f32", [capacity, length], "write"),
            (
                "workspace",
                "i32",
                [capacity, 32, (length + args.tile_s - 1) // args.tile_s * args.tile_s],
                "scratch",
            ),
        ]
        specs = [operand(*s) for s in shapes]
        function = causal_logits(
            row_count(capacity, True),
            length,
            tile_s=args.tile_s,
            tile_rows=args.tile_rows,
            vector_mask=args.vector_mask,
            direct_sum=args.direct_sum,
            prefetch_inputs=args.prefetch_inputs,
        )
        path = compile_kernel(
            function,
            "indexer_cube.causal_logits",
            specs,
            parameters,
            args.output / "cache",
            row_capacity=capacity,
        )
        if args.compile_only:
            continue
        ids = torch.arange(length).expand(capacity, -1)
        ids = torch.where(ids <= torch.arange(capacity)[:, None], ids, -1).int()
        inputs = [
            torch.randint(-127, 128, (capacity, 32, 128), dtype=torch.int8),
            (torch.rand(capacity, 32) * 0.01).half(),
            torch.randint(-127, 128, (length, 128), dtype=torch.int8),
            (torch.rand(length) * 0.01).half(),
            torch.randn(capacity, 32).bfloat16(),
            ids,
        ]
        tensors = [t.npu() for t in inputs]
        tensors += [
            torch.empty(capacity + 1, length, device="npu"),
            torch.empty(shapes[-1][2], dtype=torch.int32, device="npu"),
        ]
        library = ctypes.CDLL(str(path.parent / "kernel.so"))
        entry = library.orinfer_ascend_kernel_v2
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        entry.restype = ctypes.c_int32
        pointers = (ctypes.c_void_p * len(tensors))(*[t.data_ptr() for t in tensors])
        for rows in (1, 2, 17, 127, 128, 129):
            inputs[5] = ids.clone()
            tensors[5].copy_(inputs[5])

            def run():
                status = entry(
                    pointers, len(tensors), rows, torch.npu.current_stream()._as_parameter_
                )
                if status:
                    raise RuntimeError(f"Cube indexer native launch: {status}")

            def check():
                torch.npu.synchronize()
                live = [t[:rows] if i in (0, 1, 4, 5) else t for i, t in enumerate(inputs)]
                expected = reference("index_quantized", live, parameters)[0]
                actual = tensors[6].cpu()
                torch.testing.assert_close(actual[:rows], expected, atol=0, rtol=0)
                assert bool((actual[rows:] == 17).all()), "Cube indexer wrote the tail"

            tensors[6].fill_(17)
            run()
            check()
            _, graph = npu_benchmark(run, 2, 1)
            inputs[0].neg_()
            tensors[0].copy_(inputs[0])
            graph.replay()
            check()
            inputs[1].mul_(0.5)
            inputs[3].mul_(1.25)
            inputs[4].neg_()
            inputs[5] = torch.where(inputs[5] < 7, inputs[5], -1)
            for slot in (1, 3, 4, 5):
                tensors[slot].copy_(inputs[slot])
            graph.replay()
            check()
            reports.append(
                dict(length=length, rows=rows, changed_input_graph=True, changed_scales_mask=True)
            )
            print(reports[-1], flush=True)
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
