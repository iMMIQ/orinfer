"""Check causal INT8 Cube indexing against the independent FP32 reference."""

import argparse
import ctypes
from pathlib import Path

from tools.build.files import atomic_json


def check_reference_scale():
    """FP16 halfway cases with independently known even subnormal results."""
    import torch
    from tools.model.deepseek_v41.validate import reference

    weights = torch.zeros((2, 32), dtype=torch.bfloat16)
    weights[:, 0] = torch.tensor([129, -191]) * 2**-19
    values = [
        torch.ones((2, 32, 128), dtype=torch.int8),
        torch.ones((2, 32), dtype=torch.float16),
        torch.ones((1, 128), dtype=torch.int8),
        torch.ones((1,), dtype=torch.float16),
        weights,
        torch.zeros((2, 1), dtype=torch.int32),
    ]
    expected = (torch.tensor([64, -96]) * (128 * 2**-24)).reshape(2, 1)
    torch.testing.assert_close(
        reference("index_quantized", values, {})[0], expected, atol=0, rtol=0
    )


def selection_check(capacity, length, root):
    """Compare the actual TopK chain with poisoned and canonical score padding."""
    import torch
    from kernels.model.rows import row_count
    from tools.operators.ascend import topk
    from tools.operators.ascend.build import compile_kernel, operand
    from tools.operators.ascend.geometry import physical_shapes

    stages = []
    sources = (length + 4095) // 4096
    definitions = [("local", dict(rows=capacity, length=length))]
    while sources > 1:
        definitions.append(("merge", dict(rows=capacity, sources=sources)))
        sources = (sources + 3) // 4
    for name, params in definitions:
        function = getattr(topk, name)(**{**params, "rows": row_count(capacity, True)})
        shapes = physical_shapes(function, capacity)
        buffers = [function.buffer_map[p] for p in function.params if p in function.buffer_map]
        specs = [
            operand(
                b.name,
                "i32" if str(b.dtype) == "int32" else "f32",
                shape,
                "write" if i == len(buffers) - 1 else "read",
            )
            for i, (b, shape) in enumerate(zip(buffers, shapes, strict=True))
        ]
        path = compile_kernel(function, "topk." + name, specs, params, root, row_capacity=capacity)
        library = ctypes.CDLL(str(path.parent / "kernel.so"))
        entry = library.orinfer_ascend_kernel_v2
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        entry.restype = ctypes.c_int32
        stages.append((library, entry, torch.empty(shapes[-1], device="npu")))

    def select(scores, lengths, rows):
        source = scores
        for index, (_, entry, output) in enumerate(stages):
            tensors = [source, lengths, output] if index == 0 else [source, output]
            pointers = (ctypes.c_void_p * len(tensors))(*[t.data_ptr() for t in tensors])
            status = entry(pointers, len(tensors), rows, torch.npu.current_stream()._as_parameter_)
            if status:
                raise RuntimeError(f"TopK padding check launch: {status}")
            source = output
        return source[:rows].cpu().clone()

    def check(scores, expected, lengths, rows):
        counts = torch.zeros(capacity, dtype=torch.int32, device="npu")
        counts[:rows].copy_(lengths.flatten())
        actual = select(scores, counts, rows)
        canonical = torch.full((capacity, length), -float("inf"), device="npu")
        canonical[:rows].copy_(expected)
        torch.testing.assert_close(actual, select(canonical, counts, rows), atol=0, rtol=0)

    return check


def check_length_producers(capacity, length, tested_rows, root):
    import torch
    from kernels.model.rows import row_count
    from tools.operators.ascend.build import compile_kernel, operand
    from tools.operators.ascend.model_ops import causal_indices

    for ratio in (1, 2):
        params = dict(rows=capacity, capacity=length * ratio, ratio=ratio, dense=False)
        function = causal_indices(**{**params, "rows": row_count(capacity, True)})
        specs = [
            operand("positions", "i32", [capacity], "read"),
            operand("indices", "i32", [capacity, 1], "write"),
            operand("lengths", "i32", [capacity], "write"),
        ]
        path = compile_kernel(
            function, "model_ops.causal_indices", specs, params, root, row_capacity=capacity
        )
        library = ctypes.CDLL(str(path.parent / "kernel.so"))
        entry = library.orinfer_ascend_kernel_v2
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        entry.restype = ctypes.c_int32
        positions = torch.arange(capacity, dtype=torch.int32).clamp(max=length * ratio - 1)
        device_positions = positions.npu()
        dummy = torch.full((capacity, 1), 99, dtype=torch.int32, device="npu")
        out = torch.full((capacity + 1,), 17, dtype=torch.int32, device="npu")
        pointers = (ctypes.c_void_p * 3)(
            device_positions.data_ptr(), dummy.data_ptr(), out.data_ptr()
        )
        for rows in tested_rows:
            out.fill_(17)
            status = entry(pointers, 3, rows, torch.npu.current_stream()._as_parameter_)
            if status:
                raise RuntimeError(f"Causal length producer launch: {status}")
            actual = out.cpu()
            torch.testing.assert_close(actual[:rows], (positions[:rows] + 1) // ratio)
            assert bool((actual[rows:] == 17).all()), "Length producer wrote the tail"
            assert bool((dummy.cpu() == 99).all()), "Length-only producer wrote dummy indices"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=7)
    p.add_argument("--workspace-tiles", type=int, choices=(0, 64), default=0)
    p.add_argument("--length-input", action="store_true")
    p.add_argument("--row-capacity", type=int, default=129)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 2, 17, 127, 128, 129])
    p.add_argument("--position-start", type=int, default=0)
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

    check_reference_scale()
    torch.manual_seed(20261002)
    torch.set_num_threads(4)
    if not args.compile_only:
        torch.npu.set_device(args.device)
    capacity = args.row_capacity
    if not 1 <= min(args.rows) <= max(args.rows) <= capacity:
        p.error("Invalid tested row extent")
    if args.position_start < 0:
        p.error("Causal positions must be nonnegative")
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
            workspace_tiles=args.workspace_tiles,
            length_input=args.length_input,
        )
        shapes = [
            ("query", "i8", [capacity, 32, 128], "read"),
            ("query_scale", "f16", [capacity, 32], "read"),
            ("keys", "i8", [length, 128], "read"),
            ("key_scale", "f16", [length], "read"),
            ("weights", "bf16", [capacity, 32], "read"),
            ("indices", "i32", [capacity, 1 if args.length_input else length], "read"),
            ("out", "f32", [capacity, length], "write"),
            (
                "workspace",
                "i32",
                [
                    capacity,
                    32,
                    min(
                        (length + args.tile_s - 1) // args.tile_s,
                        args.workspace_tiles or (length + args.tile_s - 1) // args.tile_s,
                    )
                    * args.tile_s,
                ],
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
            workspace_tiles=args.workspace_tiles,
            length_input=args.length_input,
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
        if args.length_input:
            check_length_producers(capacity, length, args.rows, args.output / "cache")
        check_topk = (
            selection_check(capacity, length, args.output / "cache") if args.length_input else None
        )
        ids = torch.arange(length).expand(capacity, -1)
        ids = torch.where(
            ids <= args.position_start + torch.arange(capacity)[:, None], ids, -1
        ).int()
        inputs = [
            torch.randint(-127, 128, (capacity, 32, 128), dtype=torch.int8),
            (torch.rand(capacity, 32) * 0.01).half(),
            torch.randint(-127, 128, (length, 128), dtype=torch.int8),
            (torch.rand(length) * 0.01).half(),
            torch.randn(capacity, 32).bfloat16(),
            ids,
        ]

        def device_indices():
            return (inputs[5] >= 0).sum(-1, keepdim=True).int() if args.length_input else inputs[5]

        tensors = [t.npu() for t in inputs[:5]] + [device_indices().npu()]
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
        for rows in args.rows:
            inputs[5] = ids.clone()
            tensors[5].copy_(device_indices())

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
                if args.length_input:
                    valid = inputs[5][:rows] >= 0
                    torch.testing.assert_close(
                        actual[:rows][valid], expected[valid], atol=0, rtol=0
                    )
                    assert bool(actual[:rows][~valid].isnan().all()), "Wrote outside live prefixes"
                    check_topk(tensors[6], expected, device_indices()[:rows], rows)
                else:
                    torch.testing.assert_close(actual[:rows], expected, atol=0, rtol=0)
                assert bool((actual[rows:] == 17).all()), "Cube indexer wrote the tail"

            tensors[6].fill_(17)
            if args.length_input:
                tensors[6][:rows].fill_(float("nan"))
            run()
            check()
            timing, graph = npu_benchmark(run, 7, 1)
            inputs[0].neg_()
            tensors[0].copy_(inputs[0])
            graph.replay()
            check()
            inputs[1].mul_(0.5)
            inputs[3].mul_(1.25)
            inputs[4].neg_()
            inputs[5] = torch.where(inputs[5] < 7, inputs[5], -1)
            for slot in (1, 3, 4, 5):
                tensors[slot].copy_(device_indices() if slot == 5 else inputs[slot])
            if args.length_input:
                tensors[6][:rows].fill_(float("nan"))
            graph.replay()
            check()
            if args.length_input:
                inputs[5].fill_(-1)
                tensors[5].zero_()
                tensors[6][:rows].fill_(float("nan"))
                graph.replay()
                check()
            reports.append(
                dict(
                    length=length,
                    rows=rows,
                    position_start=args.position_start,
                    changed_input_graph=True,
                    changed_scales_mask=True,
                    length_input=args.length_input,
                    graph_median_ms=timing["median_ms"],
                )
            )
            print(reports[-1], flush=True)
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
