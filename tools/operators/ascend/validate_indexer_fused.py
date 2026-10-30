"""Compare fused causal index candidates against the independent score/TopK chain."""

import argparse
import ctypes
from pathlib import Path

from tools.build.files import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=131072)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 17, 256])
    parser.add_argument("--tile-rows", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--device", type=int, default=7)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.tune import npu_benchmark
    from tools.model.tuning import measure
    from tools.operators.ascend import indexer_cube, indexer_fused, topk
    from tools.operators.ascend.build import compile_kernel, operand
    from tools.operators.ascend.geometry import physical_shapes

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    capacity, heads, length = max(args.rows), 32, args.length
    if not 2 <= capacity <= 256 or length < 8192:
        parser.error("Require capacity2..256 and length>=8192")
    libraries, entries, manifests = [], [], []
    for name, factory, params in [
        (
            "indexer_cube.causal_logits",
            indexer_cube.causal_logits,
            dict(
                rows=capacity,
                length=length,
                tile_rows=8,
                tile_s=128,
                vector_scales=True,
                vector_mask=True,
                direct_sum=True,
                workspace_tiles=64,
                length_input=True,
            ),
        ),
        ("topk.local", topk.local, dict(rows=capacity, length=length)),
        (
            "indexer_fused.causal_topk",
            indexer_fused.causal_topk,
            dict(rows=capacity, length=length, tile_rows=args.tile_rows),
        ),
    ]:
        print("compiling", name, flush=True)
        function = factory(**dict(params, rows=row_count(capacity, True)))
        tensors = [function.buffer_map[p] for p in function.params if p in function.buffer_map]
        shapes = physical_shapes(function, capacity)
        kinds = {
            "int8": "i8",
            "float16": "f16",
            "bfloat16": "bf16",
            "int32": "i32",
            "float32": "f32",
        }
        specs = [
            operand(
                t.name,
                kinds[str(t.dtype)],
                shape,
                "read_write"
                if t.name in {"out", "workspace", "candidates", "cube_workspace", "score_workspace"}
                else "read",
            )
            for t, shape in zip(tensors, shapes, strict=True)
        ]
        path = compile_kernel(
            function, name, specs, params, args.output / "cache", row_capacity=capacity
        )
        library = ctypes.CDLL(str((path.parent / "kernel.so").resolve()))
        libraries.append(library)
        fn = library.orinfer_ascend_kernel_v2
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int32
        entries.append(fn)
        manifests.append(dict(path=str(path), sha256=file_hash(path)))
    q = torch.randint(-64, 64, (capacity * heads, 128), dtype=torch.int8, device="npu")
    q.view(capacity, heads, 128)[::13].zero_()
    qs = (torch.rand(capacity, heads, device="npu") * 0.02 + 0.001).half()
    keys = torch.randint(-64, 64, (length, 128), dtype=torch.int8, device="npu")
    ks = (torch.rand(length, device="npu") * 0.02 + 0.001).half()
    weights = torch.randn(capacity, heads, device="npu").bfloat16()
    counts = torch.zeros((capacity, 1), dtype=torch.int32, device="npu")
    scores = torch.empty(capacity, length, device="npu")
    workspace = torch.empty(capacity * heads, 8192, dtype=torch.int32, device="npu")
    chunks = (length + 4095) // 4096
    before = torch.empty(capacity + 1, chunks, 1024, device="npu")
    after = torch.empty_like(before)
    cube = torch.empty(64, args.tile_rows * heads, 128, dtype=torch.int32, device="npu")
    local_scores = torch.empty(64, args.tile_rows, 4096, device="npu")
    shared = [q, qs, keys, ks, weights, counts]

    def call(index, tensors, rows):
        pointers = (ctypes.c_void_p * len(tensors))(*[t.data_ptr() for t in tensors])
        status = entries[index](
            pointers, len(tensors), rows, torch.npu.current_stream()._as_parameter_
        )
        if status:
            raise RuntimeError(f"Kernel{index} status{status}")

    def baseline(rows):
        call(0, shared + [scores, workspace], rows)
        call(1, [scores, counts, before], rows)

    def fused(rows):
        call(2, shared + [after, cube, local_scores], rows)

    reports = []
    for rows in args.rows:
        counts.zero_()
        baseline(rows)
        fused(rows)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            baseline(rows)
            fused(rows)
        for phase in range(3):
            q.neg_()
            ks.mul_(0.5)
            if rows == 1:
                end = [0, 4097, length][phase]
                counts[0, 0] = end
            else:
                maximum = [2049, 8192, min(length, 32769)][phase]
                counts[:rows, 0].copy_(torch.linspace(0, maximum, rows, device="npu").int())
            before.fill_(12345)
            after.fill_(12345)
            graph.replay()
            torch.npu.synchronize()
            actual, expected = after[:rows].cpu(), before[:rows].cpu()
            if not torch.equal(actual, expected):
                bad = (actual != expected).nonzero()[:12]
                print(
                    dict(
                        rows=rows,
                        phase=phase,
                        mismatch=[
                            dict(
                                at=x.tolist(),
                                actual=float(actual[tuple(x)]),
                                expected=float(expected[tuple(x)]),
                            )
                            for x in bad
                        ],
                    ),
                    flush=True,
                )
            assert torch.equal(actual, expected), (rows, phase, "candidate mismatch")
            assert bool((after[rows:] == 12345).all()), "Fused indexer wrote row tail"
            reports.append(
                dict(
                    rows=rows,
                    phase=phase,
                    exact_candidates=True,
                    changed_input_graph=True,
                    tail=True,
                )
            )
            print(reports[-1], flush=True)
        graph.reset()
    timings = []
    for context in [2048, 32768]:
        counts.fill_(min(length, context))
        values = []
        for fn in [baseline, fused]:
            values.append(
                measure(
                    lambda: fn(capacity), budget_ms=8, max_repetitions=7, benchmark=npu_benchmark
                )
            )
        timings.append(
            dict(
                rows=capacity,
                context=context,
                baseline_ms=values[0],
                fused_ms=values[1],
                change_pct=100 * (values[1] / values[0] - 1),
            )
        )
        print(timings[-1], flush=True)
    atomic_json(
        args.output / "result.json", dict(manifests=manifests, validation=reports, timing=timings)
    )


if __name__ == "__main__":
    main()
