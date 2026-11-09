"""Check candidate sets, compact INT8 scores, causal tails and changed-input Graphs."""

import argparse
import ctypes
from pathlib import Path

from orinfer_build.common.files import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--length", type=int, default=262144)
    parser.add_argument("--chunk-size", type=int, choices=(512, 1024, 2048, 4096), default=1024)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from kernels.ascend import candidate, indexer_cube, indexer_fused, topk
    from kernels.shared.rows import row_count
    from orinfer_build.compiler.ascend.build import compile_kernel, operand
    from orinfer_build.contracts.ascend.geometry import physical_shapes

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    capacity, length, selected = 17, args.length, 2048
    if length < 32768 or length % 128:
        parser.error("Require a 128-aligned capacity of at least 32768")
    kinds = {"int8": "i8", "float16": "f16", "bfloat16": "bf16", "float32": "f32", "int32": "i32"}
    libraries = []

    def compile_call(factory, name, params, writes):
        function = factory(**dict(params, rows=row_count(capacity, True)))
        tensors = [function.buffer_map[p] for p in function.params if p in function.buffer_map]
        shapes = physical_shapes(function, capacity)
        specs = [
            operand(
                t.name, kinds[str(t.dtype)], shape, "read_write" if t.name in writes else "read"
            )
            for t, shape in zip(tensors, shapes, strict=True)
        ]
        path = compile_kernel(
            function, name, specs, params, args.output / "cache", row_capacity=capacity
        )
        library = ctypes.CDLL(str(path.parent / "kernel.so"))
        libraries.append(library)
        entry = library.orinfer_ascend_kernel_v2
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        entry.restype = ctypes.c_int32

        def call(values, rows):
            pointers = (ctypes.c_void_p * len(values))(*[t.data_ptr() for t in values])
            status = entry(pointers, len(values), rows, torch.npu.current_stream()._as_parameter_)
            if status:
                raise RuntimeError(f"{name}: {status}")

        return call

    print("compile candidate selector", flush=True)
    select = compile_call(
        candidate.select_blocks,
        "candidate.select_blocks",
        dict(rows=capacity, length=length),
        {"out"},
    )
    print("compile compact indexer", flush=True)
    params = dict(
        rows=capacity,
        length=length,
        tile_s=128,
        tile_rows=1,
        vector_mask=True,
        direct_sum=True,
        workspace_tiles=64,
        length_input=True,
    )
    compact = compile_call(
        indexer_cube.causal_logits,
        "indexer_cube.causal_logits",
        dict(params, candidate_blocks=selected),
        {"out", "workspace"},
    )
    baseline = compile_call(
        indexer_cube.causal_logits, "indexer_cube.causal_logits", params, {"out", "workspace"}
    )
    print("compile fused candidate indexer", flush=True)
    fused = compile_call(
        indexer_fused.causal_topk,
        "indexer_fused.causal_topk",
        dict(
            rows=capacity,
            length=length,
            tile_rows=1,
            candidate_blocks=selected,
            chunk_size=args.chunk_size,
        ),
        {"candidates", "cube_workspace", "score_workspace"},
    )
    local = compile_call(
        topk.local, "topk.local", dict(rows=capacity, length=selected * 8), {"out"}
    )
    restore = compile_call(candidate.restore, "candidate.restore", dict(rows=capacity), {"indices"})
    reference_finalize = compile_call(
        topk.finalize, "topk.finalize", dict(rows=capacity), {"indices", "values"}
    )
    mapped_finalize = compile_call(
        candidate.finalize, "candidate.finalize", dict(rows=capacity), {"indices", "values"}
    )

    def merge_chain(sources, leaf_size):
        entries = []
        if sources in (16, 32, 64):
            fn = compile_call(
                topk.merge_tree,
                "topk.merge_tree",
                dict(rows=capacity, sources=sources, leaf_size=leaf_size),
                {"out"},
            )
            entries.append((fn, torch.empty((capacity + 1, 1, 1024), device="npu"), True))
        else:
            while sources > 1:
                fn = compile_call(
                    topk.merge, "topk.merge", dict(rows=capacity, sources=sources), {"out"}
                )
                sources = (sources + 3) // 4
                entries.append(
                    (fn, torch.empty((capacity + 1, sources, 1024), device="npu"), False)
                )

        def execute(source, counts, rows):
            for fn, output, lengths in entries:
                fn([source, counts, output] if lengths else [source, output], rows)
                source = output
            return source

        return execute

    reference_merge = merge_chain(4, 4096)
    fused_merge = merge_chain(16384 // args.chunk_size, args.chunk_size)
    scores = torch.empty((capacity, length), dtype=torch.float32, device="npu")
    # Query positions inside an owner are monotonically increasing, as required
    # by the common bounded score schedule's last-row extent.
    counts_cpu = torch.tensor(
        sorted(
            min(length, n)
            for n in [
                0,
                1,
                7,
                8,
                2048,
                2049,
                16383,
                16384,
                16385,
                17001,
                32768,
                50000,
                100000,
                100001,
                length - 1,
                length,
                length,
            ]
        ),
        dtype=torch.int32,
    )
    counts = counts_cpu.npu()
    blocks = torch.full((capacity, selected + 8), -99, dtype=torch.int32, device="npu")
    q = torch.randint(-64, 64, (capacity * 32, 128), dtype=torch.int8, device="npu")
    q[14 * 32 : 15 * 32].zero_()  # Stable selection must also preserve full-row ties.
    qs = (torch.rand((capacity, 32), device="npu") * 0.02 + 0.001).half()
    keys = torch.randint(-64, 64, (length, 128), dtype=torch.int8, device="npu")
    ks = (torch.rand((length,), device="npu") * 0.02 + 0.001).half()
    weights = torch.randn((capacity, 32), device="npu").bfloat16()
    full = torch.full_like(scores, float("nan"))
    actual = torch.full((capacity, selected * 8), float("nan"), device="npu")
    scratch = torch.empty((capacity * 32, 8192), dtype=torch.int32, device="npu")
    ids = torch.empty((capacity, 512), dtype=torch.int32, device="npu")

    partial_reference = torch.full((capacity + 1, 4, 1024), 71.0, device="npu")
    partial_fused = torch.full((capacity + 1, 16384 // args.chunk_size, 1024), 71.0, device="npu")
    cube = torch.empty((64, 32, 128), dtype=torch.int32, device="npu")
    tile_scores = torch.empty((64, 1, args.chunk_size), dtype=torch.float32, device="npu")

    reference_indices = torch.empty((capacity, 512), dtype=torch.int32, device="npu")
    reference_values = torch.empty((capacity, 512), dtype=torch.float32, device="npu")
    fused_indices = torch.empty_like(reference_indices)
    fused_values = torch.empty_like(reference_values)

    def candidate_pipeline(rows):
        fused([q, qs, keys, ks, weights, blocks, partial_fused, cube, tile_scores], rows)
        final = fused_merge(partial_fused, counts, rows)
        mapped_finalize([blocks, final, fused_indices, fused_values], rows)

    def run(rows, diagnostic=False):
        select([scores, counts, blocks], rows)
        if diagnostic:
            torch.npu.synchronize()
            print("selector complete", flush=True)
        baseline([q, qs, keys, ks, weights, counts, full, scratch], rows)
        if diagnostic:
            torch.npu.synchronize()
            print("baseline complete", flush=True)
        compact([q, qs, keys, ks, weights, blocks, actual, scratch], rows)
        if diagnostic:
            torch.npu.synchronize()
            print("compact complete", flush=True)
        local([actual, counts, partial_reference], rows)
        ref_final = reference_merge(partial_reference, counts, rows)
        reference_finalize([ref_final, reference_indices, reference_values], rows)
        restore([blocks, reference_indices], rows)
        candidate_pipeline(rows)
        restore([blocks, ids], rows)

    reports = []
    for rows in (1, 3, 17):
        for changed in (False, True):
            reference_scores = torch.stack(
                [torch.randperm(length).float() - length for _ in range(capacity)]
            )
            for row, live in enumerate(counts_cpu.tolist()):
                reference_scores[row, live:] = 1e9
                if live % 8:
                    reference_scores[row, live // 8 * 8 : live] = -1e6
            scores.copy_(reference_scores)
            compact_ids = (torch.arange(512, dtype=torch.int32) * 31).expand(capacity, -1).clone()
            compact_ids[:, -1] = selected * 8 - 1
            if changed:
                q.neg_()
                compact_ids[:, ::3] = -1
            ids.copy_(compact_ids)
            if not changed:
                run(rows, diagnostic=True)
                ids.copy_(compact_ids)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    run(rows)
                graph.replay()
            else:
                graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(
                fused_indices[:rows].cpu(), reference_indices[:rows].cpu(), atol=0, rtol=0
            )
            torch.testing.assert_close(
                fused_values[:rows].cpu(), reference_values[:rows].cpu(), atol=0, rtol=0
            )
            assert bool((partial_fused[capacity].cpu() == 71.0).all()), (
                "Fused indexer wrote the guard row"
            )
            b, f, a, restored = blocks.cpu(), full.cpu(), actual.cpu(), ids.cpu()
            for row in range(rows):
                live = counts_cpu[row].item()
                source = reference_scores[row].clone()
                source[live:] = -float("inf")
                maxima = source.reshape(-1, 8).max(dim=1).values
                n = min(selected, (live + 7) // 8)
                expected_blocks = torch.topk(maxima, n).indices.to(torch.int32)
                valid = b[row, 8:][b[row, 8:] >= 0]
                assert torch.equal(valid.sort().values, expected_blocks.sort().values), (
                    "blocks",
                    row,
                    live,
                    b[row, :20].tolist(),
                    valid[:20].tolist(),
                    expected_blocks[:20].tolist(),
                )
                absolute = (b[row, 8:, None] * 8 + torch.arange(8)).flatten()
                count = min(live, selected * 8)
                valid = (absolute[:count] >= 0) & (absolute[:count] < live)
                torch.testing.assert_close(
                    a[row, :count][valid], f[row, absolute[:count][valid]], atol=0, rtol=0
                )
                assert bool(torch.isneginf(a[row, :count][~valid]).all())
                expected = compact_ids[row].clone()
                ok = expected >= 0
                expected[ok] = absolute[expected[ok]].to(torch.int32)
                expected[(expected >= live) | (expected < 0)] = -1
                assert torch.equal(restored[row], expected), (
                    "restore",
                    row,
                    live,
                    restored[row, :20].tolist(),
                    expected[:20].tolist(),
                )
            reports.append(dict(rows=rows, changed_graph=changed, passed=True))
            print(reports[-1], flush=True)
    # Non-finite/padded pairs must be rejected before forming a gather address.
    poisoned = torch.zeros((capacity, 1024), device="npu")
    invalid = torch.tensor([float("nan"), float("inf"), -float("inf"), 1e30, -1, 16384])
    poisoned[:, 1 : 2 * len(invalid) : 2] = invalid.npu()
    mapped_finalize([blocks, poisoned, fused_indices, fused_values], capacity)
    torch.npu.synchronize()
    assert bool((fused_indices[:, : len(invalid)].cpu() == -1).all())
    reports.append(dict(poisoned_indices=True, passed=True))
    from orinfer_build.tuning.ascend import npu_benchmark

    counts.fill_(100000)
    scores.normal_()
    select([scores, counts, blocks], capacity)
    timings = {}
    for rows in (1, 3, 17):
        timing, _ = npu_benchmark(lambda: candidate_pipeline(rows), repetitions=7, warmup=2)
        timings[str(rows)] = timing
    atomic_json(
        args.output / "result.json",
        dict(cases=reports, index_chunk=args.chunk_size, timings=timings),
    )


if __name__ == "__main__":
    main()
