"""Check fused postquant SWA staging against INT8 scatter and selected attention."""

import argparse
import ctypes
from pathlib import Path

from tools.build.files import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compressed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-long", action="store_true")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--device", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.tune import npu_benchmark
    from tools.model.tuning import measure
    from tools.operators.ascend import kv_int8, attention_fused
    from tools.operators.ascend.build import compile_kernel, operand
    from tools.operators.ascend.geometry import physical_shapes

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    capacity, storage, sources, heads, context, long_capacity = 256, 4096, 2048, 64, 262144, 131072
    selected = 640 if args.compressed else 128
    libraries, entries, manifests = [], [], []
    attention_params = dict(
        rows=capacity,
        capacity=context,
        compressed_capacity=long_capacity,
        swa_storage_rows=storage,
        heads=heads,
        compressed=args.compressed,
        tile_slots=16,
        reuse_long=args.reuse_long,
    )
    stages = [
        (
            "kv_int8.scatter",
            kv_int8.scatter,
            dict(rows=sources, table_rows=storage, logical_capacity=context),
            sources,
        ),
        (
            "kv_int8.scatter_staged",
            kv_int8.scatter_staged,
            dict(rows=sources, table_rows=storage, logical_capacity=context),
            sources,
        ),
        (
            "attention_fused.selected_int8",
            attention_fused.selected_int8,
            attention_params,
            capacity,
        ),
        (
            "attention_fused.selected_int8",
            attention_fused.selected_int8,
            dict(attention_params, staged_swa=True),
            capacity,
        ),
    ]
    if args.reuse_long:
        stages.append(
            (
                "attention_fused.selected_int8",
                attention_fused.selected_int8,
                dict(attention_params, reuse_long=False),
                capacity,
            )
        )
    for op, factory, params, bound in stages:
        print("compiling", op, params, flush=True)
        f = factory(**dict(params, rows=row_count(bound, True)))
        tensors = [f.buffer_map[p] for p in f.params if p in f.buffer_map]
        kinds = {
            "int8": "i8",
            "int32": "i32",
            "float16": "f16",
            "float32": "f32",
            "bfloat16": "bf16",
        }
        writes = {
            "cache",
            "scales",
            "errors",
            "staged",
            "output",
            "kv_workspace",
            "mask_workspace",
            "score_workspace",
            "probability_workspace",
            "output_workspace",
        }
        specs = [
            operand(
                t.name, kinds[str(t.dtype)], shape, "read_write" if t.name in writes else "read"
            )
            for t, shape in zip(tensors, physical_shapes(f, bound), strict=True)
        ]
        path = compile_kernel(
            f, op, specs, params, args.cache or args.output / "cache", row_capacity=bound
        )
        lib = ctypes.CDLL(str((path.parent / "kernel.so").resolve()))
        libraries.append(lib)
        fn = lib.orinfer_ascend_kernel_v2
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int32
        entries.append(fn)
        manifests.append(dict(path=str(path), sha256=file_hash(path)))
    source = torch.randn(sources, 8, 64, device="npu").bfloat16()
    positions = torch.empty(sources, dtype=torch.int32, device="npu")
    cache = torch.randint(-127, 128, (storage, 8, 64), dtype=torch.int8, device="npu")
    scales = (torch.rand(storage, 8, device="npu") * 0.02 + 0.001).half()
    cache2, scales2 = cache.clone(), scales.clone()
    errors = torch.empty(sources, dtype=torch.int32, device="npu")
    errors2 = torch.empty_like(errors)
    borrowed = torch.empty(64, 256, 128, dtype=torch.int32, device="npu")
    staged = borrowed.view(torch.bfloat16).flatten()[: storage * 512].view(storage, 8, 64)
    staged.fill_(17)
    query = torch.randn(capacity, heads, 512, device="npu").bfloat16()
    long = torch.randint(-127, 128, (long_capacity, 8, 64), dtype=torch.int8, device="npu")
    long_scales = (torch.rand(long_capacity, 8, device="npu") * 0.02 + 0.001).half()
    swa_ids = torch.empty(capacity, 128, dtype=torch.int32, device="npu")
    long_ids = torch.randint(0, 16384, (capacity, 512), dtype=torch.int32, device="npu")
    long_ids[:, ::17] = -1
    sink = torch.randn(heads, device="npu")

    def scratch():
        return [
            torch.full((capacity + 1, heads, 512), 17, dtype=torch.bfloat16, device="npu"),
            torch.empty(capacity, selected, 512, dtype=torch.bfloat16, device="npu"),
            torch.empty(capacity, selected, dtype=torch.int32, device="npu"),
            torch.empty(capacity, heads, selected, device="npu"),
            torch.empty(capacity, heads, selected, dtype=torch.bfloat16, device="npu"),
            torch.empty(capacity, heads, 512, device="npu"),
        ]

    a, b = scratch(), scratch()

    def call(index, tensors, rows):
        ptr = (ctypes.c_void_p * len(tensors))(*[v.data_ptr() for v in tensors])
        status = entries[index](ptr, len(tensors), rows, torch.npu.current_stream()._as_parameter_)
        if status:
            raise RuntimeError(f"kernel{index}: {status}")

    def old(rows):
        call(0, [source, positions, cache, scales, errors], sources)
        call(2, [query, cache, scales, long, long_scales, swa_ids, long_ids, sink] + a, rows)

    def new(rows):
        call(1, [source, positions, cache2, scales2, errors2, staged], sources)
        call(
            3,
            [query, cache2, scales2, long, long_scales, swa_ids, long_ids, sink] + b + [staged],
            rows,
        )

    def data(prefix):
        positions.copy_(torch.arange(sources, dtype=torch.int32, device="npu") + prefix)
        swa_ids.copy_(
            torch.arange(128, dtype=torch.int32, device="npu")[None, :]
            + prefix
            + torch.arange(capacity, dtype=torch.int32, device="npu")[:, None] * 8
            - 127
        )
        swa_ids[:, ::19] = -1

    reports = []
    if args.reuse_long:
        data(0)
        call(4, [query, cache, scales, long, long_scales, swa_ids, long_ids, sink] + a, capacity)
        b[1].copy_(a[1])
        b[2].copy_(a[2])
    for rows in [1, 17, capacity]:
        data(0)
        old(rows)
        new(rows)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            old(rows)
            new(rows)
        for prefix in [0, 4000, 32768]:
            borrowed.fill_(31 + prefix)
            source.neg_()
            query.neg_()
            data(prefix)
            a[0].fill_(17)
            b[0].fill_(17)
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(cache, cache2), "persistent INT8 differs"
            assert torch.equal(scales, scales2), "FP16 scales differ"
            assert torch.equal(errors, errors2), "error flags differ"
            stage_ids = torch.arange(max(0, prefix - 127), prefix + sources, device="npu") % storage
            expected_stage = (
                cache[stage_ids].float() * scales[stage_ids].float()[:, :, None]
            ).bfloat16()
            stage_error = (staged[stage_ids].float() - expected_stage.float()).abs().max()
            print(dict(rows=rows, prefix=prefix, stage_max_abs=float(stage_error)), flush=True)
            torch.testing.assert_close(staged[stage_ids], expected_stage, atol=0, rtol=0)
            for index in [1, 2, 4, 0]:
                if not torch.equal(a[index][:rows], b[index][:rows]):
                    print(
                        dict(
                            buffer=index,
                            maximum=float(
                                (a[index][:rows].float() - b[index][:rows].float()).abs().max()
                            ),
                            old=a[index][0].flatten()[-24:].float().cpu().tolist(),
                            new=b[index][0].flatten()[-24:].float().cpu().tolist(),
                        ),
                        flush=True,
                    )
                torch.testing.assert_close(a[index][:rows], b[index][:rows], atol=0, rtol=0)
            assert bool((b[0][rows:] == 17).all()), "row tail changed"
            ids = torch.arange(max(0, prefix - 127), prefix + sources, device="npu") % storage
            wanted = (cache[ids].float() * scales[ids].float()[:, :, None]).bfloat16()
            torch.testing.assert_close(staged[ids], wanted, atol=0, rtol=0)
            reports.append(
                dict(
                    rows=rows,
                    prefix=prefix,
                    exact=True,
                    changed_graph=True,
                    tail=True,
                    persistent_int8=True,
                )
            )
            print(reports[-1], flush=True)
        graph.reset()
    data(32768)
    timings = []
    for rows in [17, 256]:
        t = []
        for run in [old, new]:
            t.append(
                measure(lambda: run(rows), budget_ms=8, max_repetitions=9, benchmark=npu_benchmark)
            )
        timings.append(
            dict(
                rows=rows,
                source_rows=sources,
                baseline_ms=t[0],
                staged_ms=t[1],
                change_pct=100 * (t[1] / t[0] - 1),
            )
        )
        print(timings[-1], flush=True)
    atomic_json(
        args.output / "result.json",
        dict(manifests=manifests, validation=reports, timing=timings, compressed=args.compressed),
    )


if __name__ == "__main__":
    main()
