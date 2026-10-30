"""Check shared head-tiled Cube attention against small-head calls and FP64 math."""

import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

from tools.build.files import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", type=int, nargs="+", default=[8, 17, 64])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 7, 9, 17, 65])
    parser.add_argument("--head-tiles", type=int, nargs="+", choices=[16, 32, 64], default=[16])
    parser.add_argument("--benchmark-rows", type=int, nargs="+", default=[])
    parser.add_argument("--compile-workers", type=int, default=4)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--device", type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.compile_workers <= 8:
        parser.error("compile workers must be in1..8")
    if not 1 <= min(args.heads) <= max(args.heads) <= 64:
        parser.error("heads must be in1..64")
    if not 1 <= min(args.rows + args.benchmark_rows) <= max(args.rows + args.benchmark_rows) <= 256:
        parser.error("query rows must be in1..256")
    import torch
    from kernels.model.rows import row_count
    from tools.operators.ascend import attention_cube
    from tools.operators.ascend.build import LOWER_LOCK, compile_kernel, operand
    from tools.operators.ascend.geometry import physical_shapes

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    capacity = max(args.rows + args.benchmark_rows)
    dtype = {"bfloat16": "bf16", "float32": "f32", "int32": "i32"}
    kernels = {}
    # Small-head calls are an independent execution partition of the same
    # arithmetic, with identical K128 accumulation and BF16 boundaries.
    heads = sorted(set(args.heads) | {8} | {h % 8 for h in args.heads if h % 8})
    configs = [
        (selected, head, name, head_tile)
        for selected in [128, 640]
        for head in heads
        for name, head_tile in [("softmax", 16)]
        + list(product(["qk", "pv"], args.head_tiles if head > 16 else [16]))
    ]

    def compile_one(key):
        selected, head, name, head_tile = key
        params = dict(rows=capacity, heads=head, selected=selected)
        if name != "softmax":
            params["head_tile"] = head_tile
        if name == "pv":
            params["tile_d"] = 256
        with LOWER_LOCK:
            function = getattr(attention_cube, name)(**dict(params, rows=row_count(capacity, True)))
            tensors = [function.buffer_map[p] for p in function.params if p in function.buffer_map]
            physical = physical_shapes(function, capacity)
        writes = dict(qk={"scores"}, softmax={"probabilities"}, pv={"output", "workspace"})[name]
        specs = [
            operand(
                t.name, dtype[str(t.dtype)], shape, "read_write" if t.name in writes else "read"
            )
            for t, shape in zip(tensors, physical, strict=True)
        ]
        manifest = compile_kernel(
            function,
            "attention_cube." + name,
            specs,
            params,
            args.output / "cache",
            row_capacity=capacity,
        )
        return key, manifest

    with ThreadPoolExecutor(max_workers=args.compile_workers) as pool:
        kernels.update(pool.map(compile_one, configs))
    if args.compile_only:
        atomic_json(args.output / "compiled.json", [str(p) for p in kernels.values()])
        return
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    entries = {}
    for key, path in kernels.items():
        library = ctypes.CDLL(str((path.parent / "kernel.so").resolve()))
        fn = library.orinfer_ascend_kernel_v2
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int32
        entries[key] = library, fn

    def launch(key, values, rows):
        pointers = (ctypes.c_void_p * len(values))(*[x.data_ptr() for x in values])
        status = entries[key][1](
            pointers, len(values), rows, torch.npu.current_stream()._as_parameter_
        )
        if status:
            raise RuntimeError(f"{key}: C ABI returned {status}")

    def buffers(selected, head):
        padded = (head + 15) // 16 * 16
        return [
            torch.empty((capacity, padded, selected), dtype=torch.float32, device="npu"),
            torch.empty((capacity, padded, selected), dtype=torch.bfloat16, device="npu"),
            torch.empty((capacity, head, 512), dtype=torch.bfloat16, device="npu"),
            torch.empty((capacity, padded, 512), dtype=torch.float32, device="npu"),
        ]

    reports, measurements = [], []
    for selected in [128, 640]:
        for head, head_tile in product(args.heads, args.head_tiles):
            query = torch.randn((capacity, head, 512)).bfloat16().npu()
            kv = torch.randn((capacity, selected, 512)).bfloat16().npu()
            indices = torch.arange(selected).repeat(capacity, 1).int()
            indices[:, selected // 2 :] = -1
            indices[0, :] = -1  # sink-only row, including no visible keys
            indices = indices.npu()
            sink = torch.randn(head).float().npu()
            scores, probability, output, workspace = buffers(selected, head)

            def chain(q, h, state, count, sinks, tile=16):
                s, p, o, w = state
                launch((selected, h, "qk", tile if h > 16 else 16), [q, kv, s], count)
                launch((selected, h, "softmax", 16), [s, indices, sinks, p], count)
                launch((selected, h, "pv", tile if h > 16 else 16), [p, kv, o, w], count)

            # Stable input/output addresses across changing data and lengths.
            for count in args.rows:
                chain(query, head, [scores, probability, output, workspace], count, sink, head_tile)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    chain(
                        query,
                        head,
                        [scores, probability, output, workspace],
                        count,
                        sink,
                        head_tile,
                    )
                for phase in range(3):
                    if phase:
                        query.neg_()
                        kv.mul_(0.5)
                        sink.add_(0.25)
                    output.fill_(17)
                    graph.replay()
                    torch.npu.synchronize()
                    actual = output[:count].cpu()
                    actual_scores = scores[:count, :head].cpu()
                    actual_probability = probability[:count, :head].cpu()
                    expected = []
                    for first in range(0, head, 8):
                        h = min(8, head - first)
                        state = buffers(selected, h)
                        q = query[:, first : first + h].contiguous()
                        chain(q, h, state, count, sink[first : first + h].contiguous())
                        torch.npu.synchronize()
                        assert torch.equal(
                            actual_scores[:, first : first + h], state[0][:count, :h].cpu()
                        )
                        assert torch.equal(
                            actual_probability[:, first : first + h], state[1][:count, :h].cpu()
                        )
                        expected.append(state[2][:count].cpu())
                    assert torch.equal(actual, torch.cat(expected, dim=1))
                    q = query[:count].cpu().double()
                    k = kv[:count].cpu().double()
                    score = q @ k.transpose(-1, -2) / 512**0.5
                    score.masked_fill_((indices[:count].cpu() < 0)[:, None], -torch.inf)
                    sink_logits = sink.cpu().double()[None, :, None].expand(count, -1, 1)
                    reference_p = torch.softmax(torch.cat([score, sink_logits], -1), -1)[
                        ..., :-1
                    ].bfloat16()
                    reference = (reference_p.double() @ k).bfloat16()
                    torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)
                reports.append(
                    dict(
                        selected=selected,
                        heads=head,
                        rows=count,
                        head_tile=head_tile,
                        phases=3,
                        exact_head_partitions=True,
                    )
                )
                atomic_json(args.output / "validation.json", reports)
            if args.benchmark_rows:
                from tools.model.deepseek_v41.tune import npu_benchmark
                from tools.model.tuning import measure

                probability.fill_(1.0 / selected)
                for count in args.benchmark_rows:
                    for name, values in [
                        ("qk", [query, kv, scores]),
                        ("pv", [probability, kv, output, workspace]),
                    ]:
                        key = selected, head, name, head_tile if head > 16 else 16
                        milliseconds = measure(
                            lambda: launch(key, values, count),
                            budget_ms=2,
                            max_repetitions=32,
                            benchmark=npu_benchmark,
                        )
                        measurements.append(
                            dict(
                                selected=selected,
                                heads=head,
                                head_tile=head_tile,
                                rows=count,
                                op="attention_cube." + name,
                                milliseconds=milliseconds,
                                manifest=str(kernels[key]),
                            )
                        )
                atomic_json(
                    args.output / "measurements.json",
                    dict(
                        source_sha256=file_hash(Path(attention_cube.__file__)),
                        validator_sha256=file_hash(Path(__file__)),
                        validation=reports,
                        measurements=measurements,
                    ),
                )
    print(
        f"Verified {len(reports)} head/row geometries with changed-input Graph replay", flush=True
    )


if __name__ == "__main__":
    main()
