"""Check explicit A2 mask synchronization, integer outputs and Graph replay."""

import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.build.files import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--capacity", type=int, default=2049)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 4224, 8448])
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--store-unroll", type=int, nargs="+", default=[1, 8, 16, 32])
    args = parser.parse_args()
    if not 2 <= args.capacity <= 4096 or any(n < 128 or n % 128 for n in args.lengths):
        parser.error("Require row capacity 2..4096 and positive 128-aligned lengths")
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.tune import npu_benchmark
    from tools.operators.ascend.attention_prefill import selected_mask
    from tools.operators.ascend.build import LOWER_LOCK, compile_kernel, operand
    from tools.operators.ascend.indexer_cube import causal_logits

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    if not args.compile_only:
        torch.npu.set_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    capacity = args.capacity
    m = row_count(capacity, True)
    # An explicit stage cannot silently acquire an automatically split producer.
    split_specs = [
        operand("query", "i8", [capacity * 32, 128]),
        operand("query_scale", "f16", [capacity, 32]),
        operand("keys", "i8", [128, 128]),
        operand("key_scale", "f16", [128]),
        operand("weights", "bf16", [capacity, 32]),
        operand("indices", "i32", [capacity, 128]),
        operand("out", "f32", [capacity, 128], "write"),
        operand("workspace", "i32", [capacity * 32, 128], "scratch"),
    ]
    try:
        compile_kernel(
            causal_logits(m, 128).with_attr("orinfer.explicit_sync", True),
            "indexer_cube.causal_logits",
            split_specs,
            dict(rows=capacity, length=128),
            args.output / "guard-cache",
            row_capacity=capacity,
        )
    except ValueError as error:
        if "single native stage" not in str(error):
            raise
    else:
        raise AssertionError("Explicit synchronization accepted multiple native stages")
    atomic_json(args.output / "synchronization-guard.json", dict(rejected_split_stages=True))
    configs = [dict(explicit_sync=False, store_unroll=1)] + [
        dict(explicit_sync=True, store_unroll=u) for u in sorted(set(args.store_unroll))
    ]
    reports = []
    for length in args.lengths:
        specs = [
            operand("indices", "i32", [capacity, 512]),
            operand("mask", "i32", [capacity, length], "write"),
        ]

        def build(config):
            params = dict(rows=capacity, length=length, **config)
            with LOWER_LOCK:
                function = selected_mask(m, length, **config)
            path = compile_kernel(
                function,
                "attention_prefill.selected_mask",
                specs,
                params,
                args.output / "cache",
                row_capacity=capacity,
            )
            print(dict(length=length, config=config, manifest=str(path)), flush=True)
            return path

        with ThreadPoolExecutor(max_workers=3) as pool:
            paths = list(pool.map(build, configs))
        if args.compile_only:
            continue
        ids = torch.empty(capacity, 512, dtype=torch.int32, device="npu")
        outs = [torch.empty(capacity + 1, length, dtype=torch.int32, device="npu") for _ in paths]
        libraries = [ctypes.CDLL(str(path.parent / "kernel.so")) for path in paths]
        entries, pointers = [], []
        for library, out in zip(libraries, outs, strict=True):
            entry = library.orinfer_ascend_kernel_v2
            entry.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            entry.restype = ctypes.c_int32
            entries.append(entry)
            pointers.append((ctypes.c_void_p * 2)(ids.data_ptr(), out.data_ptr()))
        rows_to_check = [r for r in (1, 2, 17, 31, 32, 128, 129, 2048, capacity) if r <= capacity]
        for rows in sorted(set(rows_to_check)):
            for pattern in ("random", "invalid", "duplicate", "extreme"):
                source = torch.randint(-3, length + 3, (capacity, 512), dtype=torch.int32)
                if pattern == "invalid":
                    source.fill_(-1)
                elif pattern == "duplicate":
                    source[:, :256] = 0
                    source[:, 256:] = length - 1
                elif pattern == "extreme":
                    source[:] = torch.tensor(
                        [-2147483648, 2147483647, -1, 0, length - 1, length, length + 1, 1],
                        dtype=torch.int32,
                    ).repeat(64)
                ids.copy_(source)
                for out in outs:
                    out.fill_(17)

                def run(index):
                    status = entries[index](
                        pointers[index], 2, rows, torch.npu.current_stream()._as_parameter_
                    )
                    if status:
                        raise RuntimeError(f"Mask native launch failed: {status}")

                def check():
                    keys = source[:rows].long()
                    keys = torch.where((keys >= 0) & (keys < length), keys, length)
                    expected = torch.zeros(rows, length + 1, dtype=torch.int32)
                    expected.scatter_(1, keys, 1)
                    expected = expected[:, :length].contiguous().npu()
                    for config, out in zip(configs, outs, strict=True):
                        torch.testing.assert_close(
                            out[:rows],
                            expected,
                            atol=0,
                            rtol=0,
                            msg=f"{pattern} rows={rows} length={length} config={config}",
                        )
                        assert bool((out[rows:] == 17).all()), "Mask wrote the row tail"

                for index in range(len(entries)):
                    run(index)
                check()
                graphs, times = [], []
                for index in range(len(entries)):
                    timing, graph = npu_benchmark(lambda: run(index), 3, 1)
                    times.append(timing["median_ms"])
                    graphs.append(graph)
                source = (source.long().add(11).remainder(length + 5) - 2).int()
                ids.copy_(source)
                for graph in graphs:
                    graph.replay()
                check()
                source = source.roll(1, 0).flip(1)
                ids.copy_(source)
                for graph in graphs:
                    graph.replay()
                check()
                report = dict(
                    length=length,
                    rows=rows,
                    pattern=pattern,
                    configs=configs,
                    milliseconds=times,
                    exact=True,
                    changed_input_graph=True,
                )
                reports.append(report)
                print(report, flush=True)
    if not args.compile_only:
        atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
