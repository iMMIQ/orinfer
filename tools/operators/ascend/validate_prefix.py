"""Check bounded expert scans, flags, padding and changed-input Graph replay."""

import argparse
import ctypes
from pathlib import Path

from tools.build.files import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.capacity <= 4096:
        parser.error("capacity must be in 2..4096")
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.tune import npu_benchmark
    from tools.operators.ascend.build import compile_kernel, operand
    from tools.operators.ascend.routing_parallel import prefix

    torch.manual_seed(20261002)
    torch.set_num_threads(4)
    if not args.compile_only:
        torch.npu.set_device(args.device)
    capacity = args.capacity
    size = 48 * ((capacity + 31) // 32)
    specs = [
        operand("counts", "i32", [size]),
        operand("offsets", "i32", [size], "write"),
        operand("groups", "i64", [48], "write"),
        operand("errors", "i32", [capacity + 1], "read_write"),
    ]
    libraries, entries, reports = [], {}, []
    for pad in (False, True):
        for vector in (False, True):
            parameters = dict(rows=capacity, pad_last=pad, scan_rows=capacity if vector else 0)
            path = compile_kernel(
                prefix(**dict(parameters, rows=row_count(capacity, True))),
                "routing_parallel.prefix",
                specs,
                parameters,
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
            entries[pad, vector] = call
    if args.compile_only:
        return
    counts = torch.empty(size, dtype=torch.int32, device="npu")
    offsets = torch.empty(size + 8, dtype=torch.int32, device="npu")
    groups = torch.empty(49, dtype=torch.int64, device="npu")
    errors = torch.empty(capacity + 2, dtype=torch.int32, device="npu")
    pointers = (ctypes.c_void_p * 4)(*[t.data_ptr() for t in (counts, offsets, groups, errors)])
    lengths = sorted(
        {
            r
            for r in (1, 2, 17, 31, 32, 33, 127, 129, 511, 2048, 2049, capacity - 1, capacity)
            if r <= capacity
        }
    )
    for rows in lengths:
        chunks = (rows + 31) // 32
        for kind in ("random", "empty", "full", "flags"):
            experts = torch.randint(0, 384, (rows, 6))
            if kind == "empty":
                experts.fill_(48)
            elif kind == "full":
                experts.zero_()
            buckets = experts * chunks + torch.arange(rows)[:, None] // 32
            original = torch.bincount(buckets[experts < 48], minlength=48 * chunks).int()
            original_errors = torch.zeros(rows + 1, dtype=torch.int32)
            original_errors[-1] = 23
            if kind == "flags":
                original_errors[0] = 17
                if rows > 1:
                    original_errors[rows - 1] = -2
            for pad in (False, True):
                for vector in (False, True):
                    counts[: original.numel()].copy_(original)
                    offsets.fill_(17)
                    groups.fill_(17)
                    errors.fill_(17)
                    errors[: rows + 1].copy_(original_errors)
                    call = entries[pad, vector]

                    def run():
                        status = call(pointers, 4, rows, torch.npu.current_stream()._as_parameter_)
                        if status:
                            raise RuntimeError(f"Expert prefix launch failed: {status}")

                    def check():
                        torch.npu.synchronize()
                        cumulative = original.cumsum(0)
                        expected_groups = original.reshape(48, chunks).sum(1)
                        if pad:
                            expected_groups[-1] += rows * 6 - cumulative[-1]
                        expected_errors = original_errors.clone()
                        expected_errors[rows] = cumulative[-1]
                        if bool((original_errors[1:rows] != 0).any()):
                            expected_errors[0] = 1
                        for actual, wanted in (
                            (offsets, (cumulative - original).int()),
                            (groups, expected_groups),
                            (errors, expected_errors),
                        ):
                            values = actual.cpu()
                            torch.testing.assert_close(
                                values[: wanted.numel()], wanted, atol=0, rtol=0
                            )
                            assert bool((values[wanted.numel() :] == 17).all()), "Prefix wrote tail"

                    run()
                    check()
                    timing, graph = npu_benchmark(run, 3, 1)
                    # Change groups and per-chunk offsets without recapturing.
                    original = original.flip(0).contiguous()
                    counts[: original.numel()].copy_(original)
                    errors[: rows + 1].copy_(original_errors)
                    graph.replay()
                    check()
                    reports.append(
                        dict(
                            rows=rows,
                            kind=kind,
                            pad=pad,
                            vector=vector,
                            median_ms=timing["median_ms"],
                        )
                    )
                    print(reports[-1], flush=True)
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
