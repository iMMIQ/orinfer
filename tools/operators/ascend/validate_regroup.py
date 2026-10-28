"""Check stable parallel A2 regrouping, padding, invalid routes and Graph replay."""

import argparse
import ctypes
from pathlib import Path

from tools.build.files import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=7)
    args = p.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.tune import npu_benchmark
    from tools.operators.ascend import routing_parallel as ops
    from tools.operators.ascend.build import compile_kernel, operand

    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    capacity, experts, base = 2049, 48, 144
    chunks = (capacity + 31) // 32
    device = "npu"
    ids = torch.empty((capacity, 8), dtype=torch.int32, device=device)
    gather = torch.empty(capacity * 6 + 8, dtype=torch.int32, device=device)
    restore = torch.empty((capacity + 1, 8), dtype=torch.int32, device=device)
    groups = torch.empty(49, dtype=torch.int64, device=device)
    errors = torch.empty(capacity + 2, dtype=torch.int32, device=device)
    positions = torch.empty((experts * chunks * 256,), dtype=torch.int32, device=device)
    counts = torch.empty((experts * chunks,), dtype=torch.int32, device=device)
    offsets = torch.empty_like(counts)
    specs = {
        "indices": operand("indices", "i32", [capacity, 8]),
        "gather": operand("gather", "i32", [capacity * 6], "write"),
        "restore": operand("restore", "i32", [capacity, 8], "write"),
        "errors": operand("errors", "i32", [capacity + 1], "read_write"),
        "positions": operand("positions", "i32", [experts, chunks, 256], "read_write"),
        "counts": operand("counts", "i32", [experts, chunks], "read_write"),
        "offsets": operand("offsets", "i32", [experts, chunks], "read_write"),
        "groups": operand("groups", "i64", [experts], "write"),
    }
    tensors = dict(
        indices=ids,
        gather=gather,
        restore=restore,
        errors=errors,
        positions=positions,
        counts=counts,
        offsets=offsets,
        groups=groups,
    )
    libraries, calls = [], []
    m = row_count(capacity, True)
    for name, params, names in [
        ("clear", dict(experts=384), ["indices", "gather", "restore", "errors"]),
        ("local", dict(base=base, experts=experts), ["indices", "positions", "counts"]),
        ("prefix", dict(experts=experts), ["counts", "offsets", "groups", "errors"]),
        ("pack", dict(experts=experts), ["positions", "counts", "offsets", "gather"]),
        ("restore", dict(base=base, experts=experts), ["indices", "offsets", "restore"]),
    ]:
        path = compile_kernel(
            getattr(ops, name)(m, **params),
            "routing_parallel." + name,
            [specs[n] for n in names],
            dict(rows=capacity, **params),
            args.output / "cache",
            row_capacity=capacity,
        )
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
        pointers = (ctypes.c_void_p * len(names))(*[tensors[n].data_ptr() for n in names])
        calls.append((call, pointers, len(names)))

    reports = []
    for rows in (1, 2, 17, 31, 32, 33, 127, 128, 129, 2048, 2049):
        for kind in ("random", "duplicates", "empty", "invalid"):
            original = torch.randint(0, 384, (rows, 8), dtype=torch.int32)
            if kind == "duplicates":
                original[:, :6] = base + 3
            elif kind == "empty":
                original[:, :6] = 0
            elif kind == "invalid":
                original[0, 0] = -1
                original[-1, 5] = 384
            # Padded lanes must never create work.
            original[:, 6:] = base
            ids[:rows].copy_(original)
            for t in (gather, restore, groups, errors):
                t.fill_(17)

            def run():
                stream = torch.npu.current_stream()._as_parameter_
                for call, pointers, count in calls:
                    status = call(pointers, count, rows, stream)
                    if status:
                        raise RuntimeError(f"Parallel regroup native launch: {status}")

            def check():
                torch.npu.synchronize()
                want_g = torch.full((rows * 6,), -1, dtype=torch.int32)
                want_r = torch.full((rows, 8), -1, dtype=torch.int32)
                want_c = torch.zeros(experts, dtype=torch.int64)
                cursor = 0
                for expert in range(experts):
                    matches = (original[:, :6] == base + expert).nonzero().tolist()
                    want_c[expert] = len(matches)
                    for row, lane in matches:
                        want_g[cursor] = row
                        want_r[row, lane] = cursor
                        cursor += 1
                for t, want in ((gather, want_g), (restore, want_r), (groups, want_c)):
                    actual = t.cpu()
                    if not torch.equal(actual[: len(want)], want):
                        print(
                            dict(
                                rows=rows,
                                kind=kind,
                                indices=original.tolist(),
                                actual=actual[: len(want)].tolist(),
                                wanted=want.tolist(),
                                counts=counts[: experts * ((rows + 31) // 32)].cpu().tolist(),
                                offsets=offsets[: experts * ((rows + 31) // 32)].cpu().tolist(),
                            ),
                            flush=True,
                        )
                    torch.testing.assert_close(actual[: len(want)], want, atol=0, rtol=0)
                    assert bool((actual[len(want) :] == 17).all()), "Regroup wrote a tail"
                actual_e = errors.cpu()
                assert actual_e[0] == int(kind == "invalid")
                assert actual_e[rows] == cursor
                assert bool((actual_e[rows + 1 :] == 17).all())

            run()
            check()
            latency, graph = npu_benchmark(run, 2, 1)
            original = original.flip(0).contiguous()
            ids[:rows].copy_(original)
            graph.replay()
            check()
            reports.append(dict(rows=rows, kind=kind, latency_ms=latency["median_ms"]))
            print(reports[-1], flush=True)
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
