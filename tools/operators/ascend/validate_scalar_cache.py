"""Verify Scalar regroup output through a DMA consumer in the same Graph."""

import argparse
import ctypes
import json
from pathlib import Path
from tools.build.files import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=7)
    args = p.parse_args()
    import torch
    import torch_npu  # noqa: F401
    import tilelang.language as T
    from kernels.model.rows import row_count
    from tools.operators.ascend import routing
    from tools.operators.ascend.build import compile_kernel, operand
    from tools.model.deepseek_v41.tune import npu_benchmark

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    model = json.loads(args.plan.read_text())
    packet = next(p for p in model["ranks"][0]["packets"] if p["rows"] > 1)
    for step in packet["body"]:
        if step["op"] != "kernel":
            continue
        old = args.plan.parent / step["manifest"]
        spec = json.loads(old.read_text())
        if spec["op"] == "routing.regroup":
            break
    else:
        raise ValueError("Plan has no Scalar regroup reference")
    params = dict(spec["parameters"])
    cap = params.pop("_row_capacity")
    limit = params.pop("_grid_limit", 64)
    params.pop("rows")
    m = row_count(cap, True)
    new = compile_kernel(
        routing.regroup(m, **params),
        spec["op"],
        spec["operands"],
        dict(rows=cap, **params),
        args.output / "cache",
        row_capacity=cap,
        grid_limit=limit,
    )

    @T.prim_func
    def consume(
        gather: T.Tensor((m * 6,), "int32"),
        restore: T.Tensor((m * 8,), "int32"),
        counts: T.Tensor((48,), "int64"),
        seen_gather: T.Tensor((m * 6,), "int32"),
        seen_restore: T.Tensor((m * 8,), "int32"),
        seen_counts: T.Tensor((48,), "int64"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            values = T.alloc_ub((256,), "int32")
            groups = T.alloc_ub((48,), "int64")
            if vid == 0:
                for i in T.serial(T.ceildiv(m * 6, 256)):
                    valid = T.min(256, m * 6 - i * 256)
                    T.copy(gather[i * 256 : i * 256 + valid], values[:valid])
                    T.copy(values[:valid], seen_gather[i * 256 : i * 256 + valid])
                    T.set_flag("MTE3", "MTE2", 7)
                    T.wait_flag("MTE3", "MTE2", 7)
                for i in T.serial(T.ceildiv(m * 8, 256)):
                    valid = T.min(256, m * 8 - i * 256)
                    T.copy(restore[i * 256 : i * 256 + valid], values[:valid])
                    T.copy(values[:valid], seen_restore[i * 256 : i * 256 + valid])
                    T.set_flag("MTE3", "MTE2", 7)
                    T.wait_flag("MTE3", "MTE2", 7)
                T.copy(counts, groups)
                T.copy(groups, seen_counts)

    specs = [
        operand(
            n,
            "i64" if "counts" in n else "i32",
            [48] if "counts" in n else [cap * (6 if "gather" in n else 8)],
            "write" if n.startswith("seen") else "read",
        )
        for n in ("gather", "restore", "counts", "seen_gather", "seen_restore", "seen_counts")
    ]
    copy = compile_kernel(
        consume,
        "scalar_cache_consumer",
        specs,
        dict(rows=cap),
        args.output / "cache",
        row_capacity=cap,
    )
    tensors = [
        torch.empty(cap, 8, dtype=torch.int32, device="npu"),
        torch.empty(cap * 6, dtype=torch.int32, device="npu"),
        torch.empty(cap * 8, dtype=torch.int32, device="npu"),
        torch.empty(48, dtype=torch.int64, device="npu"),
        torch.empty(cap + 1, dtype=torch.int32, device="npu"),
    ]
    observed = [torch.empty_like(t) for t in tensors[1:4]]
    libraries = []

    def entry(path):
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
        return call

    writers = [entry(old), entry(new)]
    reader = entry(copy)
    pointer = (ctypes.c_void_p * 5)(*[t.data_ptr() for t in tensors])
    copyp = (ctypes.c_void_p * 6)(
        *[t.data_ptr() for t in [tensors[1], tensors[2], tensors[3], *observed]]
    )
    reports = []
    for rows in (1, 2, 16, 17, 32, 129, 2048):
        if rows > cap:
            continue
        counts = []
        for index, writer in enumerate(writers):

            def run():
                stream = torch.npu.current_stream()._as_parameter_
                if writer(pointer, 5, rows, stream) or reader(copyp, 6, rows, stream):
                    raise RuntimeError("Chained Scalar cache launch failed")

            _, graph = npu_benchmark(run, 1, 1)
            bad = 0
            for repetition in range(5):
                ids = torch.randint(0, 384, (rows, 8), dtype=torch.int32)
                ids[:, 6:] = -1
                tensors[0][:rows].copy_(ids)
                for t in tensors[1:]:
                    t.fill_(-123)
                graph.replay()
                torch.npu.synchronize()
                want_g = torch.full((rows * 6,), -1, dtype=torch.int32)
                want_r = torch.full((rows, 8), -1, dtype=torch.int32)
                want_c = torch.zeros(48, dtype=torch.int64)
                cursor = 0
                for expert in range(48):
                    for row, lane in (ids[:, :6] == params["base"] + expert).nonzero().tolist():
                        want_g[cursor] = row
                        want_r[row, lane] = cursor
                        want_c[expert] += 1
                        cursor += 1
                mismatch = sum(
                    int((actual[: len(want)] != want).sum())
                    for actual, want in zip(
                        [
                            observed[0].cpu(),
                            observed[1][: rows * 8].cpu().reshape(rows, 8),
                            observed[2].cpu(),
                        ],
                        [want_g, want_r, want_c],
                        strict=True,
                    )
                )
                bad += mismatch
                if index == 1:
                    assert mismatch == 0, (
                        f"Chained regroup mismatch rows={rows}, repeat={repetition}, bad={mismatch}"
                    )
            counts.append(bad)
        reports.append(dict(rows=rows, old_mismatches=counts[0], coherent_mismatches=counts[1]))
        print(reports[-1], flush=True)
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
