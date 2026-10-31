"""Compare BF16 Fixpipe output with the shared scaled projection epilogue."""

import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.build.files import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.tune import npu_benchmark
    from tools.operators.ascend.build import LOWER_LOCK, compile_kernel, operand
    from tools.operators.ascend.dense import projection

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    if not args.compile_only:
        torch.npu.set_device(args.device)
    reports, libraries = [], []
    for capacity, n, k, tile_m, tile_n in [
        (32, 5120, 1024, 16, 64),
        (2049, 5120, 1024, 128, 128),
        (2049, 1024, 4096, 128, 128),
    ]:
        padded = (capacity + tile_m - 1) // tile_m * tile_m
        slots = [
            operand("activation", "bf16", [capacity, k]),
            operand("weight", "bf16", [n, k]),
            operand("token_scale", "f32", [capacity]),
            operand("channel_scale", "f32", [n]),
            operand("out", "bf16", [capacity, n], "write"),
            operand("workspace", "f32", [padded, n], "scratch"),
        ]

        def compile_one(unit):
            parameters = dict(
                rows=capacity,
                output=n,
                reduction=k,
                block_k=512,
                tile_m=tile_m,
                tile_n=tile_n,
                unit_scale=unit,
            )
            with LOWER_LOCK:
                function = projection(**dict(parameters, rows=row_count(capacity, True)))
            return compile_kernel(
                function,
                "dense.projection",
                slots,
                parameters,
                args.output / "cache",
                row_capacity=capacity,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            paths = list(pool.map(compile_one, [False, True]))
        print(
            dict(capacity=capacity, output=n, reduction=k, manifests=list(map(str, paths))),
            flush=True,
        )
        if args.compile_only:
            continue
        x = torch.randn(capacity, k, device="npu").bfloat16()
        weight = (torch.randn(n, k, device="npu") * 0.02).bfloat16()
        token_scale = torch.ones(capacity, device="npu")
        channel_scale = torch.ones(n, device="npu")
        outputs = [torch.empty(capacity + 1, n, dtype=torch.bfloat16, device="npu") for _ in paths]
        scratch = [torch.empty(padded + 1, n, device="npu") for _ in paths]
        entries, pointers = [], []
        for i, path in enumerate(paths):
            lib = ctypes.CDLL(str(path.parent / "kernel.so"))
            libraries.append(lib)
            entry = lib.orinfer_ascend_kernel_v2
            entry.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            entry.restype = ctypes.c_int32
            entries.append(entry)
            pointers.append(
                (ctypes.c_void_p * 6)(
                    x.data_ptr(),
                    weight.data_ptr(),
                    token_scale.data_ptr(),
                    channel_scale.data_ptr(),
                    outputs[i].data_ptr(),
                    scratch[i].data_ptr(),
                )
            )
        numerical, timings = [], []
        for rows in sorted({r for r in [1, 2, 17, 31, 32, 129, 800, 2048, 2049] if r <= capacity}):
            for pattern in ["random", "ties", "tiny"]:
                if pattern == "random":
                    x.normal_()
                    weight.normal_(0, 0.02)
                else:
                    x.zero_()
                    weight.zero_()
                    if pattern == "ties":
                        x[:, 0] = 1
                        x[:, 1] = 1 / 256
                        weight[:, 0] = 1
                        weight[1::2, 0] = 1 + 1 / 128
                        weight[:, 1] = 1
                    else:
                        x[:, 0] = 2.0**-65
                        weight[:, 0] = 2.0**-65
                for t in outputs + scratch:
                    t.fill_(17)

                def run(index):
                    status = entries[index](
                        pointers[index], 6, rows, torch.npu.current_stream()._as_parameter_
                    )
                    if status:
                        raise RuntimeError(f"Projection launch failed: {status}")

                def check():
                    torch.npu.synchronize()
                    torch.testing.assert_close(outputs[0][:rows], outputs[1][:rows], atol=0, rtol=0)
                    for t in outputs:
                        assert bool((t[rows:] == 17).all()), "Projection wrote output tail"
                    assert bool((scratch[1] == 17).all()), "Fixpipe touched unused scratch"
                    assert bool((scratch[0][padded:] == 17).all()), "Projection exceeded scratch"

                run(0)
                run(1)
                check()
                numerical.append(dict(rows=rows, pattern=pattern, exact=True))
                if pattern == "random":
                    measured, graphs = [], []
                    for i in [0, 1]:
                        timing, graph = npu_benchmark(lambda: run(i), 2, 3)
                        measured.append(timing["median_ms"])
                        graphs.append(graph)
                    x.neg_()
                    weight.mul_(0.5)
                    for graph in graphs:
                        graph.replay()
                    check()
                    timings.append(
                        dict(
                            rows=rows,
                            baseline_ms=measured[0],
                            direct_ms=measured[1],
                            change_pct=100 * (measured[1] / measured[0] - 1),
                            changed_input_graph=True,
                        )
                    )
        reports.append(
            dict(
                capacity=capacity,
                output=n,
                reduction=k,
                tile_m=tile_m,
                tile_n=tile_n,
                numerical=numerical,
                timings=timings,
            )
        )
        atomic_json(args.output / "validation.json", reports)
        print(reports[-1], flush=True)


if __name__ == "__main__":
    main()
