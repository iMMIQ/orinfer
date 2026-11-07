"""Check three-part BF16 projections, FP32 boundaries and dynamic row tails."""

import argparse
import ctypes
from pathlib import Path
from orinfer_build.common.files import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=7)
    p.add_argument("--two-part-head-only", action="store_true")
    args = p.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from kernels.shared.rows import row_count
    from orinfer_build.models.deepseek_v41.floating import bf16_parts
    from orinfer_build.tuning.ascend import npu_benchmark
    from kernels.ascend import floating
    from orinfer_build.compiler.ascend.build import compile_kernel, operand

    torch.manual_seed(20261002)
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    reports = []
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

    cases = [
        (1, 2, False, 129, 128, 5120, 128),
        (1, 3, False, 129, 64, 1024, 128),
        (3, 1, False, 129, 64, 1024, 128),
        (3, 1, False, 129, 384, 5120, 128),
        (1, 3, False, 129, 128, 5120, 128),
        (1, 3, True, 17, 32, 20480, 256),
    ]
    if args.two_part_head_only:
        cases = cases[:1]
    for ap, wp, variance, capacity, n, k, bk in cases:
        original_a = torch.randn(capacity, k) * 0.2
        original_w = torch.randn(n, k) * 0.02
        if ap == 1:
            original_a = original_a.bfloat16().float()
        if wp == 1:
            original_w = original_w.bfloat16().float()
        if variance:
            original_w[24:].zero_()
        cpu_w = (
            torch.from_numpy(bf16_parts(original_w.numpy())[:wp]).view(torch.bfloat16)
            if wp > 1
            else original_w.bfloat16()[None]
        )
        device_a = torch.empty((ap * capacity * k,), dtype=torch.bfloat16, device="npu")
        device_w = cpu_w.npu()
        out = torch.empty((capacity + 1, n * 8), device="npu")
        source = original_a.npu()
        params = dict(
            rows=capacity,
            output=n,
            reduction=k,
            input_parts=ap,
            weight_parts=wp,
            variance=variance,
            block_k=bk,
        )
        if wp == 2:
            params.update(reuse_operand=True, tile_m=64, tile_n=128)
        specs = [
            operand("activation", "bf16", [ap, capacity, k]),
            operand("weight", "bf16", [wp, n, k]),
            operand("out", "f32", [capacity, n * 8], "write"),
        ]
        path = compile_kernel(
            floating.partials(
                row_count(capacity, True), **{a: v for a, v in params.items() if a != "rows"}
            ),
            "floating.partials",
            specs,
            params,
            args.output / "cache",
            row_capacity=capacity,
        )
        call = entry(path)
        pointers = (ctypes.c_void_p * 3)(device_a.data_ptr(), device_w.data_ptr(), out.data_ptr())
        if ap == 3:
            split_specs = [
                operand("source", "f32", [capacity, k]),
                operand("out", "bf16", [3, capacity, k], "write"),
            ]
            split_path = compile_kernel(
                floating.split(row_count(capacity, True), k),
                "floating.split",
                split_specs,
                dict(rows=capacity, width=k),
                args.output / "cache",
                row_capacity=capacity,
            )
            split_call = entry(split_path)
            split_pointers = (ctypes.c_void_p * 2)(source.data_ptr(), device_a.data_ptr())
        for rows in sorted({r for r in (1, 2, 16, 17, 31, 32, 128, capacity) if r <= capacity}):

            def run():
                stream = torch.npu.current_stream()._as_parameter_
                if ap == 3:
                    status = split_call(split_pointers, 2, rows, stream)
                    if status:
                        raise RuntimeError(f"FP32 decomposition launch failed: {status}")
                status = call(pointers, 3, rows, stream)
                if status:
                    raise RuntimeError(f"FP32 projection launch failed: {status}")

            def expected():
                tile_n = params.get("tile_n", 32)
                result = torch.empty(rows, n // tile_n, 8, tile_n)
                reference_w = cpu_w.float().sum(0) if wp == 2 else original_w
                for split in range(8):
                    begin, end = split * (k // 8), (split + 1) * (k // 8)
                    value = (
                        original_a[:rows, begin:end].double() @ reference_w[:, begin:end].double().T
                    ).float()
                    result[:, :, split] = value.reshape(rows, n // tile_n, tile_n)
                    if variance:
                        square = (
                            original_a[:rows, begin:end].square().reshape(rows, 10, 256).sum(-1)
                        )
                        total = torch.zeros(rows)
                        for i in range(10):
                            total += square[:, i]
                        result[:, 0, split, 24] = total
                return result.reshape(rows, n * 8)

            def check():
                torch.npu.synchronize()
                if ap == 3:
                    parts = torch.from_numpy(bf16_parts(original_a[:rows].numpy())).view(
                        torch.bfloat16
                    )
                    torch.testing.assert_close(
                        device_a[: 3 * rows * k].cpu().reshape(3, rows, k), parts, atol=0, rtol=0
                    )
                actual = out.cpu()
                torch.testing.assert_close(actual[:rows], expected(), atol=2e-5, rtol=2e-5)
                assert bool((actual[rows:] == 17).all()), "Floating Cube wrote the tail"

            out.fill_(17)
            if ap == 1:
                device_a[: rows * k].copy_(original_a[:rows].bfloat16().reshape(-1))
            run()
            check()
            _, graph = npu_benchmark(run, 2, 1)
            original_a.neg_()
            source.copy_(original_a)
            if ap == 1:
                device_a[: rows * k].copy_(original_a[:rows].bfloat16().reshape(-1))
            graph.replay()
            check()
            reports.append(dict(input_parts=ap, weight_parts=wp, variance=variance, rows=rows))
            print(reports[-1], flush=True)
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
