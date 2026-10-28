"""Check vector mHC row broadcasts against FP64, including repeated Graphs."""

import argparse
import ctypes
from pathlib import Path

from tools.build.files import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=7)
    p.add_argument("--tile-rows", type=int, default=1)
    p.add_argument("--compile-only", action="store_true")
    args = p.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.tune import npu_benchmark
    from tools.operators.ascend.build import compile_kernel, operand
    from tools.operators.ascend.vector import mhc_split_sinkhorn

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    if not args.compile_only:
        torch.npu.set_device(args.device)
    capacity = 2049
    inputs = [torch.randn(capacity, 32) * 4, torch.randn(8), torch.randn(32)]
    path = compile_kernel(
        mhc_split_sinkhorn(row_count(capacity, True), tile_rows=args.tile_rows),
        "vector.mhc_split_sinkhorn",
        [
            operand("mixes", "f32", [capacity, 32]),
            operand("scale", "f32", [8]),
            operand("base", "f32", [32]),
            operand("pre", "f32", [capacity, 8], "write"),
            operand("post", "f32", [capacity, 8], "write"),
            operand("combination", "f32", [capacity, 16], "write"),
        ],
        dict(rows=capacity, tile_rows=args.tile_rows),
        args.output / "cache",
        row_capacity=capacity,
    )
    if args.compile_only:
        return
    tensors = [v.npu() for v in inputs]
    outputs = [torch.empty(capacity + 1, w, device="npu") for w in (8, 8, 16)]
    tensors.extend(outputs)
    lib = ctypes.CDLL(str(path.parent / "kernel.so"))
    call = lib.orinfer_ascend_kernel_v2
    call.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_void_p,
    ]
    call.restype = ctypes.c_int32
    ptrs = (ctypes.c_void_p * 6)(*[t.data_ptr() for t in tensors])
    reports = []
    for rows in (1, 2, 17, 31, 32, 129, 2048, 2049):
        for out in outputs:
            out.fill_(17)

        def run():
            status = call(ptrs, 6, rows, torch.npu.current_stream()._as_parameter_)
            if status:
                raise RuntimeError(f"mHC Sinkhorn launch failed: {status}")

        def check():
            torch.npu.synchronize()
            mix, scale, base = [v.double() for v in inputs]
            z = mix[:rows, :24] * torch.repeat_interleave(scale[:3], torch.tensor([4, 4, 16]))
            z += base[:24]
            pre = torch.zeros(rows, 8, dtype=torch.float64)
            post = torch.zeros_like(pre)
            pre[:, :4] = z[:, :4].sigmoid() + 1e-6
            post[:, :4] = z[:, 4:8].sigmoid() * 2
            a = z[:, 8:24].reshape(rows, 4, 4).softmax(-1) + 1e-6
            a /= a.sum(-2, keepdim=True) + 1e-6
            for _ in range(19):
                a /= a.sum(-1, keepdim=True) + 1e-6
                a /= a.sum(-2, keepdim=True) + 1e-6
            errors = []
            for actual, wanted in zip(outputs, (pre, post, a.flatten(1)), strict=True):
                value = actual[:rows].cpu().double()
                torch.testing.assert_close(value, wanted, atol=3e-6, rtol=3e-5)
                assert bool((actual[rows:] == 17).all()), "mHC Sinkhorn wrote tail"
                errors.append(float((value - wanted).abs().max()))
            return errors

        run()
        errors = check()
        timing, graph = npu_benchmark(run, 3, 1)
        snapshot = [v[:rows].clone() for v in outputs]
        for _ in range(10):
            graph.replay()
            torch.npu.synchronize()
            for v, previous in zip(outputs, snapshot, strict=True):
                torch.testing.assert_close(v[:rows], previous, atol=0, rtol=0)
        inputs[0].neg_()
        tensors[0].copy_(inputs[0])
        graph.replay()
        check()
        reports.append(dict(rows=rows, milliseconds=timing["median_ms"], errors=errors))
        print(reports[-1], flush=True)
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
