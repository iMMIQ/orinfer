"""Check batched mHC against the original FP32 kernel, including Graph and tails."""

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
    from tools.operators.ascend import hc_prefill, vector
    from tools.operators.ascend.build import compile_kernel, operand
    from tools.model.deepseek_v41.tune import npu_benchmark

    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    cap = 2049
    m = row_count(cap, True)
    source = torch.randn(cap, 8, 32)
    source[:, :, 24] = torch.rand(cap, 8) * 10240 + 1
    source[:, :, 25:] = 0
    seed_source = source.clone()
    data = source.npu()
    outputs = [torch.empty(cap + 1, 32, device="npu") for _ in range(2)]
    entries = []
    libraries = []
    for i, (name, factory) in enumerate(
        [
            ("vector.mhc_prenorm_reduce", vector.mhc_prenorm_reduce),
            ("hc_prefill.reduce", hc_prefill.reduce),
        ]
    ):
        path = compile_kernel(
            factory(m, 5120),
            name,
            [operand("partials", "f32", [cap, 8, 32]), operand("out", "f32", [cap, 32], "write")],
            dict(rows=cap, width=5120),
            args.output / "cache",
            row_capacity=cap,
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
        pointers = (ctypes.c_void_p * 2)(data.data_ptr(), outputs[i].data_ptr())
        entries.append((call, pointers))
    reports = []
    for rows in (1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 127, 128, 129, 2048, 2049):
        for kind in ("random", "cancellation"):
            source = seed_source.clone()
            if kind == "cancellation":
                source[:rows, :4, :24] *= 2**12
                source[:rows, 4:, :24] = -source[:rows, :4, :24]
            data.copy_(source)
            for out in outputs:
                out.fill_(17)

            def run(which):
                call, pointers = entries[which]
                status = call(pointers, 2, rows, torch.npu.current_stream()._as_parameter_)
                if status:
                    raise RuntimeError(f"Batched mHC launch {status}")

            def check():
                torch.npu.synchronize()
                a, b = [out.cpu() for out in outputs]
                torch.testing.assert_close(b[:rows], a[:rows], atol=0, rtol=0)
                for out in (a, b):
                    assert bool((out[rows:] == 17).all()), "mHC wrote tail"
                wanted = source[:rows].double().sum(1)
                scale = (wanted[:, 24:25] / 20480 + 1e-20).rsqrt()
                bound = source[:rows].double().abs().sum(1) * scale * (8 * 2**-23)
                wanted = wanted * scale
                wanted[:, 24:] = 0
                assert bool(((a[:rows].double() - wanted).abs() <= bound + 1e-5).all()), (
                    "mHC exceeds the FP32 split-sum error bound"
                )

            run(0)
            run(1)
            check()
            times = []
            graphs = []
            for i in (0, 1):
                timing, graph = npu_benchmark(lambda: run(i), 4, 1)
                times.append(timing["median_ms"])
                graphs.append(graph)
            source[:rows, :, :24].neg_()
            data.copy_(source)
            for graph in graphs:
                graph.replay()
            check()
            reports.append(dict(rows=rows, kind=kind, original_ms=times[0], batched_ms=times[1]))
            print(reports[-1], flush=True)
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
