"""Check signed Walsh rotation against the original bank and FP64 reference."""

import argparse
import ctypes
from pathlib import Path

from tools.build.files import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rotation", type=Path, required=True)
    p.add_argument("--device", type=int, default=7)
    args = p.parse_args()
    import numpy as np
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.tune import npu_benchmark
    from tools.operators.ascend import engram
    from tools.operators.ascend.build import compile_kernel, operand

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    rotation = torch.from_numpy(np.fromfile(args.rotation, dtype="<f4").reshape(32, 32))
    h = torch.tensor([[(-1) ** ((i & j).bit_count()) for j in range(32)] for i in range(32)])
    magnitude = rotation[0, 0]
    assert magnitude > 0 and torch.equal(rotation, rotation[:, :1] * h), "Not a signed Walsh bank"
    assert bool((rotation.abs() == magnitude).all()), "Nonuniform Walsh magnitude"
    capacity, width = 2049, 5120
    source = torch.randn(capacity, 4, width).bfloat16()
    device_x, device_r = source.npu(), rotation.npu()
    outputs = [torch.empty(capacity + 1, 4, width, device="npu") for _ in range(4)]
    libraries, entries = [], []
    for i in range(4):
        params = dict(rows=capacity, width=width)
        if i == 0:
            params["block"] = 32
            factory, name = engram.rotate_blocks, "engram.rotate_blocks"
        else:
            params["tile_groups"] = (16, 32, 64)[i - 1]
            factory, name = engram.hadamard32, "engram.hadamard32"
        path = compile_kernel(
            factory(**dict(params, rows=row_count(capacity, True))),
            name,
            [
                operand("hidden", "bf16", [capacity, 4, width]),
                operand("rotation", "f32", [32, 32]),
                operand("out", "f32", [capacity, 4, width], "write"),
            ],
            params,
            args.output / "cache",
            row_capacity=capacity,
        )
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
        ptrs = (ctypes.c_void_p * 3)(
            device_x.data_ptr(), device_r.data_ptr(), outputs[i].data_ptr()
        )
        entries.append((entry, ptrs))
    reports = []
    for rows in (1, 2, 17, 31, 129, 2048, 2049):
        for out in outputs:
            out.fill_(17)

        def run(which):
            call, ptrs = entries[which]
            status = call(ptrs, 3, rows, torch.npu.current_stream()._as_parameter_)
            if status:
                raise RuntimeError(f"Walsh rotation launch failed: {status}")

        def check():
            torch.npu.synchronize()
            values = [out[:rows].cpu().float().reshape(-1, 32) for out in outputs]
            reference = source[:rows].float().reshape(-1, 32).double() @ rotation.double().T
            bound = source[:rows].float().reshape(-1, 32).double().abs().sum(1, keepdim=True)
            bound = bound * magnitude * (32 * 2**-23)
            for i, value in enumerate(values):
                assert bool(((value.double() - reference).abs() <= bound + 1e-7).all()), (
                    f"Rotation {i} exceeded the FP32 reduction error bound"
                )
                assert bool((outputs[i][rows:] == 17).all()), "Walsh rotation wrote tail"
            return [
                dict(
                    max_abs=float((v - values[0]).abs().max()),
                    different=int((v != values[0]).sum()),
                )
                for v in values[1:]
            ]

        for i in range(4):
            run(i)
        error = check()
        times, graphs = [], []
        for i in range(4):
            timing, graph = npu_benchmark(lambda: run(i), 3, 1)
            times.append(timing["median_ms"])
            graphs.append(graph)
        source.neg_()
        device_x.copy_(source)
        for graph in graphs:
            graph.replay()
        check()
        reports.append(dict(rows=rows, milliseconds=times, errors=error))
        print(reports[-1], flush=True)
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
