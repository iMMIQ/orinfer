"""Validate bounded dynamic A2 rows, exact tails and changed-input Graph replay."""

import argparse
import ctypes
import json
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
    from tools.operators.ascend.build import compile_kernel, operand
    from tools.operators.ascend import dense, memory, model_ops
    from tools.model.deepseek_v41.tune import npu_benchmark

    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    capacity = 129
    m = row_count(capacity, True)
    reports = []
    for kind in ("last_row", "causal1", "causal2", "integer_projection", "expert_projection"):
        if kind == "last_row":
            function = memory.last_row(m, 512)
            parameters = dict(rows=capacity, width=512)
            specs = [
                operand("source", "bf16", [capacity, 512]),
                operand("out", "bf16", [1, 512], "write"),
            ]
            tensors = [
                torch.randn(capacity, 512).bfloat16().npu(),
                torch.empty(2, 512, dtype=torch.bfloat16, device="npu"),
            ]
        elif kind.startswith("causal"):
            ratio = int(kind[-1])
            function = model_ops.causal_indices(m, 320, ratio)
            parameters = dict(rows=capacity, capacity=320, ratio=ratio)
            specs = [
                operand("positions", "i32", [capacity]),
                operand("indices", "i32", [capacity, 320 // ratio], "write"),
                operand("lengths", "i32", [capacity], "write"),
            ]
            tensors = [
                torch.arange(capacity, dtype=torch.int32).npu(),
                torch.empty(capacity + 1, 320 // ratio, dtype=torch.int32, device="npu"),
                torch.empty(capacity + 1, dtype=torch.int32, device="npu"),
            ]
        else:
            multiplier = 6 if kind == "expert_projection" else 1
            physical = capacity * multiplier
            function = dense.projection(m * multiplier, 512, 128, True)
            parameters = dict(rows=physical, output=512, reduction=128, integer=True)
            specs = [
                operand("activation", "i8", [physical, 128]),
                operand("weight", "i8", [512, 128]),
                operand("token_scale", "f32", [physical]),
                operand("channel_scale", "f32", [512]),
                operand("out", "bf16", [physical, 512], "write"),
                operand("workspace", "i32", [(physical + 15) // 16 * 16, 512], "scratch"),
            ]
            tensors = [
                torch.randint(-127, 128, (physical, 128), dtype=torch.int8).npu(),
                torch.randint(-127, 128, (512, 128), dtype=torch.int8).npu(),
                torch.full((physical,), 0.001, device="npu"),
                torch.full((512,), 0.002, device="npu"),
                torch.empty(physical + 1, 512, dtype=torch.bfloat16, device="npu"),
                torch.empty((physical + 15) // 16 * 16, 512, dtype=torch.int32, device="npu"),
            ]
        path = compile_kernel(
            function, kind, specs, parameters, args.output / "cache", row_capacity=capacity
        )
        library = ctypes.CDLL(str(path.parent / "kernel.so"))
        entry = library.orinfer_ascend_kernel_v2
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        entry.restype = ctypes.c_int32
        pointers = (ctypes.c_void_p * len(tensors))(*[t.data_ptr() for t in tensors])
        outputs = [i for i, spec in enumerate(specs) if spec["access"] == "write"]
        for rows in (1, 2, 17, 127, 128, 129):

            def expected():
                if kind == "last_row":
                    return [tensors[0][rows - 1 : rows].cpu()]
                if kind.startswith("causal"):
                    lengths = (tensors[0][:rows].cpu() + 1) // ratio
                    ids = torch.arange(320 // ratio).expand(rows, -1)
                    return [torch.where(ids < lengths[:, None], ids, -1).int(), lengths]
                a, w, sa, sw = [t.cpu() for t in tensors[:4]]
                actual_rows = rows * multiplier
                dot = a[:actual_rows].int() @ w.int().T
                return [(dot.float() * sa[:actual_rows, None] * sw[None, :]).bfloat16()]

            def run():
                status = entry(
                    pointers, len(tensors), rows, torch.npu.current_stream()._as_parameter_
                )
                if status:
                    raise RuntimeError(f"{kind} native error {status}")

            def check():
                torch.npu.synchronize()
                for slot, wanted in zip(outputs, expected(), strict=True):
                    actual = tensors[slot].cpu()
                    count = wanted.shape[0]
                    torch.testing.assert_close(actual[:count], wanted, atol=0, rtol=0)
                    assert bool((actual[count:] == 17).all()), "Dynamic A2 tail was written"

            for slot in outputs:
                tensors[slot].fill_(17)
            run()
            check()
            _, graph = npu_benchmark(run, 2, 1)
            if kind.startswith("causal"):
                tensors[0].add_(3)
            else:
                tensors[0].neg_()
            graph.replay()
            check()
            reports.append(dict(op=kind, rows=rows, changed_input_graph=True))
            print(json.dumps(reports[-1]), flush=True)
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
