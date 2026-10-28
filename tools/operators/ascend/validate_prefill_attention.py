"""Check packed prefill attention against an independent BF16-probability reference."""

import argparse
import ctypes
from pathlib import Path

from tools.build.files import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=7)
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 2, 17, 31, 32, 128, 129])
    p.add_argument("--bases", type=int, nargs="+", default=[0, 64])
    args = p.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.tune import npu_benchmark
    from tools.operators.ascend import attention_prefill as ops
    from tools.operators.ascend.build import compile_kernel, operand

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    if not args.compile_only:
        torch.npu.set_device(args.device)
    capacity, context = max(2, max(args.rows)), args.context
    if context % 256 or not 256 <= context <= 8448 or not 2 <= capacity <= 4096:
        p.error("Require aligned context 256..8448 and row capacity 2..4096")
    if min(args.rows) < 1 or min(args.bases) < 0 or max(args.bases) + capacity > context:
        p.error("Requested positions exceed the context")
    m = row_count(capacity, True)
    reports, libraries = [], []
    for compressed, ratio in ((False, 2), (True, 2), (True, 1)):
        long_length = context // ratio
        length = context + (long_length if compressed else 0)
        query = torch.randn(capacity, 8, 512).bfloat16()
        swa = (torch.randn(context, 512) * 0.2).bfloat16()
        long = (torch.randn(long_length, 512) * 0.2).bfloat16()
        sink = torch.randn(8)
        tensors = None
        if not args.compile_only:
            tensors = dict(
                query=query.npu(),
                swa=swa.npu(),
                long_kv=long.npu(),
                sink=sink.npu(),
                positions=torch.empty(capacity, dtype=torch.int32, device="npu"),
                indices=torch.empty(capacity, 512, dtype=torch.int32, device="npu"),
                mask=torch.empty(capacity, long_length, dtype=torch.int32, device="npu"),
                scores=torch.empty((capacity * 8, length), device="npu"),
                probabilities=torch.empty(
                    ((capacity + 1) * 8, length), dtype=torch.bfloat16, device="npu"
                ),
                output=torch.empty(((capacity + 1) * 8, 512), dtype=torch.bfloat16, device="npu"),
                workspace=torch.empty(((capacity * 8 + 127) // 128 * 128, 512), device="npu"),
            )
        layouts = dict(
            query=("bf16", [capacity * 8, 512]),
            swa=("bf16", [context, 512]),
            long_kv=("bf16", [long_length, 512]),
            sink=("f32", [8]),
            positions=("i32", [capacity]),
            indices=("i32", [capacity, 512]),
            mask=("i32", [capacity, long_length]),
            scores=("f32", [capacity * 8, length]),
            probabilities=("bf16", [capacity * 8, length]),
            output=("bf16", [capacity * 8, 512]),
            workspace=("f32", [(capacity * 8 + 127) // 128 * 128, 512]),
        )
        calls = []
        for name, parameters, slots in [
            (
                "selected_mask",
                dict(length=long_length),
                [("indices", "indices", "read"), ("mask", "mask", "write")],
            ),
            (
                "qk",
                dict(capacity=context, ratio=ratio, compressed=compressed),
                [
                    (n, n, "write" if n == "scores" else "read")
                    for n in ("query", "swa", "long_kv", "positions", "scores")
                ],
            ),
            (
                "softmax",
                dict(capacity=context, ratio=ratio, compressed=compressed),
                [
                    ("scores", "scores", "read"),
                    ("positions", "positions", "read"),
                    ("selected", "mask", "read"),
                    ("sink", "sink", "read"),
                    ("probabilities", "probabilities", "write"),
                ],
            ),
            (
                "pv",
                dict(capacity=context, ratio=ratio, compressed=compressed),
                [
                    (n, n, "write" if n == "output" else "scratch" if n == "workspace" else "read")
                    for n in ("probabilities", "swa", "long_kv", "positions", "output", "workspace")
                ],
            ),
        ]:
            specs = [operand(role, *layouts[key], access) for role, key, access in slots]
            path = compile_kernel(
                getattr(ops, name)(m, **parameters),
                "attention_prefill." + name,
                specs,
                dict(rows=capacity, **parameters),
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
            pointers = (ctypes.c_void_p * len(slots))(
                *[tensors[key].data_ptr() for _, key, _ in slots]
            )
            calls.append((call, pointers, len(slots)))
        if args.compile_only:
            continue
        for rows in args.rows:
            for base in args.bases:
                positions = torch.arange(rows, dtype=torch.int32) + base
                ids = torch.full((rows, 512), -1, dtype=torch.int32)
                selected = torch.zeros(rows, long_length, dtype=torch.bool)
                for row, position in enumerate(positions):
                    count = int((position + 1) // ratio)
                    chosen = torch.randperm(count)[:512]
                    ids[row, : len(chosen)] = chosen.int()
                    selected[row, chosen] = True
                tensors["positions"][:rows].copy_(positions)
                tensors["indices"][:rows].copy_(ids)
                tensors["probabilities"].fill_(17)
                tensors["output"].fill_(17)

                def run():
                    for call, pointers, count in calls:
                        status = call(
                            pointers, count, rows, torch.npu.current_stream()._as_parameter_
                        )
                        if status:
                            raise RuntimeError(f"Packed attention launch failed: {status}")

                def check():
                    torch.npu.synchronize()
                    torch.testing.assert_close(
                        tensors["mask"][:rows].cpu().bool(), selected, atol=0, rtol=0
                    )
                    scores = query[:rows].float().reshape(rows * 8, 512) @ swa.float().T
                    valid = (torch.arange(context)[None] <= positions[:, None]) & (
                        torch.arange(context)[None] > positions[:, None] - 128
                    )
                    scores = scores.reshape(rows, 8, context).masked_fill(
                        ~valid[:, None], -torch.inf
                    )
                    bank = swa.float()
                    if compressed:
                        extra = (
                            query[:rows].float().reshape(rows * 8, 512) @ long.float().T
                        ).reshape(rows, 8, long_length)
                        extra = extra.masked_fill(~selected[:, None], -torch.inf)
                        scores = torch.cat((scores, extra), dim=-1)
                        bank = torch.cat((bank, long.float()))
                    scores *= 512**-0.5
                    scores = torch.cat((scores, sink[None, :, None].expand(rows, -1, 1)), dim=-1)
                    probs = scores.softmax(-1)[..., :-1].bfloat16()
                    expected = (probs.float().reshape(rows * 8, length) @ bank).bfloat16()
                    actual = tensors["output"].cpu()
                    torch.testing.assert_close(actual[: rows * 8], expected, atol=0.003, rtol=0.02)
                    torch.testing.assert_close(
                        tensors["probabilities"][: rows * 8].cpu().reshape(rows, 8, length),
                        probs,
                        atol=0.00025,
                        rtol=0.01,
                    )
                    assert bool((actual[rows * 8 :] == 17).all()), "Packed PV wrote tail"
                    assert bool((tensors["probabilities"][rows * 8 :] == 17).all()), (
                        "Packed softmax wrote tail"
                    )

                run()
                check()
                timing, graph = npu_benchmark(run, 2, 1)
                query.neg_()
                long.neg_()
                tensors["query"].copy_(query)
                tensors["long_kv"].copy_(long)
                graph.replay()
                check()
                reports.append(
                    dict(
                        compressed=compressed,
                        ratio=ratio,
                        rows=rows,
                        base=base,
                        latency_ms=timing["median_ms"],
                    )
                )
                print(reports[-1], flush=True)
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
