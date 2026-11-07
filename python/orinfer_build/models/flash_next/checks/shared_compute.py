"""Check shared projection components against an independent integer oracle.

Both instruction schedules, scalar-Q4 and W8 storage, FP16/FP32 epilogues and
rounded gate/up fusion cover dynamic tails and changed-input graph replay.
"""

import argparse
from pathlib import Path

import numpy as np
import tilelang
import torch

from orinfer_build.tuning.projection_tuning import make_prim, profile_stream, TARGET
from kernels.cuda.model.q4_projection import MAGNITUDES
from orinfer_build.compiler.cuda.common import configure, write_json


def expected(kind, a, weights, scales, token_scale, dtype):
    # INT64 on CPU is independent of either GPU dot-product implementation.
    activation = a.cpu().numpy().astype(np.int64)
    decoded = []
    for weight, scale in zip(weights, scales, strict=True):
        w = weight.cpu().numpy()
        if kind == "q4":
            shifts = np.arange(4, dtype=np.uint16) * 4
            indices = ((w[..., None] >> shifts) & 15).reshape(w.shape[0], -1)
            table = np.array((*MAGNITUDES, *(-x for x in MAGNITUDES)), dtype=np.int64)
            w = table[indices]
        accum = torch.from_numpy(activation @ w.astype(np.int64).T).to("cuda", torch.float32)
        # Explicit multiply order and independent FP16 gate/up rounding.
        decoded.append((accum * scale.float()) * token_scale[:, None].float())
    if kind == "swiglu":
        g, u = [v.half().float() for v in decoded]
        return (g / (1 + torch.exp(-g)) * u).to(dtype)
    return decoded[0].to(dtype)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    configure()
    records = []
    for kind in ("w8", "q4", "swiglu"):
        configs = [dict(algorithm="dp4a", block_m=3, block_n=8, block_k=128)]
        configs += [dict(algorithm="tensor_core", block_m=16, block_n=64, block_k=128)]
        if kind == "q4":
            configs += [
                dict(algorithm="tensor_core", block_m=128, block_n=128, block_k=256, stages=1)
            ]
        for config in configs:
            for dtype in ["float16"] if kind == "swiglu" else ["float16", "float32"]:
                capacity = (
                    4096 if kind == "swiglu" and config["algorithm"] == "tensor_core" else 128
                )
                spec = dict(
                    kind=kind, rows=capacity, outputs=130, inputs=256, dtype=dtype, dynamic=True
                )
                kernel = profile_stream(
                    tilelang.compile(
                        make_prim(spec, config),
                        out_idx=[],
                        execution_backend="nvrtc",
                        target=TARGET,
                    )
                )
                weights = [
                    (
                        torch.randint(0, 65536, (130, 64), device="cuda", dtype=torch.int32).to(
                            torch.uint16
                        )
                        if kind == "q4"
                        else torch.randint(-128, 128, (130, 256), device="cuda", dtype=torch.int8)
                    )
                    for _ in range(2 if kind == "swiglu" else 1)
                ]
                scales = [
                    torch.rand(130, device="cuda", dtype=torch.float16) * 0.003 for _ in weights
                ]
                for live in sorted({1, 3, 7, 33, 128, min(512, capacity), capacity}):
                    a = torch.randint(-128, 128, (live, 256), device="cuda", dtype=torch.int8)
                    token = torch.rand(live, device="cuda", dtype=torch.float16) * 0.002
                    storage = torch.full(
                        (live + 2, 130), 91, device="cuda", dtype=getattr(torch, dtype)
                    )
                    out = storage[:live]
                    operands = [a, *weights, *scales, token, out, live]

                    def run():
                        kernel.adapter.func(*operands)

                    run()
                    reference = expected(kind, a, weights, scales, token, out.dtype)
                    torch.cuda.synchronize()
                    if not torch.equal(out, reference):
                        raise AssertionError(
                            (kind, config, dtype, live, float((out - reference).abs().max()))
                        )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    a.bitwise_not_()
                    reference = expected(kind, a, weights, scales, token, out.dtype)
                    storage.fill_(91)
                    graph.replay()
                    torch.cuda.synchronize()
                    if not torch.equal(out, reference) or not bool((storage[live:] == 91).all()):
                        raise AssertionError(
                            (kind, config, dtype, live, "changed-input replay/tail")
                        )
                    records.append(
                        dict(kind=kind, config=config, dtype=dtype, rows=live, exact=True)
                    )
                    print("PASS", kind, config["algorithm"], dtype, live, flush=True)
    write_json(args.output / "results.json", dict(complete=True, cases=records))


if __name__ == "__main__":
    main()
