"""Check fused W8 SwiGLU against integer projections and FP16 rounding.

Includes signed INT8 extrema, zero scales, non-aligned output widths, guards,
every MTP refresh width and graph replay after an activation change.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from kernels.model.flash_next import swiglu
from kernels.model.int8_projection import int8_projection
from kernels.model.int8_swiglu import int8_swiglu
from tools.operators.common import configure, environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    report = dict(complete=False, environment=environment(), cases=[])
    rng = np.random.default_rng(20261002)
    for n, k in ((13, 256), (67, 640), (640, 2560)):
        for m in range(1, 9):
            a = rng.integers(-128, 128, (m, k), dtype=np.int8)
            gate = rng.integers(-128, 128, (n, k), dtype=np.int8)
            up = rng.integers(-128, 128, (n, k), dtype=np.int8)
            gs = rng.uniform(0.0001, 0.001, n).astype(np.float16)
            us = rng.uniform(0.0001, 0.001, n).astype(np.float16)
            ts = rng.uniform(0.0001, 0.001, m).astype(np.float16)
            gs[0] = 0
            us[-1] = 0
            tensors = [torch.from_numpy(v).cuda() for v in (a, gate, up, gs, us, ts)]
            x, wg, wu, gscale, uscale, scale = tensors
            g = torch.empty((m, n), device="cuda", dtype=torch.float16)
            u = torch.empty_like(g)
            expected = torch.empty_like(g)
            guarded = torch.full((m * n + 17,), 42, device="cuda", dtype=torch.float16)
            out = guarded[: m * n].view(m, n)
            project = int8_projection(m, n, k)
            activation = swiglu(m, n)
            fused = int8_swiglu(m, n, k, 4 if m == 1 else 64)
            operands = (
                (
                    x.view(torch.int32),
                    wg.view(torch.int32),
                    wu.view(torch.int32),
                    gscale,
                    uscale,
                    scale,
                    out,
                )
                if m == 1
                else (*tensors, out)
            )

            def reference():
                project(x, wg, gscale, scale, g)
                project(x, wu, uscale, scale, u)
                activation(g, u, expected)

            reference()
            for weight, ws, result in ((gate, gs, g), (up, us, u)):
                integers = a.astype(np.int32) @ weight.astype(np.int32).T
                rounded = (
                    (integers.astype(np.float32) * ws.astype(np.float32)[None, :])
                    * ts.astype(np.float32)[:, None]
                ).astype(np.float16)
                np.testing.assert_array_equal(result.cpu().numpy(), rounded)
            fused(*operands)
            torch.cuda.synchronize()
            assert torch.equal(out, expected), (m, n, "direct")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused(*operands)
            x.bitwise_not_()
            reference()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, expected), (m, n, "changed-input")
            assert bool((guarded[m * n :] == 42).all()), (m, n, "guard")
            report["cases"].append(
                dict(
                    M=m,
                    N=n,
                    K=k,
                    integer_reference_exact=True,
                    output_exact=True,
                    changed_input_graph_exact=True,
                    guard_intact=True,
                )
            )
            write_json(args.output / "results.json", report)
            print("PASS", m, n, k, flush=True)
    report["complete"] = True
    write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
