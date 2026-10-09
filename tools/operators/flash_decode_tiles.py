"""Exact integer/FP32 oracles, tails and changed-input graphs for decode tiles."""

import argparse
from pathlib import Path

import numpy as np
import torch

from kernels.model.flash_next import dense_projection
from kernels.model.integer_vq import integer_vq
from tools.model.flash_next.kernel_policy import expert_tile_config
from tools.operators.common import configure, write_json, error
from tools.operators.quantized_reference import upload, reference
from tools.quantization.vq import Weights, e8p_sign_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    rng = np.random.default_rng(20261002)
    report = dict(complete=False, experts=[], routers=[])
    table = (rng.integers(-63, 64, (256, 8)) * 2).astype(np.int8)
    e, n, k = 2, 65, 256
    bank = [
        Weights(
            "e8p",
            rng.integers(0, 65536, (n, k // 8), dtype=np.uint16),
            table,
            np.full(n, 0.015625, np.float16),
            np.empty(0, np.int8),
        )
        for _ in range(e)
    ]
    packed, _, ws = upload(bank)
    book = torch.from_numpy(e8p_sign_table(table)[None]).cuda()
    patch = torch.zeros(1, device="cuda", dtype=torch.uint16)
    integer = torch.from_numpy(np.stack([w.integer_weights() for w in bank])).cuda()
    for m in range(2, 9):
        x = torch.randint(-128, 128, (e, m, k), device="cuda", dtype=torch.int8)
        scales = torch.full((e, m), 0.0078125, device="cuda", dtype=torch.float16)
        guarded = torch.full((e * m + 1, n), 91.0, device="cuda", dtype=torch.float16)
        out = guarded[: e * m]
        kernel = integer_vq(e, m, n, k, kind="e8p", shared_table=True, **expert_tile_config(m))

        def run():
            kernel(x.view(e * m, k), packed, book, patch, ws, scales.view(-1), out)

        run()
        expected = reference(x, integer, ws, scales).view(e * m, n)
        assert torch.equal(out, expected), error(out, expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        x.bitwise_not_()
        graph.replay()
        torch.cuda.synchronize()
        expected = reference(x, integer, ws, scales).view(e * m, n)
        assert torch.equal(out, expected), error(out, expected)
        assert bool((guarded[-1] == 91).all())
        report["experts"].append(
            dict(rows=m, integer_oracle_exact=True, changed_graph=True, tail_guard=True)
        )
        write_json(args.output / "results.json", report)
        del graph
    for m in range(1, 9):
        for outputs in (65, 512):
            w = (torch.randn((outputs, 2560), device="cuda") * 0.01).bfloat16()
            x = (torch.randn((m, 2560), device="cuda") * 0.3).half()
            guarded = torch.full((m * outputs + 17,), 91.0, device="cuda")
            out = guarded[: m * outputs].reshape(m, outputs)
            expected = torch.empty_like(out)
            old = dense_projection(m, outputs, 2560, "bfloat16", "float32")
            kernel = dense_projection(m, outputs, 2560, "bfloat16", "float32", block_n=32)
            old(x, w, expected)
            kernel(x, w, out)
            assert torch.equal(out, expected), error(out, expected)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                kernel(x, w, out)
            x.neg_()
            old(x, w, expected)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, expected), error(out, expected)
            assert bool((guarded[m * outputs :] == 91).all())
            report["routers"].append(
                dict(
                    rows=m,
                    outputs=outputs,
                    baseline_exact=True,
                    changed_graph=True,
                    tail_guard=True,
                )
            )
            write_json(args.output / "results.json", report)
            del graph
    report["complete"] = True
    write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
