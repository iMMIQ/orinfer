"""Verify fused routing/planes against dispatch followed by exact A8 packing."""

import argparse
from pathlib import Path

import tilelang
import torch

from kernels.cuda.model.e8p_prefill import dispatch_prefill_a8, split_prefill_a8
from kernels.cuda.model.moe import expert_dispatch
from kernels.shared.rows import explicit_rows
from orinfer_build.compiler.cuda.common import configure, export_kernel, write_json


def launch(kernel, *operands):
    kernel.adapter.func(*operands, stream=torch.cuda.current_stream().cuda_stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    torch.manual_seed(20261002)
    candidate = tilelang.compile(
        explicit_rows(dispatch_prefill_a8.__wrapped__(4096, dynamic_rows=True)),
        out_idx=[],
        execution_backend="nvrtc",
        target={"kind": "cuda", "arch": "sm_87"},
    )
    export_kernel(candidate, args.output / "aot")
    report = dict(complete=False, cases=[])
    for rows in [2, 17, 129, 512, 1025]:
        slots = rows * 10
        ids = torch.rand((rows, 512), device="cuda").topk(10, dim=1).indices.to(torch.int32)
        flat = ids.flatten().long()
        counts = torch.bincount(flat, minlength=512).to(torch.int32)
        offsets = (counts.cumsum(0) - counts).to(torch.int32)
        one_hot = torch.nn.functional.one_hot(flat, 512).to(torch.int32)
        relative = one_hot.cumsum(0, dtype=torch.int32).gather(1, flat[:, None]).flatten() - 1
        relative = relative.reshape(rows, 10).to(torch.int32)
        activation = torch.randint(-128, 128, (rows, 2560), device="cuda", dtype=torch.int8)
        scale = torch.full((rows, 1), 0.01, device="cuda", dtype=torch.float16)
        routed = torch.empty((slots, 2560), device="cuda", dtype=torch.int8)
        expected = [
            torch.empty((40, slots, 8), device="cuda", dtype=torch.uint32),
            torch.empty((40, slots, 8), device="cuda", dtype=torch.uint32),
            torch.empty(slots, device="cuda", dtype=torch.int32),
            torch.empty((slots, 1), device="cuda", dtype=torch.float16),
            torch.empty_like(ids),
        ]
        actual = [torch.empty_like(t) for t in expected]
        dispatch = expert_dispatch(rows, 2560, scale_group=2560)
        pack = split_prefill_a8(slots, 2560)

        def reference():
            launch(
                dispatch,
                activation,
                scale,
                ids,
                relative,
                offsets,
                routed,
                expected[3],
                expected[4],
            )
            launch(pack, routed, *expected[:3])

        def fused():
            launch(candidate, activation, scale, ids, relative, offsets, *actual, rows)

        reference()
        fused()
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)), rows
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused()
        activation.bitwise_not_()
        reference()
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)), (
            rows,
            "graph",
        )
        report["cases"].append(dict(rows=rows, exact=True, graph_replay=True))
        write_json(args.output / "results.json", report)
    report["complete"] = True
    write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
