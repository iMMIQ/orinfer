"""Exhaust the HC stream sigmoid domain and verify changed-input graph replay."""

import argparse
from pathlib import Path

import torch
import tilelang.language as T

from kernels.cuda.model.half_math import SOURCE, stream_sigmoid
from orinfer_build.compiler.cuda.common import configure, orin_jit, write_json


@orin_jit
def sigmoid(reference=False, dtype="float16"):
    @T.prim_func
    def check(U: T.Tensor((65536,), dtype), Y: T.Tensor((65536,), dtype)):
        with T.Kernel(256, threads=256) as bx:
            T.import_source(SOURCE)
            for i in T.Parallel(256):
                value = T.cast(U[bx * 256 + i], T.float32)
                if reference:
                    Y[bx * 256 + i] = 1 / (1 + T.exp(-value))
                else:
                    Y[bx * 256 + i] = stream_sigmoid(value, dtype)

    return check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    rows = []
    for dtype in ("float16", "bfloat16"):
        original = (
            torch.arange(65536, dtype=torch.int32)
            .to(torch.uint16)
            .cuda()
            .view(getattr(torch, dtype))
        )
        value = original.clone()
        expected, actual = torch.empty_like(value), torch.empty_like(value)
        reference, candidate = sigmoid(True, dtype), sigmoid(False, dtype)
        reference(value, expected)
        candidate(value, actual)
        torch.cuda.synchronize()
        mask = ~torch.isnan(value)
        assert torch.equal(actual.view(torch.int16)[mask], expected.view(torch.int16)[mask])
        assert bool(torch.isnan(actual[~mask]).all())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            candidate(value, actual)
        value.copy_(original.roll(4093))
        reference(value, expected)
        graph.replay()
        torch.cuda.synchronize()
        mask = ~torch.isnan(value)
        assert torch.equal(actual.view(torch.int16)[mask], expected.view(torch.int16)[mask])
        assert bool(torch.isnan(actual[~mask]).all())
        rows.append(dict(dtype=dtype, codes=65536, exact=True, changed_input_graph=True))
    write_json(args.output / "results.json", dict(complete=True, cases=rows))
    print(rows, flush=True)


if __name__ == "__main__":
    main()
