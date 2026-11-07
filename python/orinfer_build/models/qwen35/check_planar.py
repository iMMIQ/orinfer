"""Bit-exact planar SwiGLU/rotation against interleaved input, including replay."""

import argparse
from pathlib import Path
import torch
from kernels.cuda.model.rotation_a8 import (
    rotate_activation_a8,
    rotate_activation_tiles,
    quantize_rotation_tiles,
)
from orinfer_build.compiler.compile import primitive
from orinfer_build.compiler.cuda.common import configure, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    configure()
    results = []
    capacity = 4096

    def launch(kernel, *values):
        kernel.adapter.func(*values, stream=torch.cuda.current_stream().cuda_stream)

    for k in (256, 5120, 17408):
        reference = primitive(rotate_activation_a8, capacity, k, swiglu=True, dynamic_rows=True)
        signs = torch.randint(0, 2, (k,), device="cuda", dtype=torch.int8) * 2 - 1
        for rows in (1, 2, 3, 17, 129, 512, 4096):
            original = torch.randn((rows, 2 * k), device="cuda", dtype=torch.float16) * 0.2
            if rows > 1:
                original[0].zero_()
            x = torch.full((capacity + rows, k), float("nan"), device="cuda", dtype=torch.float16)
            expected = torch.empty((rows, k), device="cuda", dtype=torch.int8)
            expected_s = torch.empty(rows, device="cuda", dtype=torch.float16)

            def change():
                original.neg_()
                x[:rows].copy_(original[:, :k])
                x[capacity:].copy_(original[:, k:])

            change()
            for width in (256, 512, 1024):
                first = primitive(
                    rotate_activation_tiles,
                    capacity,
                    k,
                    width,
                    True,
                    dynamic_rows=True,
                    plane_rows=capacity,
                )
                second = primitive(quantize_rotation_tiles, capacity, k, width, dynamic_rows=True)
                q = torch.full((rows + 1, k), 91, device="cuda", dtype=torch.int8)
                s = torch.full((rows + 1,), 91, device="cuda", dtype=torch.float16)
                rotated = torch.full((rows + 1, k), 91, device="cuda", dtype=torch.float16)
                maxima = torch.full((rows + 1, (k + width - 1) // width), 91, device="cuda")

                def run():
                    launch(first, x, signs, rotated, maxima, rows)
                    launch(second, rotated, maxima, q, s, rows)

                def check():
                    launch(reference, original, signs, expected, expected_s, rows)
                    assert torch.equal(q[:rows], expected) and torch.equal(s[:rows], expected_s)
                    assert all(bool(torch.all(v[rows:] == 91)) for v in (q, s, rotated, maxima))

                run()
                check()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                change()
                q[:rows].fill_(91)
                s[:rows].fill_(91)
                graph.replay()
                check()
                result = dict(k=k, rows=rows, block_k=width, exact=True, changed_input_graph=True)
                results.append(result)
                print(result, flush=True)
    write_json(args.output / "result.json", dict(passed=True, checks=results))


if __name__ == "__main__":
    main()
