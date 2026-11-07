"""SM87 exact E8P top-8 and beam refinement checks against exhaustive search."""

import argparse
from pathlib import Path

import numpy as np
import torch

from orinfer_build.compiler.cuda.common import configure, write_json
from orinfer_build.quantization.e8p import basis
from orinfer_build.quantization.e8p_gpu import Encoder as Search
from orinfer_build.quantization.e8p_q4_gpu import Encoder
from orinfer_build.quantization.vq import e8p_decode, rotate


def exhaustive(values, grid):
    """Independent FP64 CPU oracle; ties select the lower stored U16 code."""
    rows = []
    for value in values.astype(np.float64):
        scores = (grid * grid - 2 * value[None] * grid).sum(1)
        rows.append(np.argsort(scores, kind="stable")[:8])
    return np.asarray(rows, np.uint16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    rng = np.random.default_rng(20261002)
    grid = e8p_decode(np.arange(65536, dtype=np.uint16), basis()).astype(np.float64)
    raw = np.concatenate(
        [rng.normal(0, scale, (32, 8)).astype(np.float32) for scale in [0.1, 1, 4, 20, 1e4]]
        + [np.full((1, 8), value, np.float32) for value in [0, 0.5, 1, -1, 2, 10]]
        + [grid[::4096].astype(np.float32)]
        + [
            rng.uniform(-1, 1, (32, 8)).astype(np.float32) * np.float32(2.0**exponent)
            for exponent in (-126, -100, -70, -64, -63, -60, -54)
        ]
    )
    raw = np.concatenate(
        [
            raw,
            np.nextafter(raw[:32], np.float32(np.inf)),
            np.nextafter(raw[:32], np.float32(-np.inf)),
        ]
    )
    search = Search()
    results = []
    for count in [1, 3, 33, len(raw)]:
        x = torch.from_numpy(raw[:count]).cuda()
        output = search.nearest_top8(x)
        np.testing.assert_array_equal(output.cpu().numpy(), exhaustive(raw[:count], grid))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = search.nearest_top8(x)
        x.copy_(torch.from_numpy(-raw[:count]).cuda())
        graph.replay()
        np.testing.assert_array_equal(replayed.cpu().numpy(), exhaustive(-raw[:count], grid))
        results.append(dict(kind="top8", vectors=count, changed_input_graph=True))
        print(results[-1], flush=True)
    encoder = Encoder()
    for n, k in [(3, 128), (33, 256)]:
        weights = rng.normal(0, 0.02, (n, k)).astype(np.float32)
        weights[0] = 0
        signs = rng.choice(np.asarray([-1, 1], np.int8), k)
        original = encoder.fit(weights, signs)
        improved = encoder.refine(weights, original)
        repeated = encoder.refine(weights, original)
        np.testing.assert_array_equal(improved.packed, repeated.packed)
        np.testing.assert_array_equal(improved.scales, repeated.scales)
        target = rotate(weights, signs)
        before = np.square(original.dequantize() - target).sum(1, dtype=np.float64)
        after = np.square(improved.dequantize() - target).sum(1, dtype=np.float64)
        assert np.all(after <= before * (1 + 1e-6) + 1e-18), (before, after)
        for invalid in [weights[:, :-1], weights * np.nan]:
            try:
                encoder.refine(invalid, original)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid refinement accepted")
        results.append(
            dict(
                kind="refine",
                shape=[n, k],
                deterministic=True,
                sse_before=float(before.sum()),
                sse_after=float(after.sum()),
            )
        )
        print(results[-1], flush=True)
    write_json(args.output / "result.json", dict(passed=True, checks=results))


if __name__ == "__main__":
    main()
