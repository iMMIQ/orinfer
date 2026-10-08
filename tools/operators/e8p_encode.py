"""Validate fused offline encoding against exhaustive lattice search."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from tools.operators.common import configure, benchmark, export_kernel, write_json
from tools.quantization.e8p_gpu import Encoder
from tools.quantization.vq import e8p_decode, rotate, save, load
from tools.quantization.embedding_vq import pack, decode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", type=Path)
    p.add_argument("--embedding-weights", type=Path)
    a = p.parse_args()
    configure()
    enc = Encoder()
    rng = np.random.default_rng(20261002)
    grid = e8p_decode(np.arange(65536, dtype=np.uint16), enc.table).astype(np.float32)
    cases = []
    for count in (1, 3, 17, 129):
        values = (rng.normal(size=(count, 8)) * 4).astype(np.float32)
        x = torch.from_numpy(values).cuda()
        codes = enc.nearest(x)

        def run():
            enc.kernels[count](x, enc.book, codes)

        timing, graph = benchmark(run, repetitions=3)
        actual = ((values - grid[codes.cpu().numpy()]) ** 2).sum(1)
        expected = np.array([((grid - v) ** 2).sum(1).min() for v in values])
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=1e-5)
        x.copy_(enc.grid[123].expand_as(x))
        graph.replay()
        torch.cuda.synchronize()
        assert bool((codes == 123).all())
        cases.append(
            {
                "vectors": count,
                "exhaustive_nearest": True,
                "changed_input_graph": True,
                "timing": timing,
                "export": export_kernel(enc.kernels[count], a.output / f"encode-{count}"),
            }
        )
    raw = (
        np.load(a.weights, mmap_mode="r")[0]
        if a.weights
        else rng.normal(size=(65, 256)).astype(np.float32) * 0.02
    )
    signs = rng.choice(np.array([-1, 1], np.int8), raw.shape[1])
    started = time.perf_counter()
    w = enc.fit(raw, signs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    save(a.output / "weights.safetensors", w, {"fit": "weight-only", "seed": 20261002})
    restored = load(a.output / "weights.safetensors")
    np.testing.assert_array_equal(restored.indices, w.indices)
    target = rotate(raw, signs)
    relative = float(np.linalg.norm(w.dequantize() - target) / np.linalg.norm(target))
    assert relative < 0.4, relative
    embedding = (
        np.load(a.embedding_weights)
        if a.embedding_weights
        else rng.normal(size=(127, 160)).astype(np.float32) * 0.02
    )
    embedding_signs = rng.choice(np.array([-1, 1], np.int8), 160)
    embedding_codes, embedding_scales = enc.fit_arrays(embedding, embedding_signs, rotation="full")
    reconstructed = decode(pack(embedding_codes, embedding_scales), enc.table, embedding_signs)
    embedding_relative = float(
        np.linalg.norm(reconstructed - embedding) / np.linalg.norm(embedding)
    )
    assert embedding_relative < 0.4, embedding_relative
    write_json(
        a.output / "results.json",
        {
            "complete": True,
            "cases": cases,
            "matrix": list(raw.shape),
            "weight_relative_l2": relative,
            "fit_seconds": elapsed,
            "bits_per_weight": 8 * w.nbytes / raw.size,
            "seed": 20261002,
            "embedding160_relative_l2": embedding_relative,
        },
    )
    print({"fit_seconds": elapsed, "weight_relative_l2": relative}, flush=True)


if __name__ == "__main__":
    main()
