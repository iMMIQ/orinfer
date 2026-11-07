"""Check GPU Q4 fitting against the shared CPU codec and rotated weights."""

import argparse
from pathlib import Path

import numpy as np

from orinfer_build.quantization import row_q4, row_q4_gpu
from orinfer_build.quantization.vq import rotate


def main():
    from orinfer_build.compiler.cuda.common import configure, write_json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    configure()
    rng = np.random.default_rng(20261002)
    cases = []
    for n, k in ((3, 128), (33, 5120), (512, 17408)):
        raw = rng.normal(0, 0.02, (n, k)).astype(np.float32)
        raw[0] = 0
        signs = rng.choice(np.asarray([-1, 1], np.int8), k)
        encoded = row_q4_gpu.fit(raw, signs)
        repeat = row_q4_gpu.fit(raw, signs)
        assert np.array_equal(encoded.packed, repeat.packed)
        assert np.array_equal(encoded.scales, repeat.scales)
        reference = row_q4.fit(raw[:8], signs)
        rotated = rotate(raw[:8], signs)
        actual = encoded.dequantize()[:8]
        mse = np.square(actual - rotated).sum()
        oracle = np.square(reference.dequantize() - rotated).sum()
        assert mse <= oracle * 1.002 + 1e-10, (n, k, mse, oracle)
        assert np.max(np.abs(encoded.integers().astype(np.int16))) <= 127
        assert encoded.bits_per_weight <= 4.5 if n >= 32 else encoded.bits_per_weight < 9
        cases.append(
            dict(
                rows=n,
                width=k,
                deterministic=True,
                gpu_sse=float(mse),
                cpu_sse=float(oracle),
                bits_per_weight=encoded.bits_per_weight,
            )
        )
    write_json(a.output / "results.json", dict(complete=True, cases=cases))


if __name__ == "__main__":
    main()
