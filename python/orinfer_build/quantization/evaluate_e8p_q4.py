"""Compare real BF16-derived matrix fixtures and full-shape Q4 projection cost.

Input safetensors contain F32 weight and I8 signs. Results are component
measurements, not full-model quality or throughput acceptance.
"""

from orinfer_build.common.paths import ROOT


import argparse
import json
from pathlib import Path
import time

import numpy as np
from safetensors.numpy import load_file
import torch

from kernels.cuda.model.e8p_q4 import projection
from kernels.cuda.model.q4_projection import q4_decode
from orinfer_build.compiler.compile import primitive
from orinfer_build.common.files import file_hash
from orinfer_build.compiler.cuda.common import benchmark, configure, write_json
from orinfer_build.quantization.e8p import basis
from orinfer_build.quantization.e8p_q4 import Weights
from orinfer_build.quantization.e8p_q4_gpu import Encoder
from orinfer_build.quantization.row_q4_gpu import fit as scalar_fit
from orinfer_build.quantization.vq import e8p_selector_table, rotate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--full-matrix", type=Path)
    args = parser.parse_args()
    configure()
    encoder = Encoder()
    metrics = []
    for path in sorted(args.weights.glob("[0-9]*.safetensors")):
        arrays = load_file(str(path))
        raw, signs = arrays["weight"], arrays["signs"]
        target = rotate(raw, signs)
        baseline = scalar_fit(raw, signs)
        baseline_mse = float(
            np.square(baseline.dequantize() - target).sum() / np.square(target).sum()
        )
        results = []
        for coefficients in [(3, 1), (4, 1), (7, 2)]:
            start = time.monotonic()
            candidate = encoder.fit(raw, signs, coefficients=coefficients)
            elapsed = time.monotonic() - start
            candidate.save(
                args.output / f"{path.stem}-{coefficients[0]}-{coefficients[1]}.safetensors"
            )
            mse = float(np.square(candidate.dequantize() - target).sum() / np.square(target).sum())
            results.append(
                dict(
                    coefficients=coefficients,
                    normalized_mse=mse,
                    scalar_mse_ratio=mse / baseline_mse,
                    fit_s=elapsed,
                    bits_per_weight=candidate.bits_per_weight,
                )
            )
        record = dict(
            fixture=path.name,
            source=json.loads(path.with_suffix(".json").read_text()),
            scalar_normalized_mse=baseline_mse,
            candidates=results,
        )
        metrics.append(record)
        write_json(
            args.output / "quality.json", dict(scope=__doc__, seed=20261002, matrices=metrics)
        )
        print("QUALITY", record, flush=True)
    scores = {
        tuple(c): sum(
            np.log(
                next(x for x in r["candidates"] if tuple(x["coefficients"]) == tuple(c))[
                    "scalar_mse_ratio"
                ]
            )
            for r in metrics
        )
        / len(metrics)
        for c in [(3, 1), (4, 1), (7, 2)]
    }
    coefficients = min(scores, key=scores.get)
    write_json(
        args.output / "selection.json",
        dict(
            coefficients=coefficients,
            geometric_mse_ratio={str(k): float(np.exp(v)) for k, v in scores.items()},
            selection="Weight-only measured reconstruction error across all input fixtures; not model NLL.",
        ),
    )
    if args.full_matrix is None:
        return
    arrays = load_file(str(args.full_matrix))
    raw, signs = arrays["weight"], arrays["signs"]
    n, k = raw.shape
    packed = args.output / "full-e8p.safetensors"
    provenance = dict(
        fixture_sha256=file_hash(args.full_matrix),
        fitting_source=[
            file_hash(Path(__file__).with_name(name))
            for name in ["e8p_q4.py", "e8p_q4_gpu.py", "e8p_gpu.py", "e8p.py", "vq.py"]
        ],
        encoder_source=file_hash(ROOT / "kernels/cuda/model/e8p_encode.py"),
    )
    receipt = packed.with_suffix(".source.json")
    if packed.exists():
        if not receipt.exists() or json.loads(receipt.read_text()) != provenance:
            raise ValueError("Existing full matrix source changed; use a new output directory")
        candidate = Weights.load(packed)
        if candidate.coefficients != coefficients:
            raise ValueError("Existing full matrix has different codec coefficients")
    else:
        codes = []
        scales = []
        for first in range(0, n, 128):
            start = time.monotonic()
            w = encoder.fit(raw[first : first + 128], signs, coefficients=coefficients)
            codes.append(w.packed)
            scales.append(w.scales)
            print("FULL FIT", first, time.monotonic() - start, flush=True)
        candidate = Weights(np.concatenate(codes), np.concatenate(scales), signs, coefficients)
        candidate.save(packed)
        write_json(receipt, provenance)
    baseline = scalar_fit(raw, signs)
    target = rotate(raw, signs)
    full_quality = dict(
        shape=[n, k],
        coefficients=coefficients,
        scalar_normalized_mse=float(
            np.square(baseline.dequantize() - target).sum() / np.square(target).sum()
        ),
        candidate_normalized_mse=float(
            np.square(candidate.dequantize() - target).sum() / np.square(target).sum()
        ),
        fixture_sha256=provenance["fixture_sha256"],
    )
    write_json(args.output / "full-quality.json", full_quality)
    del raw, target, arrays
    book = torch.from_numpy(e8p_selector_table(basis())).cuda()
    weights = {
        "scalar": torch.from_numpy(baseline.packed).cuda().view(torch.uint16),
        "e8p": torch.from_numpy(candidate.packed).cuda(),
    }
    scales = {
        "scalar": torch.from_numpy(baseline.scales).cuda(),
        "e8p": torch.from_numpy(candidate.scales).cuda(),
    }
    decoded = {
        "scalar": torch.from_numpy(baseline.integers().astype(np.float32)).cuda(),
        "e8p": torch.from_numpy(candidate.integers().astype(np.float32)).cuda(),
    }
    configurations = (
        [
            dict(algorithm="dp4a", block_m=1, block_n=16, block_k=128, stages=2),
            dict(algorithm="dp4a", block_m=4, block_n=16, block_k=128, stages=2),
        ]
        + [
            dict(algorithm="tensor_core", block_m=m, block_n=bn, block_k=128, stages=2)
            for m in [8, 16, 32, 64]
            for bn in [64, 128]
        ]
        + [
            dict(algorithm="tensor_core", block_m=128, block_n=bn, block_k=bk, stages=stages)
            for bn in [64, 128]
            for bk, stages in [(128, 1), (128, 2), (256, 1)]
        ]
    )
    kernels = {}
    for kind in ["scalar", "e8p"]:
        for i, cfg in enumerate(configurations):
            try:
                fn = projection if kind == "e8p" else q4_decode
                kwargs = dict(cfg, output_dtype="float32", dynamic_rows=True)
                if kind == "e8p":
                    kwargs["coefficients"] = coefficients
                kernels[kind, i] = primitive(fn, 4096, n, k, **kwargs)
            except Exception as error:
                print("REJECTED", kind, cfg, str(error), flush=True)
    measurements = []
    for rows in [1, 2, 4, 8, 17, 512]:
        x = torch.randint(-127, 128, (rows, k), device="cuda", dtype=torch.int8)
        act_scale = torch.full((rows,), 0.001, device="cuda", dtype=torch.float16)
        out = torch.empty((rows, n), device="cuda")
        for kind in ["scalar", "e8p"]:
            reference = (
                (x.float() @ decoded[kind].T)
                * scales[kind].float()[None]
                * act_scale.float()[:, None]
            )
            for i, cfg in enumerate(configurations):
                if (kind, i) not in kernels:
                    continue
                kernel = kernels[kind, i]
                activation = x.view(torch.int32) if cfg["algorithm"] == "dp4a" else x
                launch_args = (
                    (activation, weights[kind], book, scales[kind], act_scale, out, rows)
                    if kind == "e8p"
                    else (activation, weights[kind], scales[kind], act_scale, out, rows)
                )

                def run():
                    kernel.adapter.func(
                        *launch_args, stream=torch.cuda.current_stream().cuda_stream
                    )

                run()
                torch.cuda.synchronize()
                torch.testing.assert_close(out, reference, rtol=1e-4, atol=1e-5)
                timing, _ = benchmark(run, repetitions=8)
                record = dict(rows=rows, kind=kind, config=cfg, ms=timing["median_ms"])
                measurements.append(record)
                print("TIMING", record, flush=True)
                write_json(
                    args.output / "performance.json",
                    dict(
                        scope="Actual full BF16-derived matrix; isolated CUDA-event graph replay, kernel only. No full-model TPS claim.",
                        shape=[n, k],
                        coefficients=coefficients,
                        measurements=measurements,
                    ),
                )


if __name__ == "__main__":
    main()
