"""Run converted original expert shards through the complete native A8 FFN."""

import argparse
from pathlib import Path
import os

import numpy as np
from safetensors import safe_open
import torch

from kernels.model.integer_vq import integer_vq, rotate_activation
from kernels.operators.op30_activation_quantization import activation_quantization, launch
from tools.operators.common import configure, benchmark, error, identity, write_json
from tools.quantization.vq import Weights, rotate
from tools.quantization.reference_math import a8, swiglu, floating_ffn


def matrix(path):
    with safe_open(path, framework="np") as f:
        p = f.get_tensor("indices")[0].transpose(1, 0, 2)
        return Weights(
            "e8p",
            p.reshape(p.shape[0], -1).copy(),
            f.get_tensor("table"),
            f.get_tensor("scales")[0],
            f.get_tensor("signs"),
        )


def project(x, w):
    q, s = a8(rotate(x, w.signs).astype(np.float16))
    integer_dot = (q.astype(np.float64) @ w.integer_weights().astype(np.float64).T).astype(
        np.float32
    )
    return ((integer_dot * w.scales.astype(np.float32)[None, :]) * s.astype(np.float32)).astype(
        np.float16
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 17])
    p.add_argument(
        "--expert-id", type=int, default=0, help="Original expert ID represented by the fixture"
    )
    p.add_argument(
        "--activation-source", help="Checkpoint, revision and precision producing captured inputs"
    )
    a = p.parse_args()
    if any(m < 1 for m in a.rows) or a.expert_id < 0:
        p.error("Rows must be positive and expert ID nonnegative")
    configure()
    wg = matrix(a.weights / "layer-00-gate_up-000.safetensors")
    wd = matrix(a.weights / "layer-00-down-000.safetensors")
    graw = np.load(a.original / "gate_up-bf16.npy", mmap_mode="r")[0]
    draw = np.load(a.original / "down-bf16.npy", mmap_mode="r")[0]
    gpu_weights = []
    for w in (wg, wd):
        pp, book, scale = w.gpu_layout()
        gpu_weights.append(
            tuple(torch.from_numpy(v[None].copy()).cuda() for v in (pp, book, scale))
        )
    n, h = wg.validate()
    _, f = wd.validate()
    rng = np.random.default_rng(20261002)
    source = os.environ.get("ORINFER_REFERENCE_ACTIVATIONS")
    captured = None
    if source:
        with np.load(source) as s:
            captured = s["activations"][(s["routed_ids"] == a.expert_id).any(1)]
    elif a.activation_source:
        p.error("--activation-source requires captured activations")
    cases = []
    for m in a.rows:
        raw = (
            captured[:m].astype(np.float16)
            if captured is not None
            else rng.normal(size=(m, h)).astype(np.float16) * 0.2
        )
        if raw.shape != (m, h):
            raise ValueError("Insufficient real expert input rows")
        x = torch.from_numpy(raw).cuda()
        xr = torch.empty_like(x)
        qg = torch.empty_like(x, dtype=torch.int8)
        sg = torch.empty((m, 1), device="cuda", dtype=torch.float16)
        gu = torch.empty((m, n), device="cuda", dtype=torch.float16)
        fr = torch.empty((m, f), device="cuda", dtype=torch.float16)
        qd = torch.empty_like(fr, dtype=torch.int8)
        sd = torch.empty((m, 1), device="cuda", dtype=torch.float16)
        out = torch.empty_like(x)
        rotate_g = rotate_activation(m, h)
        rotate_d = rotate_activation(m, f, swiglu=True)
        quant_g = activation_quantization(h)
        quant_d = activation_quantization(f)
        kg = integer_vq(1, m, n, h, kind="e8p", shared_table=True)
        kd = integer_vq(1, m, h, f, kind="e8p", shared_table=True)
        maskg = torch.zeros(h, device="cuda", dtype=torch.uint8)
        maskd = torch.zeros(f, device="cuda", dtype=torch.uint8)
        signs_g = torch.from_numpy(wg.signs).cuda()
        signs_d = torch.from_numpy(wd.signs).cuda()
        patch = torch.zeros(1, device="cuda", dtype=torch.uint16)

        def run():
            stream = torch.cuda.current_stream().cuda_stream
            rotate_g(x, signs_g, xr)
            launch(quant_g, xr, maskg, qg, sg, stream=stream)
            pp, book, scale = gpu_weights[0]
            kg(qg, pp, book, patch, scale, sg.view(-1), gu)
            rotate_d(gu, signs_d, fr)
            launch(quant_d, fr, maskd, qd, sd, stream=stream)
            pp, book, scale = gpu_weights[1]
            kd(qd, pp, book, patch, scale, sd.view(-1), out)

        timing, graph = benchmark(run, repetitions=5)
        expected = project(swiglu(project(raw, wg)).astype(np.float16), wd)
        compressed_error = error(out, torch.from_numpy(expected).cuda())
        if compressed_error["relative_l2"] >= 0.003:
            print(
                {
                    "rotation": error(
                        xr, torch.from_numpy(rotate(raw, wg.signs).astype(np.float16)).cuda()
                    ),
                    "gate": error(gu, torch.from_numpy(project(raw, wg)).cuda()),
                    "scales": sg.cpu().numpy().tolist(),
                    "gate_range": [float(gu.min()), float(gu.max())],
                    "down_scale": sd.cpu().numpy().tolist(),
                },
                flush=True,
            )
        assert compressed_error["finite"] and compressed_error["relative_l2"] < 0.003, (
            compressed_error
        )
        original = floating_ffn(raw, graw, draw)
        reference_error = error(out, torch.from_numpy(original).cuda())
        x.zero_()
        out.fill_(99)
        graph.replay()
        torch.cuda.synchronize()
        assert bool((out == 0).all())
        cases.append(
            {
                "rows": m,
                "cpu_codec_reference": compressed_error,
                "original_bf16_local_ffn": reference_error,
                "changed_input_graph": True,
                "complete_ffn_timing": timing,
            }
        )
    write_json(
        a.output / "results.json",
        {
            "complete": True,
            "seed": 20261002,
            "cases": cases,
            "input_source": source or "synthetic Gaussian",
            "activation_producer": (a.activation_source or "Community Q2 layer0 capture")
            if source
            else "synthetic Gaussian",
            "expert_id": a.expert_id,
            "local_reference_precision": "Original BF16-derived weights with FP32 local FFN math; FP16 input boundary",
            "inputs": [
                identity(path)
                for path in [
                    a.weights / "layer-00-gate_up-000.safetensors",
                    a.weights / "layer-00-down-000.safetensors",
                    a.original / "gate_up-bf16.npy",
                    a.original / "down-bf16.npy",
                    *([Path(source)] if source else []),
                ]
            ],
            "full_model_quality_verified": False,
            "scope": "One original expert; local FFN reconstruction, not token quality or model TPS",
        },
    )
    print(cases, flush=True)


if __name__ == "__main__":
    main()
