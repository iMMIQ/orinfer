"""Offline op12 FP32 reference, shared-head/tail/graph verification and export."""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from abi import parse_host
from common import benchmark, configure, environment, error, export_kernel, identity, write_json
from gdn_reference import chunk_matrices
from kernels.operators.op12_gdn_chunk_matrices import HK, HV, DK, gdn_chunk_matrices, launch

ROOT = Path(__file__).resolve().parents[2]
SCALE = DK**-0.5


def host_checks():
    try:
        gdn_chunk_matrices()
    except TypeError:
        pass
    else:
        raise AssertionError("q_scale must be required")
    for params in (
        {"q_scale": 0},
        {"q_scale": float("nan")},
        {"q_scale": True},
        {"q_scale": SCALE, "bt": 63},
        {"q_scale": SCALE, "qk_dtype": "bfloat16"},
    ):
        try:
            gdn_chunk_matrices(**params)
        except ValueError:
            continue
        raise AssertionError("invalid specialization accepted")
    return {"q_scale_required": True, "invalid_rejected": 5}


def inputs(b, tokens, bt, dtype, mode, scaled=False):
    chunks = (tokens + bt - 1) // bt
    shape = (b, HK, chunks, bt, DK)
    rawq = torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1)
    rawk = torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1)
    # Request/head/channel fingerprints expose incorrect kh=hv//3 mappings.
    if scaled:
        rawq *= SCALE
    q, k = rawq.to(getattr(torch, dtype)), rawk.to(getattr(torch, dtype))
    g = -torch.rand((b, HV, chunks, bt), device="cuda") * 0.08
    beta = torch.rand_like(g)
    if mode == "g0":
        g.zero_()
    elif mode == "beta0":
        beta.zero_()
    elif mode == "beta1":
        beta.fill_(1.0)
    elif mode == "extreme":
        g.fill_(-1000.0)
    if chunks * bt > tokens:
        valid = tokens - (chunks - 1) * bt
        q[:, :, -1, valid:].zero_()
        k[:, :, -1, valid:].zero_()
        rawq[:, :, -1, valid:].zero_()
        rawk[:, :, -1, valid:].zero_()
        g[:, :, -1, valid:].zero_()
        beta[:, :, -1, valid:].zero_()
    return q, k, g.cumsum(-1), beta, rawq, rawk


def check(system, qk, expected, beta):
    metrics = {"L": error(system, expected[0]), "QK": error(qk, expected[1])}
    assert torch.allclose(system, expected[0], atol=2e-6, rtol=3e-4), metrics
    assert torch.allclose(qk, expected[1], atol=2e-6, rtol=3e-4), metrics
    bt = system.shape[-1]
    upper = torch.triu(torch.ones((bt, bt), device="cuda", dtype=torch.bool), diagonal=1)
    # Positive zero bit pattern, not merely floating equality.
    assert bool((system[..., upper].contiguous().view(torch.int32) == 0).all())
    assert bool((qk[..., upper].contiguous().view(torch.int32) == 0).all())
    assert bool((system.diagonal(dim1=-2, dim2=-1) == 1.0).all())
    if bool((beta == 0).all()):
        assert torch.equal(system, torch.eye(bt, device="cuda").expand_as(system))
    metrics.update({"upper_triangle_positive_zero_bits": True, "L_unit_diagonal_exact": True})
    return metrics


def export(kernel, dest, bt, dtype, scale, env):
    artifacts = export_kernel(kernel, dest)
    write_json(
        dest / "abi.json",
        {
            "operator": "op12_gdn_chunk_matrices",
            "sm": 87,
            "BT": bt,
            "HK": HK,
            "HV": HV,
            "DK": DK,
            "q_scale": scale,
            "qk_dtype": dtype,
            "logical_parameters": [
                f"Q_{dtype}[B,16,C,BT,128]",
                f"K_{dtype}[B,16,C,BT,128]",
                "G_fp32[B,48,C,BT]",
                "Beta_fp32[same]",
                "L_fp32[B,48,C,BT,BT]",
                "QK_fp32[same]",
            ],
            "actual_generated_launches": parse_host((dest / "host.txt").read_text()),
            "layout": "contiguous row-major; shared query/key head kh=hv//3",
            "workspace_bytes": 0,
            "resident_parameter_bytes": 0,
            "cooperative_launch": False,
            "stream": "explicit caller stream",
            "alias_policy": "all buffers disjoint; inputs read-only; outputs fully overwritten",
            "rounding": "FP16 tensorcore dot with FP32 accumulation or explicit FP32 SIMT reduction; q_scale/exp/beta in FP32 after dot; no additional Q cast",
            "tail_policy": "caller Q/K=0, g=0, beta=0; cumulative G repeats final valid value",
            "toolchain": env,
            "artifacts": artifacts,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    configure()
    env = environment()
    freeze = out / "source-freeze"
    freeze.mkdir(exist_ok=True)
    paths = [
        "kernels/operators/op12_gdn_chunk_matrices.py",
        "tools/operators/op12_gdn_chunk_matrices.py",
        "tools/operators/gdn_reference.py",
        "tools/operators/common.py",
        "tools/operators/abi.py",
    ]
    frozen = []
    for path in paths:
        dest = freeze / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, dest)
        frozen.append(identity(dest))
    report = {
        "environment": env,
        "source_freeze": frozen,
        "host_checks": host_checks(),
        "reference": "shared FP32 mathematical chunk_matrices; synthetic normalized inputs, not real model trace",
        "TF32": False,
        "cases": [],
        "compilation": [],
        "failures": [],
        "workspace_bytes": 0,
        "resident_parameter_bytes": 0,
        "budget_ms_B1_T512": 0.200,
    }
    specs = [(1, 512, 64, "float16", "random", False)]
    if not args.quick:
        specs = [
            (b, bt + 1, bt, "float16", "random", False)
            for bt in (16, 32, 64)
            for b in (1, 2, 3, 4, 5, 7, 8)
        ]
        specs += [
            (1, t, bt, "float16", "random", False)
            for bt in (16, 32, 64)
            for t in (511, 512, 513, 2048, 8192)
        ]
        specs += [
            (3, 65, bt, "float16", mode, False)
            for bt in (16, 32, 64)
            for mode in ("g0", "beta0", "beta1", "extreme")
        ]
        specs += [
            (3, bt + 1, bt, "float32", "random", scaled)
            for bt in (16, 32, 64)
            for scaled in (False, True)
        ]
    kernels = {}
    for b, tokens, bt, dtype, mode, scaled in specs:
        scale = 1.0 if scaled else SCALE
        key = (bt, dtype, scale)
        if key not in kernels:
            started = time.perf_counter()
            kernels[key] = gdn_chunk_matrices(q_scale=scale, bt=bt, qk_dtype=dtype)
            report["compilation"].append(
                {
                    "BT": bt,
                    "dtype": dtype,
                    "q_scale": scale,
                    "compile_prepare_s": time.perf_counter() - started,
                }
            )
        kernel = kernels[key]
        started = time.perf_counter()
        q, k, gc, beta, rawq, rawk = inputs(b, tokens, bt, dtype, mode, scaled)
        chunks = q.shape[2]
        system = torch.empty((b, HV, chunks, bt, bt), device="cuda")
        qk = torch.empty_like(system)
        torch.cuda.synchronize()
        prepare = time.perf_counter() - started

        def run():
            return launch(
                kernel, q, k, gc, beta, system, qk, stream=torch.cuda.current_stream().cuda_stream
            )

        started = time.perf_counter()
        run()
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - started) * 1000
        expected = chunk_matrices(q, k, gc, beta, q_scale=scale)
        mathematical = chunk_matrices(rawq, rawk, gc, beta, q_scale=scale)
        case = {
            "B": b,
            "T": tokens,
            "BT": bt,
            "C": chunks,
            "mode": mode,
            "dtype": dtype,
            "q_scale": scale,
            "Q_already_scaled": scaled,
            "prepare_input_output_s": prepare,
            "first_use_ms": first_ms,
            "implementation_error_vs_FP32_on_same_input": check(system, qk, expected, beta),
            "upstream_input_cast_error_vs_original_FP32": {
                "L": error(expected[0], mathematical[0]),
                "QK": error(expected[1], mathematical[1]),
            },
            "input_bytes": sum(x.numel() * x.element_size() for x in (q, k, gc, beta)),
            "output_bytes": 8 * system.numel(),
            "workspace_bytes": 0,
        }
        del mathematical, rawq, rawk
        case["timing"], graph = benchmark(run, repetitions=12, calls_per_replay=16)
        # Independent changes exercise every input parameter using stable addresses.
        originals = (q.clone(), k.clone(), gc.clone(), beta.clone())
        case["graph_mutations"] = []
        for name, tensor, factor in [
            ("Q", q, 0.5),
            ("K", k, -0.75),
            ("G", gc, 1.5),
            ("Beta", beta, 0.25),
        ]:
            tensor.mul_(factor)
            system.fill_(float("nan"))
            qk.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            changed = chunk_matrices(q, k, gc, beta, q_scale=scale)
            case["graph_mutations"].append(
                {"input": name, "check": check(system, qk, changed, beta)}
            )
            for dst, src in zip((q, k, gc, beta), originals):
                dst.copy_(src)
            del changed
        system.fill_(float("nan"))
        qk.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        case["graph_restored"] = check(system, qk, expected, beta)
        if chunks * bt > tokens:
            valid = tokens - (chunks - 1) * bt
            assert bool((qk[:, :, -1, valid:] == 0.0).all())
            tail_l = system[:, :, -1, valid:]
            eye_tail = torch.eye(bt, device="cuda")[valid:].expand_as(tail_l)
            assert torch.equal(tail_l, eye_tail)
            case["padded_rows_L_identity_QK_zero"] = True
        del graph
        del originals
        del expected
        q = None
        k = None
        gc = None
        beta = None
        system = None
        qk = None
        report["cases"].append(case)
        write_json(out / "progress.json", report)
    for (bt, dtype, scale), kernel in kernels.items():
        export(kernel, out / f"bt{bt}-{dtype}-scale{scale:.8g}", bt, dtype, scale, env)
    report["peak_torch_validation_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "passed"
    write_json(out / "results.json", report)
    print(
        json.dumps(
            {
                "status": "passed",
                "cases": len(report["cases"]),
                "compilation": report["compilation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
