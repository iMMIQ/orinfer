"""Offline op13 synthetic GDN matrices, inverse accuracy, graphs and timing."""

import argparse
import json
import time
from pathlib import Path

import torch

from abi import parse_host
from common import benchmark, configure, environment, error, export_kernel, identity, write_json
from gdn_reference import chunk_matrices
from kernels.operators.op13_gdn_chunk_solve import HEADS, gdn_chunk_solve, launch

ROOT = Path(__file__).resolve().parents[2]


def make_system(b, tokens, bt, mode="random"):
    chunks = (tokens + bt - 1) // bt
    k = torch.randn((b, 16, chunks, bt, 128), device="cuda")
    if mode == "identical":
        k[..., 1:, :] = k[..., :1, :]
    elif mode == "structured":
        k *= 0.025
        k[..., 0] += 1
        k[..., 1] += torch.linspace(-0.4, 0.4, bt, device="cuda")
    k = torch.nn.functional.normalize(k, dim=-1)
    beta = torch.rand((b, HEADS, chunks, bt), device="cuda")
    g = -torch.rand_like(beta) * 0.1
    if mode in ("identical", "structured", "beta1"):
        beta.fill_(1)
    if mode == "beta0":
        beta.zero_()
    if mode in ("identical", "structured", "g0"):
        g.zero_()
    if mode == "negative_g":
        g.fill_(-100.0)
    if tokens % bt:
        valid = tokens % bt
        k[:, :, -1, valid:] = 0
        beta[:, :, -1, valid:] = 0
        g[:, :, -1, valid:] = 0
    system, qk = chunk_matrices(k, k, g.cumsum(-1), beta, q_scale=1.0)
    del k, beta, g, qk
    return system


def reference(system, dtype=torch.float32):
    n = system.shape[-1]
    ident = torch.eye(n, device=system.device, dtype=dtype).expand_as(system)
    return torch.linalg.solve_triangular(system.to(dtype), ident, upper=False, unitriangular=True)


def double_error(actual, expected):
    actual = actual.double()
    delta = actual - expected
    norm = float(expected.norm())
    return {
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        "relative_l2": float(delta.norm()) / max(norm, 1e-30),
        "reference_l2": norm,
        "max_abs": float(delta.abs().max()),
        "rms_abs": float(delta.square().mean().sqrt()),
        "metric_dtype": "float64",
    }


def check(system, actual, expected=None, full=True):
    if expected is None:
        expected = reference(system)
    metrics = {"inverse_fp32": error(actual, expected)}
    assert metrics["inverse_fp32"]["finite"] and metrics["inverse_fp32"]["relative_l2"] <= 0.002, (
        metrics
    )
    bt = actual.shape[-1]
    assert torch.equal(
        actual.diagonal(dim1=-2, dim2=-1),
        torch.ones_like(actual[..., 0, 0]).unsqueeze(-1).expand(*actual.shape[:-2], bt),
    )
    assert bool((actual.triu(1) == 0).all())
    metrics["diagonal_exact_one"] = metrics["upper_exact_zero"] = True
    if full:
        double_ref = reference(system, torch.float64)
        metrics["inverse_fp64"] = double_error(actual, double_ref)
        # FP64 norms/products expose FP32 recurrence error without adding a
        # second FP32 GEMM rounding point to the measured normalized residual.
        clean = system.tril(-1).double()
        clean.diagonal(dim1=-2, dim2=-1).fill_(1)
        a64 = actual.double()
        product = clean @ a64
        ident = torch.eye(bt, device="cuda", dtype=torch.float64)
        numerator = torch.linalg.matrix_norm(product - ident)
        denominator = torch.linalg.matrix_norm(clean) * torch.linalg.matrix_norm(a64) + bt**0.5
        ratio = numerator / denominator
        metrics["normalized_residual_max"] = float(ratio.max())
        metrics["normalized_residual_mean"] = float(ratio.mean())
        metrics["frobenius_condition_proxy_max"] = float(
            (torch.linalg.matrix_norm(clean) * torch.linalg.matrix_norm(double_ref)).max()
        )
        rhs = torch.randn((*system.shape[:-1], 8), device="cuda")
        metrics["transformed_rhs"] = error(actual @ rhs, expected @ rhs)
        assert metrics["inverse_fp64"]["relative_l2"] <= 0.002, metrics
        assert metrics["normalized_residual_max"] <= 0.001, metrics
        assert metrics["transformed_rhs"]["relative_l2"] <= 0.002, metrics
    return metrics, expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--variant", type=int, choices=(1, 2, 4), default=0)
    parser.add_argument(
        "--algorithm", choices=("columns", "elimination", "registers"), default="registers"
    )
    parser.add_argument("--threads", type=int, choices=(128, 256), default=256)
    args = parser.parse_args()
    out = Path(args.output)
    configure()
    env = environment()
    report = {
        "environment": env,
        "source": identity(ROOT / "kernels/operators/op13_gdn_chunk_solve.py"),
        "reference_source": identity(ROOT / "tools/operators/gdn_reference.py"),
        "input_provenance": "synthetic normalized K/g/beta via shared chunk_matrices; no model trace/quality claim",
        "tuning": [],
        "cases": [],
        "workspace_bytes": 0,
        "resident_parameter_bytes": 0,
        "budget_ms_B1": {"512": 0.080, "2048": 0.320, "8192": 1.280},
    }
    kernels = {}
    for variant in (
        (args.variant,) if args.variant else ((1, 2, 4) if args.algorithm == "columns" else (1,))
    ):
        started = time.perf_counter()
        kernel = gdn_chunk_solve(64, variant, args.algorithm, args.threads)
        prepare_s = time.perf_counter() - started
        kernels[64, variant] = kernel
        system = make_system(1, 512, 64)
        output = torch.empty_like(system)

        def run():
            return launch(kernel, system, output, stream=torch.cuda.current_stream().cuda_stream)

        started = time.perf_counter()
        run()
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - started) * 1000
        metrics, _ = check(system, output)
        timing, graph = benchmark(run, repetitions=32, calls_per_replay=16)
        del graph
        report["tuning"].append(
            {
                "algorithm": args.algorithm,
                "threads": args.threads,
                "matrices_per_block": variant,
                "prepare_s": prepare_s,
                "first_use_ms": first_ms,
                "accuracy": metrics,
                "timing": timing,
            }
        )
        write_json(out / "progress.json", report)
        print(json.dumps(report["tuning"][-1]), flush=True)
    variant = min(report["tuning"], key=lambda x: x["timing"]["median_ms"])["matrices_per_block"]
    report["selected_matrices_per_block"] = variant
    specs = [(1, 512, 64, "random")]
    if not args.quick:
        specs = [
            (b, t, 64, "random") for b in (1, 2, 3, 4, 5, 7, 8) for t in (511, 512, 513, 2048, 8192)
        ]
        specs += [
            (b, t, bt, "random")
            for bt in (16, 32, 64)
            for b in (1, 2, 3, 4, 5, 7, 8)
            for t in (bt - 1, bt + 1)
        ]
        specs += [
            (3, 2 * bt + 1, bt, mode)
            for bt in (16, 32, 64)
            for mode in ("beta0", "beta1", "g0", "negative_g", "identical", "structured")
        ]
    for b, tokens, bt, mode in specs:
        if (bt, variant) not in kernels:
            started = time.perf_counter()
            kernels[bt, variant] = gdn_chunk_solve(bt, variant, args.algorithm, args.threads)
            report.setdefault("compilation", []).append(
                {"BT": bt, "prepare_s": time.perf_counter() - started}
            )
        kernel = kernels[bt, variant]
        system = make_system(b, tokens, bt, mode)
        output = torch.empty_like(system)

        def run():
            return launch(kernel, system, output, stream=torch.cuda.current_stream().cuda_stream)

        run()
        torch.cuda.synchronize()
        accuracy, expected = check(system, output)
        case = {
            "B": b,
            "T": tokens,
            "BT": bt,
            "C": system.shape[2],
            "mode": mode,
            "accuracy": accuracy,
            "input_bytes": system.numel() * 4,
            "output_bytes": output.numel() * 4,
        }
        if tokens % bt:
            valid = tokens % bt
            ident = torch.eye(bt, device="cuda").expand(b, HEADS, bt, bt)
            assert torch.equal(output[:, :, -1, valid:], ident[:, :, valid:])
            assert bool((output[:, :, -1, :valid, valid:] == 0).all())
            case["tail_exact_padded_identity"] = True
        large = b * HEADS * system.shape[2] >= 6144
        case["timing"], graph = benchmark(
            run, repetitions=6 if large else 20, calls_per_replay=4 if large else 16
        )
        # Updating strict lower input and poisoning output proves captured nodes
        # execute. Input diagonal/upper are NaN: neither can influence inversion.
        original = system.clone()
        mask = torch.ones((bt, bt), device="cuda", dtype=torch.bool).tril(-1)
        system.mul_(0.5)
        system.masked_fill_(~mask, float("nan"))
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        case["graph_changed_upper_diagonal_nan"], _ = check(system, output, full=False)
        system.copy_(original)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        case["graph_restored"], _ = check(system, output, expected, full=False)
        assert torch.equal(system, original)
        case["input_immutable"] = True
        if b == 1 and tokens in (512, 2048, 8192) and bt == 64:
            case["single_node_timing"], single = benchmark(run, repetitions=100, calls_per_replay=1)
            del single
        del graph
        del original
        del expected
        output = None
        system = None
        report["cases"].append(case)
        write_json(out / "progress.json", report)
        print(
            json.dumps(
                {
                    "case": [b, tokens, bt, mode],
                    "ms": case["timing"]["median_ms"],
                    "residual": accuracy["normalized_residual_max"],
                }
            ),
            flush=True,
        )
    for (bt, mpb), kernel in kernels.items():
        if mpb != variant:
            continue
        dest = out / f"bt{bt}-mpb{mpb}"
        artifacts = export_kernel(kernel, dest)
        write_json(
            dest / "abi.json",
            {
                "operator": "op13_gdn_chunk_solve",
                "BT": bt,
                "matrices_per_block": mpb,
                "algorithm": args.algorithm,
                "threads": args.threads,
                "sm": 87,
                "toolchain": env,
                "logical_parameters": ["L_fp32[B,48,C,BT,BT]", "A_fp32[same]"],
                "layout": "contiguous row-major; flatten matrix id=((b*48+h)*C+c)",
                "actual_generated_launches": parse_host((dest / "host.txt").read_text()),
                "cooperative_launch": False,
                "workspace_bytes": 0,
                "resident_parameter_bytes": 0,
                "alias_policy": "all allocations disjoint",
                "stream": "explicit caller stream",
                "input_policy": "strict lower only read; implicit diagonal one, upper ignored; finite lower required",
                "rounding": "FP32 SIMT multiply/add with compiler FMA; no TF32/FP16/BF16 conversion",
                "artifacts": artifacts,
            },
        )
    report["peak_torch_validation_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "passed"
    write_json(out / "results.json", report)
    print(json.dumps({"status": "passed", "cases": len(report["cases"]), "selected": variant}))


if __name__ == "__main__":
    main()
