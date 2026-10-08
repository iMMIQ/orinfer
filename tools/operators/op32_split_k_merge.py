"""Independent op32 numerical/graph/ABI and real op05/op18 chain evidence."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import torch
import tilelang.language as T
from common import (
    configure,
    environment,
    error,
    export_kernel,
    identity,
    tensor_sha,
    write_json,
    benchmark,
)
from abi import parse_host
from kernels.operators.op32_split_k_merge import split_k_merge, split_k_merge_residual
from kernels.projections.candidates import splitk_reduce
from kernels.operators.op05_ffn_down import ffn_down_partial
from kernels.operators.op18_mixer_out import mixer_out_partial
import op05_ffn_down as down_evidence
import op18_mixer_out as mixer_evidence

ROOT = Path(__file__).resolve().parents[2]
ROWS = (1, 2, 3, 4, 5, 7, 8)
SPLITS = (1, 2, 3, 4, 5, 7, 8, 16, 31)


def ordered(p):
    out = torch.zeros_like(p[0])
    for sk in range(len(p)):
        out = out + p[sk]
    return out


def double_error(actual, reference):
    delta = actual.double() - reference.double()
    norm = float(reference.double().norm())
    return dict(
        relative_l2=float(delta.norm()) / max(norm, 1e-30),
        max_abs=float(delta.abs().max()),
        finite=bool(torch.isfinite(actual).all()),
    )


def exact(actual, reference):
    # Signed zero and all finite FP32 bit patterns matter here.
    assert torch.equal(
        actual.view(torch.int32 if actual.dtype == torch.float32 else torch.int16),
        reference.view(torch.int32 if reference.dtype == torch.float32 else torch.int16),
    )


def export(kernel, directory, contract):
    artifacts = export_kernel(kernel, directory)
    abi = dict(
        schema_version=1,
        target="sm_87",
        cooperative=False,
        contract=contract,
        dynamic_dimension="M inferred from tensor shape at each invocation",
        generated_launches=parse_host((directory / "host.txt").read_text()),
        toolchain={k: str(environment()[k]) for k in ("torch", "tilelang", "cuda")},
        artifact=artifacts,
    )
    for name, flag in [("resources.txt", "--dump-resource-usage"), ("sass.txt", "--dump-sass")]:
        process = subprocess.run(
            ["/usr/local/cuda/bin/cuobjdump", flag, str(directory / "kernel.cubin")],
            capture_output=True,
            text=True,
        )
        (directory / name).write_text(process.stdout + process.stderr)
        abi[name] = dict(exit_code=process.returncode, **identity(directory / name))
    write_json(directory / "abi.json", abi)
    return abi


def replay_check(run, inputs, output):
    run()
    torch.cuda.synchronize()
    expected = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    exact(output, expected)
    for name, tensor in inputs:
        saved = tensor.clone()
        # Powers of two preserve exact representability and visibly change output.
        tensor.mul_(0.5)
        run()
        torch.cuda.synchronize()
        changed = output.clone()
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        exact(output, changed)
        assert not torch.equal(output, expected), name
        tensor.copy_(saved)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        exact(output, expected)
    return dict(poison_changed_inputs_restore=[name for name, _ in inputs], captured_nodes=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-chain", action="store_true")
    parser.add_argument(
        "--supplement-only",
        action="store_true",
        help="Only cancellation/extreme/rounding checks; no repeated full timing",
    )
    args = parser.parse_args()
    configure()
    report = dict(
        environment=environment(),
        numerical=[],
        timings=[],
        chains=[],
        ABI=[],
        precision="ordered FP32 split sum; output one F16/F32 cast; residual F32(F16(sum))+F32(R)",
        budget="op32 included in complete projection budget; down .280ms/M1,1.8ms/M512; mixer .105ms/M1,.600ms/M512",
        model_or_quantization_quality_evaluated=False,
    )

    def save():
        write_json(args.output / "results.json", report)

    dependencies = [
        "kernels/operators/op32_split_k_merge.py",
        "tools/operators/op32_split_k_merge.py",
        "kernels/operators/op05_ffn_down.py",
        "kernels/operators/op18_mixer_out.py",
        "kernels/operators/op03_ffn_gate_up.py",
        "kernels/projections/candidates.py",
        "tools/operators/op05_ffn_down.py",
        "tools/operators/op18_mixer_out.py",
        "tools/operators/common.py",
        "tools/operators/abi.py",
        "tools/projections/decode_common.py",
    ]
    report["frozen_dependencies"] = []
    for relative in dependencies:
        destination = args.output / "measurement-source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        report["frozen_dependencies"].append(identity(destination))
    for keywords in (
        {"M": 0},
        {"N": 0},
        {"SPLIT": 0},
        {"output_dtype": "bfloat16"},
        {"threads": 32},
    ):
        try:
            split_k_merge(**keywords)
        except ValueError:
            continue
        raise AssertionError(("invalid API accepted", keywords))
    report["API_rejections"] = ["M0", "N0", "SPLIT0", "BF16 output", "threads32"]
    cache = {}

    def build(s, n, dtype="float32", residual=False):
        key = (s, n, dtype, residual)
        if key not in cache:
            started = time.perf_counter()
            kernel = (
                split_k_merge_residual(N=n, SPLIT=s, residual_dtype=dtype)
                if residual
                else split_k_merge(N=n, SPLIT=s, output_dtype=dtype)
            )
            cache[key] = kernel
            # Export all compiled variants, with actual host ABI and frozen cubin.
            folder = (
                args.output / "compiled" / f"S{s}_N{n}_{dtype}_{'residual' if residual else 'sum'}"
            )
            abi = export(
                kernel,
                folder,
                dict(
                    partial=[s, "M", n, "float32", "SMN contiguous"],
                    output=["M", n, "float32" if residual else dtype, "MN contiguous"],
                    residual=dtype if residual else None,
                    internal_workspace_bytes=0,
                ),
            )
            report["ABI"].append(
                dict(
                    S=s,
                    N=n,
                    dtype=dtype,
                    residual=residual,
                    compile_or_cache_s=time.perf_counter() - started,
                    manifest=abi,
                )
            )
            save()
        return cache[key]

    # All irregular S with small column tails and all row tails; real widths S8.
    cases = [(s, n, ROWS) for s in SPLITS for n in (1, 17, 257)]
    cases += [(s, 5120, (7,)) for s in SPLITS if s != 8]
    cases += [(8, n, ROWS + (512,)) for n in (5120, 6144, 16384, 34816)]
    if args.supplement_only:
        cases = []
    for s, n, rows in cases:
        k32, k16 = build(s, n), build(s, n, "float16")
        for m in rows:
            p = torch.randn((s, m, n), device="cuda", dtype=torch.float32) * 0.25
            before = tensor_sha(p)
            reference = ordered(p)
            y32 = torch.empty((m, n), device="cuda", dtype=torch.float32)
            y16 = torch.empty((m, n), device="cuda", dtype=torch.float16)
            k32(p, y32)
            k16(p, y16)
            torch.cuda.synchronize()
            exact(y32, reference)
            exact(y16, reference.half())
            assert tensor_sha(p) == before, "partial modified"
            # Requests/columns are independent: rows as separate calls must match.
            if n == 17 and m == 7:
                separate = torch.empty_like(y32)
                for row in range(m):
                    tmp = torch.empty((1, n), device="cuda")
                    k32(p[:, row : row + 1].contiguous(), tmp)
                    separate[row].copy_(tmp[0])
                exact(separate, y32)
            result = dict(
                S=s,
                M=m,
                N=n,
                ordered_fp32_and_fp16_bit_exact=True,
                partial_unchanged=True,
                error_vs_fp64=double_error(y32, p.double().sum(0)),
                partial_bytes=p.numel() * 4,
                output_bytes=y16.numel() * 2,
            )
            if s == 8 and n in (5120, 6144, 16384, 34816) and m in (1, 7, 512):

                def run():
                    return k16(p, y16, stream=torch.cuda.current_stream().cuda_stream)

                result["graph"] = replay_check(run, [("Partial", p)], y16)
                measured, _ = benchmark(run, repetitions=20 if m < 512 else 5, calls_per_replay=16)
                old = splitk_reduce(T.dynamic("M"), n, SPLIT=s)
                old_y = torch.empty_like(y16)

                def old_run():
                    return old(p, old_y, stream=torch.cuda.current_stream().cuda_stream)

                old_timing, _ = benchmark(
                    old_run, repetitions=20 if m < 512 else 5, calls_per_replay=16
                )
                exact(old_y, y16)
                report["timings"].append(
                    dict(
                        S=s,
                        M=m,
                        N=n,
                        new=measured,
                        old_candidates_reduce=old_timing,
                        logical_bytes=(s * 4 + 2) * m * n,
                        workspace_bytes=p.numel() * 4,
                        internal_workspace_bytes=0,
                    )
                )
            report["numerical"].append(result)
            del p, reference, y32, y16
        save()
    # Cancellation and permutations: floating sum isn't associative. Check error
    # against FP64 with a sum(abs(partial))-scaled forward-error bound, not L2 near zero.
    for s in SPLITS:
        n, m = 17, 7
        kernel = build(s, n)
        pattern = torch.tensor(
            [16777216.0, 1.0, -16777216.0, -1.0, 2.0**-120, -(2.0**-120), 65504.0, -65504.0],
            device="cuda",
        )
        p = (
            pattern[torch.arange(s, device="cuda") % len(pattern)][:, None, None]
            .expand(s, m, n)
            .clone()
        )
        p[:, :, 1::2].neg_()
        y = torch.empty((m, n), device="cuda")
        reference = p.double().sum(0)
        bound = torch.finfo(torch.float32).eps * s * p.double().abs().sum(0) + 1e-30
        permutations = []
        for permutation in (
            torch.arange(s, device="cuda"),
            torch.arange(s - 1, -1, -1, device="cuda"),
            torch.randperm(s, device="cuda"),
        ):
            permuted = p[permutation].contiguous()
            kernel(permuted, y)
            torch.cuda.synchronize()
            exact(y, ordered(permuted))
            assert bool(((y.double() - reference).abs() <= bound).all())
            permutations.append(
                dict(
                    order=permutation.cpu().tolist(),
                    max_abs=float((y.double() - reference).abs().max()),
                    max_forward_bound=float(bound.max()),
                )
            )
        report["numerical"].append(dict(S=s, M=m, N=n, cancellation_permutations=permutations))
    # FP16 rounding / residual-once contract, nonaligned real-width-independent tails.
    for dtype in ("float16", "float32"):
        for s, n, m in (
            ()
            if args.supplement_only
            else ((1, 17, 7), (3, 257, 5), (8, 5120, 1), (8, 5120, 512), (31, 17, 3))
        ):
            kernel = build(s, n, dtype, True)
            p = torch.randn((s, m, n), device="cuda") * 0.25
            r = torch.randn((m, n), device="cuda", dtype=getattr(torch, dtype)) * 0.25
            y = torch.empty((m, n), device="cuda")

            def run():
                return kernel(p, r, y, stream=torch.cuda.current_stream().cuda_stream)

            run()
            torch.cuda.synchronize()
            exact(y, ordered(p).half().float() + r.float())
            graph_check = replay_check(run, [("Partial", p), ("R", r)], y)
            measured, _ = benchmark(run, repetitions=20 if m < 512 else 5, calls_per_replay=16)
            report["timings"].append(
                dict(
                    S=s,
                    N=n,
                    M=m,
                    residual_dtype=dtype,
                    native_rounding_bit_exact=True,
                    graph=graph_check,
                    new=measured,
                    logical_bytes=(s * 4 + (2 if dtype == "float16" else 4) + 4) * m * n,
                )
            )
    # Signed zeros, subnormals, largest finite F32, explicit overflow and NaN/Inf.
    kernel = build(1, 17)
    patterns = torch.tensor(
        [
            0.0,
            -0.0,
            2.0**-149,
            -(2.0**-149),
            torch.finfo(torch.float32).max,
            -torch.finfo(torch.float32).max,
            float("inf"),
            -float("inf"),
            float("nan"),
        ],
        device="cuda",
    )
    p = patterns[torch.arange(17, device="cuda") % len(patterns)][None, None, :].contiguous()
    y = torch.empty((1, 17), device="cuda")
    kernel(p, y)
    torch.cuda.synchronize()
    ref = ordered(p)
    mask = ~torch.isnan(ref)
    exact(y[mask], ref[mask])
    assert torch.equal(torch.isnan(y), torch.isnan(ref))
    # Random raw F32 bit patterns with finite exponents span far beyond normal
    # randn inputs. S1 isolates IEEE add-zero behavior; row/column tails are real.
    bits = torch.randint(-(2**31), 2**31 - 1, (1, 7, 17), device="cuda", dtype=torch.int32)
    bits.bitwise_and_(-16777217)  # Clear one exponent bit: every value finite.
    raw = bits.view(torch.float32)
    y = torch.empty((7, 17), device="cuda")
    kernel(raw, y)
    torch.cuda.synchronize()
    exact(y, ordered(raw))
    overflow = build(2, 17, "float16")
    p = torch.full((2, 3, 17), 65504.0, device="cuda")
    y16 = torch.empty((3, 17), device="cuda", dtype=torch.float16)
    overflow(p, y16)
    torch.cuda.synchronize()
    exact(y16, ordered(p).half())
    assert bool(torch.isposinf(y16).all())
    # Explicit witness separates native FP16 projection rounding from adding
    # R to the unrounded FP32 sum; this must stay true through compilation.
    rk = build(3, 17, "float32", True)
    p = torch.full((3, 1, 17), 0.3333, device="cuda")
    r = torch.full((1, 17), 0.125, device="cuda")
    y = torch.empty_like(r)
    rk(p, r, y)
    torch.cuda.synchronize()
    exact(y, ordered(p).half().float() + r)
    assert not torch.equal(y, ordered(p) + r)
    report["extreme"] = {
        "signed_zero_subnormal_maxfinite_nan_inf": True,
        "finite_raw_float32_bitpatterns": True,
        "fp16_overflow_matches_explicit_cast": True,
        "native_residual_rounding_witness": True,
        "NaN_payload_identity_promised": False,
    }
    save()
    if not args.skip_chain and not args.supplement_only:
        for name, evidence, producer, k, budget in [
            ("op05", down_evidence, ffn_down_partial, 17408, {1: 0.280, 512: 1.8}),
            ("op18", mixer_evidence, mixer_out_partial, 6144, {1: 0.105, 512: 0.600}),
        ]:
            p0, s0, z0, q, source = evidence.load_weights(0)
            p, s, z = p0.cuda(), s0.cuda(), z0.cuda()
            w = evidence.real.dequant(q, s0, z0).float()
            partial_kernel = producer(T.dynamic("M"), implementation="register", SPLIT=8)
            projection_abi = export(
                partial_kernel,
                args.output / "compiled" / f"{name}_real_register_partial",
                dict(input=["M", k, "float16"], partial=[8, "M", 5120, "float32"]),
            )
            merge = build(8, 5120, "float16")
            old = splitk_reduce(T.dynamic("M"), 5120, SPLIT=8)
            old_abi = export(
                old,
                args.output / "compiled" / f"{name}_old_reduce",
                dict(partial=[8, "M", 5120, "float32"], output=["M", 5120, "float16"]),
            )
            for phase in ("decode", "prefill"):
                for m, cpu_a, origin in evidence.load_inputs(0, phase):
                    if m not in ROWS + (512,):
                        continue
                    a = cpu_a.cuda()
                    partial = torch.empty((8, m, 5120), device="cuda")
                    y = torch.empty((m, 5120), device="cuda", dtype=torch.float16)
                    old_y = torch.empty_like(y)
                    ref = a.float() @ w.T

                    def run():
                        stream = torch.cuda.current_stream().cuda_stream
                        partial_kernel(a, p, s, z, partial, stream=stream)
                        merge(partial, y, stream=stream)

                    def old_run():
                        stream = torch.cuda.current_stream().cuda_stream
                        partial_kernel(a, p, s, z, partial, stream=stream)
                        old(partial, old_y, stream=stream)

                    started = time.perf_counter()
                    run()
                    torch.cuda.synchronize()
                    first = time.perf_counter() - started
                    numerical = error(y, ref)
                    assert numerical["finite"] and numerical["relative_l2"] <= 0.002, numerical
                    measured, graph = benchmark(
                        run, repetitions=20 if m < 512 else 3, calls_per_replay=16 if m < 512 else 4
                    )
                    expected = y.clone()
                    expected_p = partial.clone()
                    y.fill_(float("nan"))
                    partial.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    exact(y, expected)
                    exact(partial, expected_p)
                    saved = a.clone()
                    a.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    assert bool((y == 0).all()) and bool((partial == 0).all())
                    a.copy_(saved)
                    graph.replay()
                    torch.cuda.synchronize()
                    exact(y, expected)
                    exact(partial, expected_p)
                    old_timing, _ = benchmark(
                        old_run,
                        repetitions=20 if m < 512 else 3,
                        calls_per_replay=16 if m < 512 else 4,
                    )
                    exact(y, old_y)
                    report["chains"].append(
                        dict(
                            producer=name,
                            layer=0,
                            M=m,
                            K=k,
                            N=5120,
                            S=8,
                            phase=phase,
                            activation=origin,
                            weight_source=source,
                            checkpoint_sha256=evidence.real.checkpoint_sha256(evidence.real.MODEL),
                            projection_ABI=projection_abi,
                            old_merge_ABI=old_abi,
                            error=numerical,
                            new_projection_plus_merge=measured,
                            old_projection_plus_merge=old_timing,
                            first_launch_host_s=first,
                            workspace_bytes=partial.numel() * 4,
                            output_bytes=y.numel() * 2,
                            resident_weight_bytes=p.numel() + s.numel() * 2 + z.numel(),
                            budget_ms=budget.get(m),
                            graph_partial_output_poison_changed_X_restore=True,
                        )
                    )
                    print(
                        json.dumps(
                            dict(
                                producer=name,
                                M=m,
                                error=numerical,
                                new_ms=measured["median_ms"],
                                old_ms=old_timing["median_ms"],
                            )
                        ),
                        flush=True,
                    )
                    save()
                    del a, partial, y, old_y, ref, expected, expected_p, saved
            del p, s, z, p0, s0, z0, q, w
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["reference_buffers_note"] = (
        "FP32 expanded weight/ref allocation excluded from production resident weights/workspace"
    )
    report["status"] = "passed"
    save()


if __name__ == "__main__":
    main()
