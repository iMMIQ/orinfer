"""Offline full-vocabulary correctness, deterministic RNG, graph and ABI suite."""

import argparse
import gc
import math
import re
import time
from pathlib import Path
import torch
from common import ROOT, benchmark, configure, environment, export_kernel, identity, write_json
from abi import parse_host
from kernels.operators.op25_token_selection import (
    VOCAB,
    counter_uniform,
    topk_partials,
    topk_merge,
    sampling_mass,
    sampling_prefix,
    sampling_select,
    launch,
)

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
BATCHES = (1, 2, 3, 4, 5, 7, 8)


def export(kernel, path, config):
    files = export_kernel(kernel, path)
    cuda = (path / "kernel.cu").read_text()
    host = (path / "host.txt").read_text()
    manifest = {
        "operator": "op25_token_selection",
        "config": config,
        "actual_abi": parse_host(host),
        "actual_cuda_declarations": re.findall(r"__global__\s+void\s+\w+\s*\([^)]*\)", cuda),
        "layout": "contiguous row-major; all buffers disjoint",
        "sm": 87,
        "cooperative_launch": False,
        "stream": "explicit CUDA stream; capture current on every call",
        "toolchain": environment(),
        **files,
    }
    write_json(path / "abi.json", manifest)
    return manifest


def buffers(m, v, k, chunk=4096):
    b = (v + chunk - 1) // chunk
    return dict(
        pv=torch.empty((m, b, k), device="cuda"),
        pi=torch.empty((m, b, k), device="cuda", dtype=torch.int32),
        bad=torch.empty((m, b), device="cuda", dtype=torch.int32),
        values=torch.empty((m, k), device="cuda"),
        ids=torch.empty((m, k), device="cuda", dtype=torch.int32),
        token=torch.empty((m,), device="cuda", dtype=torch.int32),
        status=torch.empty((m,), device="cuda", dtype=torch.int32),
        mass=torch.empty((m, b), device="cuda", dtype=torch.float64),
        meta=torch.empty((m, 2), device="cuda", dtype=torch.float64),
    )


def reference(x, k, uniform=None, temperature=1.0):
    host = x.cpu().float()
    invalid = ~torch.isfinite(host).all(dim=1)
    ids = torch.argsort(host, dim=1, descending=True, stable=True)[:, :k].int()
    values = host.gather(1, ids.long())
    tokens = ids[:, 0].clone()
    status = invalid.int()
    ids[invalid] = -1
    values[invalid] = -float("inf")
    tokens[invalid] = -1
    if uniform is not None:
        u = uniform.cpu()
        bad_u = ~((u >= 0) & (u < 1))
        status += bad_u.int() * 2
        for r in range(x.shape[0]):
            if status[r]:
                tokens[r] = -1
            else:
                weights = torch.exp((host[r].double() - host[r].double().max()) / temperature)
                cdf = torch.cumsum(weights, 0)
                target = u[r] * cdf[-1]
                selected = torch.searchsorted(cdf, target, right=True).item()
                if selected >= x.shape[1]:
                    selected = int(torch.nonzero(weights > 0)[-1])
                tokens[r] = selected
    return values, ids, tokens, status


def check(x, k, kernels, uniform, sampling=None, temperature=1.0):
    allocation_start = time.perf_counter()
    out = buffers(x.shape[0], x.shape[1], k, 4096)
    out["allocation_s"] = time.perf_counter() - allocation_start

    def run():
        launch(
            *kernels,
            x,
            out["pv"],
            out["pi"],
            out["bad"],
            out["values"],
            out["ids"],
            out["token"],
            out["status"],
            stream=torch.cuda.current_stream().cuda_stream,
            sampling=sampling,
            uniform=uniform,
            mass=out["mass"],
            meta=out["meta"],
        )

    start = time.perf_counter()
    run()
    torch.cuda.synchronize()
    first = time.perf_counter() - start
    ref = reference(x, k, uniform if sampling is not None else None, temperature)
    for key, expected in zip(("values", "ids", "token", "status"), ref):
        assert torch.equal(out[key].cpu(), expected), (key, out[key].cpu(), expected)
    return out, run, first


def rng_tests():
    data = []
    for n in (2, 4, 8):
        requests = [17 + i * 101 for i in range(n)]
        for step in (0, 1, 7, 512, 8192, (1 << 64) - 1):
            draws = {r: counter_uniform(r, step) for r in requests}
            assert {r: counter_uniform(r, step) for r in reversed(requests)} == draws
            assert all(counter_uniform(r, step) == draws[r] for r in requests)
            assert all(0 <= u < 1 for u in draws.values())
            data.append({"requests": requests, "absolute_step": step, "uniforms": draws})
    for args in ((-1, 0), (0, -1), (1 << 64, 0), (True, 0), (0, 1.5)):
        try:
            counter_uniform(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counter accepted")
    return {
        "contract": "splitmix64(splitmix64(seed ^ request_id) ^ absolute_step); upper53*2^-53; F64 [0,1)",
        "seed": 20261002,
        "reorder_independent": True,
        "isolated_and_queued_request_same_draw": True,
        "branch_same_identity_same_step_same_draw": True,
        "vectors": data,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--repetitions", type=int, default=6)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dtype", choices=tuple(DTYPES))
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    configure()
    rng_begin = time.perf_counter()
    rng_result = rng_tests()
    rng_test_s = time.perf_counter() - rng_begin
    draw_begin = time.perf_counter()
    for step in range(1000):
        [counter_uniform(17 + 101 * r, step) for r in range(8)]
    draw_s = time.perf_counter() - draw_begin
    results = {
        "environment": environment(),
        "rng": rng_result,
        "rng_tests_s": rng_test_s,
        "offline_python_counter_uniform_s_per_draw": draw_s / 8000,
        "exports": [],
        "cases": [],
        "boundaries": [],
        "status": "in_progress",
        "weights_bytes": 0,
    }
    dtype_names = (a.dtype,) if a.dtype else (("float16",) if a.smoke else tuple(DTYPES))
    batches = (1,) if a.smoke else BATCHES
    builds = {}

    def build(dtype, v, k, temp=1.0):
        key = (dtype, v, k, temp)
        if key in builds:
            return builds[key]
        start = time.perf_counter()
        chunk = 4096
        parts = topk_partials(dtype, v, k, chunk=chunk)
        merge = topk_merge(v, k, chunk=chunk)
        samplers = None
        if k == 3:
            samplers = (
                sampling_mass(dtype, v, k, temp, chunk=chunk),
                sampling_prefix(v, chunk=chunk),
                sampling_select(dtype, v, k, temp, chunk=chunk),
            )
        duration = time.perf_counter() - start
        for label, kernel in zip(
            ("partials", "merge", "mass", "prefix", "select"),
            (parts, merge) + (() if samplers is None else samplers),
        ):
            path = output / "aot" / f"{dtype}_v{v}_k{k}_t{temp}" / label
            results["exports"].append(
                {
                    "prepare_compile_total_s": duration,
                    "abi": export(
                        kernel,
                        path,
                        {
                            "dtype": dtype,
                            "vocab": v,
                            "k": k,
                            "chunk": chunk,
                            "temperature": temp,
                            "stage": label,
                            "rows": "dynamic int32",
                        },
                    ),
                }
            )
        builds[key] = ((parts, merge), samplers)
        write_json(output / "results.json", results)
        return builds[key]

    for dtype in dtype_names:
        for k in (1, 3):
            kernels, sampling = build(dtype, VOCAB, k)
            for m in batches:
                x = (torch.randn((m, VOCAB), device="cuda", dtype=torch.float32) * 4).to(
                    DTYPES[dtype]
                )
                # Full-vocabulary comparisons include deliberately tied maxima
                # across CTA boundaries and the final vocabulary token.
                x[:, 7] = 32
                x[:, 4097] = 32
                x[:, -1] = 32
                u = torch.tensor(
                    [counter_uniform(1001 + r, 512) for r in range(m)],
                    dtype=torch.float64,
                    device="cuda",
                )
                for mode, samplers in [("greedy_topk", None)] + (
                    [("categorical", sampling)] if sampling else []
                ):
                    out, run, first = check(x, k, kernels, u, samplers)
                    timing, graph = benchmark(
                        run, repetitions=a.repetitions, calls_per_replay=4 if samplers else 16
                    )
                    saved_x = x.clone()
                    saved_u = u.clone()
                    initial = out["token"].clone()
                    for field in ("values", "ids", "token", "status"):
                        out[field].fill_(-123)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(out["token"], initial)
                    x.fill_(-100)
                    x[:, -1] = 100
                    u.fill_(0.75)
                    for field in ("values", "ids", "token", "status"):
                        out[field].fill_(-123)
                    graph.replay()
                    torch.cuda.synchronize()
                    changed = reference(x, k, u if samplers else None)
                    for field, expected in zip(("values", "ids", "token", "status"), changed):
                        assert torch.equal(out[field].cpu(), expected)
                    assert not torch.equal(out["token"], initial)
                    x.copy_(saved_x)
                    u.copy_(saved_u)
                    for field in ("values", "ids", "token", "status"):
                        out[field].fill_(-123)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(out["token"], initial)
                    workspace = sum(
                        out[n].numel() * out[n].element_size() for n in ("pv", "pi", "bad")
                    )
                    if samplers:
                        workspace += out["mass"].numel() * 8 + out["meta"].numel() * 8
                    results["cases"].append(
                        {
                            "dtype": dtype,
                            "M": m,
                            "vocab": VOCAB,
                            "k": k,
                            "mode": mode,
                            "exact_ids_values_token_status": True,
                            "first_launch_s": first,
                            "buffer_allocation_s": out["allocation_s"],
                            "timing": timing,
                            "target_m1_ms": 0.080,
                            "target_m1_met": timing["median_ms"] <= 0.080 if m == 1 else None,
                            "workspace_bytes": workspace,
                            "logits_bytes": x.numel() * x.element_size(),
                            "outputs_bytes": m * (k * 8 + 8),
                            "uniform_bytes": m * 8 if samplers else 0,
                            "graph": {
                                "poison_outputs": True,
                                "changed_logits_uniform": True,
                                "restored_exact": True,
                            },
                            "actual_kernel_nodes_per_replay": (5 if samplers else 2)
                            * timing["calls_per_replay"],
                        }
                    )
                    print(dtype, m, k, mode, timing["median_ms"], flush=True)
                    write_json(output / "results.json", results)
                    del out, run, graph
                    gc.collect()
    if not a.smoke:
        for dtype in dtype_names:
            kernels, sampling = build(dtype, VOCAB, 3)
            x = torch.zeros((8, VOCAB), device="cuda", dtype=DTYPES[dtype])
            for row, col, value in (
                (0, 0, float("nan")),
                (1, 4096, float("inf")),
                (2, VOCAB - 1, -float("inf")),
                (3, VOCAB - 1, float("nan")),
                (4, 4095, -float("inf")),
                (5, 0, float("inf")),
            ):
                x[row, col] = value
            x[6, 13] = torch.finfo(DTYPES[dtype]).max
            x[6, -1] = -torch.finfo(DTYPES[dtype]).max
            u = torch.full((8,), 0.5, device="cuda", dtype=torch.float64)
            u[7] = float("nan")
            check(x, 3, kernels, u, None)  # temperature zero ignores invalid uniform
            out, _, _ = check(x, 3, kernels, u, sampling)
            results["boundaries"].append(
                {
                    "dtype": dtype,
                    "vocab": VOCAB,
                    "first_last_chunk_boundary_nonfinite_and_max_finite_exact": True,
                    "temperature_zero_ignores_uniform": True,
                    "sampling_status": out["status"].cpu().tolist(),
                }
            )
        # Vocab tail and exact CDF endpoint/boundary tests use representable
        # uniform and equal weights, avoiding approximate-exp ambiguities.
        for v in (4101, VOCAB):
            kernels, sampling = build("float32", v, 3)
            x = torch.full((8, v), -1000.0, device="cuda")
            x[0].fill_(0)
            x[1].fill_(-10000)
            x[2].fill_(1e30)
            x[3].fill_(-1e30)
            x[4, -1] = 1e30
            x[5].fill_(0)
            x[6].fill_(0)
            x[7].fill_(0)
            u = torch.tensor(
                [0.0, 0.25, 0.5, 0.75, math.nextafter(1.0, 0.0), 0.0, 0.0, 0.0],
                device="cuda",
                dtype=torch.float64,
            )
            for mode, samplers in [("top3", None), ("sampling", sampling)]:
                out, run, first = check(x, 3, kernels, u, samplers)
                results["boundaries"].append(
                    {
                        "vocab": v,
                        "mode": mode,
                        "finite_extremes_exact": True,
                        "first_launch_s": first,
                    }
                )
            # Sparse two-point CDF: u exactly .5 selects second token, as does
            # nextafter(.5,+inf); nextafter(.5,0) selects first. u=0 skips zeros.
            x.fill_(-1000)
            x[:, 3] = 0
            x[:, -1] = 0
            u.copy_(
                torch.tensor(
                    [
                        0,
                        0.5,
                        math.nextafter(0.5, 0),
                        math.nextafter(0.5, 1),
                        math.nextafter(1, 0),
                        0.25,
                        0.75,
                        0.125,
                    ],
                    dtype=torch.float64,
                    device="cuda",
                )
            )
            out, run, _ = check(x, 3, kernels, u, sampling)
            assert out["token"].cpu().tolist() == [3, v - 1, 3, v - 1, v - 1, 3, v - 1, 3]
            results["boundaries"].append({"vocab": v, "exact_sparse_cdf_boundaries": True})
            u.fill_(0.125)
            out, run, _ = check(x, 3, kernels, u, sampling)
            _, graph = benchmark(run, repetitions=1, calls_per_replay=2)
            first_tokens = out["token"].clone()
            u.fill_(0.875)
            out["token"].fill_(-123)
            out["status"].fill_(-123)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.all(out["token"] == v - 1) and not torch.equal(out["token"], first_tokens)
            u.fill_(0.125)
            out["token"].fill_(-123)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out["token"], first_tokens)
            results["boundaries"].append(
                {"vocab": v, "graph_uniform_only_change_and_restore": True, "nodes_per_replay": 10}
            )
            # NaN/+inf/-inf rejected even outside top-k; invalid uniforms reject.
            x.fill_(0)
            x[0, -1] = float("nan")
            x[1, 100] = float("inf")
            x[2, 7] = -float("inf")
            u.fill_(0.3)
            u[3] = -0.1
            u[4] = 1.0
            u[5] = float("nan")
            u[6] = float("inf")
            u[7] = 0
            out, run, _ = check(x, 3, kernels, u, sampling)
            results["boundaries"].append(
                {"vocab": v, "nonfinite_and_invalid_uniform_status": out["status"].cpu().tolist()}
            )
        # Multiple actual requests reordered and executed individually retain
        # the same token and top3, with independently supplied counter uniforms.
        kernels, sampling = build("float32", VOCAB, 3)
        for m in (2, 4, 8):
            x = torch.randn((m, VOCAB), device="cuda")
            u = torch.tensor(
                [counter_uniform(17 + 101 * r, 8192) for r in range(m)],
                device="cuda",
                dtype=torch.float64,
            )
            out, _, _ = check(x, 3, kernels, u, sampling)
            expected = out["token"].clone()
            order = torch.arange(m - 1, -1, -1, device="cuda")
            changed, _, _ = check(
                x[order].contiguous(), 3, kernels, u[order].contiguous(), sampling
            )
            assert torch.equal(changed["token"], expected[order])
            for r in range(m):
                isolated, _, _ = check(
                    x[r : r + 1].contiguous(), 3, kernels, u[r : r + 1].contiguous(), sampling
                )
                assert isolated["token"][0] == expected[r]
            results["boundaries"].append(
                {"request_batch": m, "gpu_reorder_and_isolated_token_exact": True}
            )
        # Configurable small K and non-unit temperature are independently built.
        kernels, _ = build("float32", 4101, 5)
        check(
            torch.randn((3, 4101), device="cuda"),
            5,
            kernels,
            torch.zeros(3, device="cuda", dtype=torch.float64),
        )
        kernels, sampling = build("float32", 4101, 3, 0.5)
        check(
            torch.randn((3, 4101), device="cuda"),
            3,
            kernels,
            torch.tensor([0.125, 0.5, 0.875], device="cuda", dtype=torch.float64),
            sampling,
            0.5,
        )
        results["configurable_k5_temperature_half"] = True
    results["identity"] = [
        identity(ROOT / "kernels/operators/op25_token_selection.py"),
        identity(Path(__file__)),
        identity(ROOT / "tools/operators/abi.py"),
    ]
    results["memory"] = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    results["status"] = (
        "passed standalone full-vocabulary exact selection, CDF sampling, counter, ABI and graph tests"
    )
    write_json(output / "results.json", results)
    print("op25 complete; wrapper cleanup releases GPU", flush=True)


if __name__ == "__main__":
    main()
