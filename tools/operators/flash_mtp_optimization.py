"""Numerical, tail and changed-graph checks for Flash MTP optimization kernels."""

import argparse
from pathlib import Path
import numpy as np
import torch
from kernels.model.hyperconnection import hc_projection, hc_silu, hc_mix, hc_up_mix
from kernels.model.gdn_sequence import gdn_sequence, gdn_commit
from kernels.model.integer_vq import integer_e8p_gemv, rotate_activation
from kernels.model.rotation_a8 import rotate_activation_a8
from kernels.operators.op30_activation_quantization import activation_quantization, launch
from tools.quantization.vq import Weights
from tools.operators.quantized_reference import upload
from tools.operators.common import configure, benchmark, error, write_json, export_kernel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    configure()
    report = {"complete": False, "hc": [], "gdn": [], "experts": [], "rotation": []}
    for m in (1, 4, 8):
        h, rank = 2560, 320
        x = (torch.randn(m, 4, h, device="cuda") * 0.3).half()
        wd = (torch.randn(rank, 4 * h, device="cuda") * 0.01).bfloat16()
        wu = (torch.randn(4 * h, rank, device="cuda") * 0.03).bfloat16()
        down, act = (torch.empty(m, rank, device="cuda", dtype=torch.float16) for _ in range(2))
        up = torch.empty_like(x)
        expected = torch.empty(m, h, device="cuda", dtype=torch.float16)
        actual = torch.empty_like(expected)
        fused_act = torch.empty_like(act)
        kd = hc_projection(m, rank, 4 * h, dtype="float16", block_n=32)
        ks = hc_silu(m, rank, dtype="float16")
        ku = hc_projection(m, 4 * h, rank, dtype="float16")
        km = hc_mix(m, h, dtype="float16")
        fd = hc_projection(m, rank, 4 * h, dtype="float16", block_n=32, silu=True)
        fu = hc_up_mix(m, h, rank)

        def reference():
            kd(x.flatten(1), wd, down)
            ks(down, act)
            ku(act, wu, up.flatten(1))
            km(x, up, expected)

        def candidate():
            fd(x.flatten(1), wd, fused_act)
            fu(fused_act, wu, x, actual)

        rt, _ = benchmark(reference, repetitions=5)
        ct, graph = benchmark(candidate, repetitions=5)
        assert torch.equal(act, fused_act), error(fused_act, act)
        assert torch.equal(actual, expected), error(actual, expected)
        x.neg_()
        reference()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(act, fused_act) and torch.equal(actual, expected)
        export_kernel(fd, a.output / f"hc-down-{m}")
        export_kernel(fu, a.output / f"hc-up-{m}")
        report["hc"].append(
            {"rows": m, "exact": True, "changed_graph": True, "reference": rt, "fused": ct}
        )
        write_json(a.output / "results.json", report)
        print("hc", report["hc"][-1], flush=True)
    for m in (2, 4, 8):
        q, k = (
            torch.nn.functional.normalize(torch.randn(16, m, 128, device="cuda"), dim=-1).half()
            for _ in range(2)
        )
        v = torch.randn(48, m, 128, device="cuda").half()
        g = -torch.rand(m, 48, device="cuda") * 0.1
        beta = torch.sigmoid(torch.randn(m, 48, device="cuda"))
        initial = torch.randn(48, 128, 128, device="cuda") * 0.01
        state = initial.clone()
        prefix = torch.empty(m, 48, 128, 128, device="cuda")
        delta = torch.empty(m, 48, 128, device="cuda")
        expected, actual = (
            torch.empty(m, 6144, device="cuda", dtype=torch.float16) for _ in range(2)
        )
        ref = gdn_sequence(m)
        candidate = gdn_sequence(m, compact=True)
        commit = gdn_commit(m)
        ref(q, k, v, g, beta, state, prefix, expected)
        candidate(q, k, v, g, beta, state, delta, actual)
        assert torch.equal(actual, expected), error(actual, expected)
        accepted = torch.ones(1, device="cuda", dtype=torch.int32)
        # Captured accepted count is an input, not a Python specialization.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            commit(k, g, delta, accepted, state)
        for count in range(1, m + 1):
            state.copy_(initial)
            accepted.fill_(count)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(state, prefix[count - 1]), error(state, prefix[count - 1])
        state.copy_(initial)
        q.neg_()
        v.neg_()
        ref(q, k, v, g, beta, state, prefix, expected)
        candidate(q, k, v, g, beta, state, delta, actual)
        assert torch.equal(actual, expected)
        for count in range(1, m + 1):
            state.copy_(initial)
            accepted.fill_(count)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(state, prefix[count - 1]), error(state, prefix[count - 1])
        export_kernel(candidate, a.output / f"gdn-compact-{m}")
        export_kernel(commit, a.output / f"gdn-commit-{m}")
        report["gdn"].append(
            {
                "rows": m,
                "all_prefixes_exact": True,
                "changed_graph": True,
                "prefix_bytes": prefix.numel() * 4,
                "update_bytes": delta.numel() * 4,
            }
        )
        write_json(a.output / "results.json", report)
        print("gdn", report["gdn"][-1], flush=True)
    rng = np.random.default_rng(20261002)
    for n, k in ((65, 256), (1280, 2560), (2560, 640)):
        e, routes = 12, 3
        table = (rng.integers(-63, 64, (256, 8)) * 2).astype(np.int8)
        bank = [
            Weights(
                "e8p",
                rng.integers(0, 65536, (n, k // 8), dtype=np.uint16),
                table,
                np.full(n, 0.015625, np.float16),
                np.empty(0, np.int8),
            )
            for _ in range(e)
        ]
        pp, book, ws = upload(bank)
        book = book[:1].contiguous()
        weights = torch.from_numpy(np.stack([w.integer_weights() for w in bank])).cuda()
        ids = torch.tensor([[11, 2, 7]], device="cuda", dtype=torch.int32)
        for shared in (True, False):
            rows = 1 if shared else routes
            x = torch.randint(-128, 128, (rows, k), device="cuda", dtype=torch.int8)
            scale = torch.full((rows,), 0.0078125, device="cuda", dtype=torch.float16)
            out = torch.full((routes + 1, n), 91.0, device="cuda", dtype=torch.float16)
            kernel = integer_e8p_gemv(e, routes, n, k, shared_input=shared)

            def run():
                kernel(x.view(torch.int32), pp, book, ws, scale, ids, out[:routes])

            def oracle():
                values = []
                for index in range(routes):
                    expert = int(ids[0, index])
                    row = 0 if shared else index
                    acc = (weights[expert].int() * x[row].int()).sum(-1).float()
                    values.append(((acc * ws[expert].float()) * scale[row].float()).half())
                return torch.stack(values)

            timing, graph = benchmark(run, repetitions=5)
            assert torch.equal(out[:routes], oracle()), error(out[:routes], oracle())
            assert (out[-1] == 91).all()
            x.neg_()
            ids.copy_(torch.tensor([[0, 9, 3]], device="cuda", dtype=torch.int32))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[:routes], oracle())
            report["experts"].append(
                {
                    "shape": [n, k],
                    "shared_input": shared,
                    "exact": True,
                    "tail_guard": True,
                    "changed_ids_graph": True,
                    "timing": timing,
                }
            )
            write_json(a.output / "results.json", report)
            print("expert", report["experts"][-1], flush=True)
    for m, k, swiglu in (
        (1, 2560, False),
        (8, 2560, False),
        (128, 2560, False),
        (512, 2560, False),
        (10, 640, True),
        (80, 640, True),
        (1280, 640, True),
        (5120, 640, True),
    ):
        x = (torch.randn(m, k * (2 if swiglu else 1), device="cuda") * 0.2).half()
        signs = (torch.randint(0, 2, (k,), device="cuda") * 2 - 1).to(torch.int8)
        rotated = torch.empty(m, k, device="cuda", dtype=torch.float16)
        mask = torch.zeros(k, device="cuda", dtype=torch.uint8)
        expected = torch.empty(m, k, device="cuda", dtype=torch.int8)
        expected_scale = torch.empty(m, 1, device="cuda", dtype=torch.float16)
        actual = torch.full((m + 1, k), 91, device="cuda", dtype=torch.int8)
        actual_scale = torch.full((m + 1, 1), 91.0, device="cuda", dtype=torch.float16)
        rotation = rotate_activation(m, k, swiglu=swiglu)
        quantization = activation_quantization(k)
        fused = rotate_activation_a8(m, k, swiglu=swiglu)

        def reference():
            rotation(x, signs, rotated)
            launch(
                quantization,
                rotated,
                mask,
                expected,
                expected_scale,
                stream=torch.cuda.current_stream().cuda_stream,
            )

        def candidate():
            fused(x, signs, actual[:m], actual_scale[:m])

        reference_timing, _ = benchmark(reference, repetitions=5)
        fused_timing, graph = benchmark(candidate, repetitions=5)

        def check():
            assert torch.equal(actual[:m], expected), error(actual[:m], expected)
            assert torch.equal(actual_scale[:m], expected_scale), error(
                actual_scale[:m], expected_scale
            )
            assert (actual[m] == 91).all() and actual_scale[m, 0] == 91

        check()
        for pattern in ("negated", "zero", "subnormal", "discrete", "large"):
            if pattern == "negated":
                x.neg_()
                signs.neg_()
            elif pattern == "zero":
                x.zero_()
            elif pattern == "subnormal":
                x.copy_(torch.randint(-4, 5, x.shape, device="cuda").float() * 2**-24)
            elif pattern == "discrete":
                x.copy_(torch.randint(-256, 257, x.shape, device="cuda").float() / 128)
            else:
                x.normal_(0, 3)
            reference()
            graph.replay()
            torch.cuda.synchronize()
            check()
        report["rotation"].append(
            {
                "rows": m,
                "width": k,
                "swiglu": swiglu,
                "exact": True,
                "changed_graph": True,
                "tail_guard": True,
                "patterns": ["gaussian", "negated", "zero", "subnormal", "discrete", "large"],
                "reference": reference_timing,
                "fused": fused_timing,
            }
        )
        write_json(a.output / "results.json", report)
        print("rotation", report["rotation"][-1], flush=True)
    report["complete"] = True
    write_json(a.output / "results.json", report)


if __name__ == "__main__":
    main()
