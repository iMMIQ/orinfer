"""Integer VQ numerical/tail/graph checks; no model-quality acceptance."""

import argparse
from pathlib import Path
import subprocess

import numpy as np
import torch

from tools.quantization.vq import Weights, rotate, e8p_sign_table
from tools.quantization.reference_math import swiglu
from tools.operators.quantized_reference import upload, reference
from tools.operators.common import (
    configure,
    benchmark,
    error,
    environment,
    export_kernel,
    write_json,
)
from kernels.model.integer_vq import integer_vq, rotate_activation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    configure()
    rng = np.random.default_rng(20261002)
    report = {
        "scope": __doc__,
        "environment": environment(),
        "cases": [],
        "rotations": [],
        "complete": False,
    }
    for kind, d, dtype in [("vq4", 4, np.uint8), ("e8p", 8, np.uint16)]:
        for m, patched, shared_table in (
            (1, False, False),
            (17, False, False),
            (17, True, False),
            (17, True, True),
        ):
            e, n, k = 2, 65, 256
            bank = []
            for _ in range(e):
                codes = rng.integers(0, 256 if d == 4 else 65536, (n, k // d), dtype=dtype)
                table = (
                    rng.integers(-128, 128, (256, d), dtype=np.int16).astype(np.int8)
                    if d == 4
                    else (rng.integers(-63, 64, (256, d)) * 2).astype(np.int8)
                )
                if shared_table and bank:
                    table = bank[0].table
                patches = (
                    rng.integers(0, 65536, (n, k // 128), dtype=np.uint16) if patched else None
                )
                if patches is not None:
                    patches[0, 0] = 32768 | (128 << 7) | 127  # signed -128, last group position
                bank.append(
                    Weights(
                        kind,
                        codes,
                        table,
                        np.full(n, 0.015625, np.float16),
                        np.empty(0, np.int8),
                        patches,
                    )
                )
            pp, book, ws = upload(bank)
            if shared_table:
                book = book[:1].contiguous()
            patch = (
                torch.from_numpy(np.stack([w.patches.T for w in bank]).copy()).cuda()
                if patched
                else torch.zeros(1, device="cuda", dtype=torch.uint16)
            )
            integer = torch.from_numpy(np.stack([w.integer_weights() for w in bank])).cuda()
            x = torch.randint(-128, 128, (e, m, k), device="cuda", dtype=torch.int8)
            sa = torch.full((e, m), 0.0078125, device="cuda", dtype=torch.float16)
            out = torch.full((e * m + 1, n), 91.0, device="cuda", dtype=torch.float16)
            kernel = integer_vq(e, m, n, k, kind=kind, patched=patched, shared_table=shared_table)

            def run():
                kernel(x.view(e * m, k), pp, book, patch, ws, sa.view(-1), out[: e * m])

            timing, graph = benchmark(run, repetitions=4)
            expected = reference(x, integer, ws, sa).view(e * m, n)
            assert torch.equal(out[: e * m], expected), error(out[: e * m], expected)
            assert bool((out[-1] == 91).all())
            original_x = x.clone()
            x.zero_()
            out[: e * m].fill_(91)
            graph.replay()
            torch.cuda.synchronize()
            assert bool((out[: e * m] == 0).all())
            x.copy_(original_x)
            saved_book = book.clone()
            book.zero_()
            mutated = [
                Weights(w.kind, w.indices, np.zeros_like(w.table), w.scales, w.signs, w.patches)
                for w in bank
            ]
            changed = torch.from_numpy(np.stack([w.integer_weights() for w in mutated])).cuda()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[: e * m], reference(x, changed, ws, sa).view(e * m, n))
            book.copy_(saved_book)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[: e * m], expected)
            if patched:
                saved_patch = patch.clone()
                patch.zero_()
                graph.replay()
                torch.cuda.synchronize()
                unpatched = [Weights(w.kind, w.indices, w.table, w.scales, w.signs) for w in bank]
                restored_integer = torch.from_numpy(
                    np.stack([w.integer_weights() for w in unpatched])
                ).cuda()
                assert torch.equal(
                    out[: e * m], reference(x, restored_integer, ws, sa).view(e * m, n)
                )
                patch.copy_(saved_patch)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(out[: e * m], expected)
            folder = a.output / f"{kind}-M{m}-patch{int(patched)}-shared{int(shared_table)}"
            exported = export_kernel(kernel, folder)
            sass = subprocess.check_output(
                ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(folder / "kernel.cubin")],
                text=True,
            )
            assert "IMMA.16832.S8.S8" in sass
            (folder / "kernel.sass").write_text(sass)
            report["cases"].append(
                {
                    "kind": kind,
                    "rows": m,
                    "patched": patched,
                    "shared_table": shared_table,
                    "integer_oracle_equal": True,
                    "tail_guard": True,
                    "changed_activation_and_table_graph": True,
                    "native_int8_mma": True,
                    "timing": timing,
                    "export": exported,
                }
            )
            if kind == "e8p":
                tables = bank[:1] if shared_table else bank
                short = torch.from_numpy(np.stack([e8p_sign_table(w.table) for w in tables])).cuda()
                fast = integer_vq(
                    e,
                    m,
                    n,
                    k,
                    kind=kind,
                    patched=patched,
                    shared_table=shared_table,
                    shortbook=True,
                    block_n=128,
                    num_stages=1,
                )

                def run_short():
                    fast(x.view(e * m, k), pp, short, patch, ws, sa.view(-1), out[: e * m])

                short_timing, short_graph = benchmark(run_short, repetitions=4)
                assert torch.equal(out[: e * m], expected), error(out[: e * m], expected)
                assert bool((out[-1] == 91).all())
                # Keep addresses stable while changing both input and decoded
                # basis. Zero basis still decodes the E8P parity shifts.
                zero = [
                    Weights(w.kind, w.indices, np.zeros_like(w.table), w.scales, w.signs, w.patches)
                    for w in bank
                ]
                short.copy_(
                    torch.from_numpy(
                        np.stack(
                            [e8p_sign_table(w.table) for w in (zero[:1] if shared_table else zero)]
                        )
                    ).cuda()
                )
                x.neg_()
                short_graph.replay()
                torch.cuda.synchronize()
                changed = torch.from_numpy(np.stack([w.integer_weights() for w in zero])).cuda()
                assert torch.equal(out[: e * m], reference(x, changed, ws, sa).view(e * m, n))
                x.copy_(original_x)
                report["cases"].append(
                    {
                        "kind": "e8p-short",
                        "rows": m,
                        "patched": patched,
                        "shared_table": shared_table,
                        "integer_oracle_equal": True,
                        "tail_guard": True,
                        "changed_activation_and_table_graph": True,
                        "timing": short_timing,
                        "export": export_kernel(fast, folder / "short"),
                    }
                )
            write_json(a.output / "results.json", report)
    for fused in (False, True):
        m, k = 3, 256
        x = (torch.randn((m, k * (2 if fused else 1)), device="cuda") * 0.2).half()
        signs = rng.choice(np.array([-1, 1], np.int8), k)
        gs = torch.from_numpy(signs).cuda()
        out = torch.empty((m, k), device="cuda", dtype=torch.float16)
        kernel = rotate_activation(m, k, swiglu=fused)

        def run():
            kernel(x, gs, out)

        timing, graph = benchmark(run, repetitions=4)
        raw = x.cpu().numpy()
        if fused:
            raw = swiglu(raw).astype(np.float16)
        expected = torch.from_numpy(rotate(raw, signs).astype(np.float16)).cuda()
        metric = error(out, expected)
        assert metric["relative_l2"] < 0.001 and metric["max_abs"] < 0.001, metric
        original = x.clone()
        x.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert bool((out == 0).all())
        x.copy_(original)
        gs.mul_(-1)
        graph.replay()
        torch.cuda.synchronize()
        changed = torch.from_numpy(rotate(raw, -signs).astype(np.float16)).cuda()
        assert error(out, changed)["relative_l2"] < 0.001
        gs.mul_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert error(out, expected)["relative_l2"] < 0.001
        report["rotations"].append(
            {
                "fused_swiglu": fused,
                "error": metric,
                "changed_signs_graph": True,
                "timing": timing,
                "export": export_kernel(kernel, a.output / f"rotation-fused{int(fused)}"),
            }
        )
        write_json(a.output / "results.json", report)
    report["complete"] = True
    write_json(a.output / "results.json", report)


if __name__ == "__main__":
    main()
