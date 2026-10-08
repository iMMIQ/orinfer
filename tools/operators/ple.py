"""Check PLE gating, causal dilation, graph execution and request state."""

import argparse
import gc
from pathlib import Path

import torch

from kernels.model.hyperconnection import hc_norm
from kernels.model.ple import ple_gate, ple_conv, ple_history
from tools.operators.common import (
    configure,
    benchmark,
    error,
    export_kernel,
    environment,
    write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    report = {
        "environment": environment(),
        "cases": [],
        "complete": False,
        "scope": "PLE gate/norm/convolution with synthetic inputs; no full-model quality/TPS.",
    }
    for dtype_name in ("float16", "bfloat16"):
        dtype = getattr(torch, dtype_name)
        for m, h in ((1, 2560), (3, 257), (17, 64), (512, 2560)):
            streams, history = 4, 9
            c = streams * h
            key = torch.randn((m, streams, h), device="cuda", dtype=dtype) * 0.3
            query = torch.randn_like(key) * 0.3
            key[0, 0] = 0  # Sign(0) produces a gate of exactly 0.5.
            value = torch.randn((m, h), device="cuda", dtype=dtype) * 0.2
            weight = torch.randn((streams, h), device="cuda") * 0.1
            conv_weight = (torch.randn((c, 4), device="cuda") * 0.1).half()
            initial = (torch.randn((history, c), device="cuda") * 0.2).to(dtype)
            state = initial.clone()
            gated = torch.empty_like(key)
            normed = torch.empty_like(key)
            out = torch.empty((m, c), device="cuda", dtype=dtype)
            gate = ple_gate(m, h, streams, dtype_name)
            norm = hc_norm(m, h, streams, dtype=dtype_name)
            conv = ple_conv(m, c, dtype=dtype_name)
            advance = ple_history(m, c, history, dtype_name)

            def run():
                state.copy_(initial)
                gate(key, query, value, gated)
                norm(gated, weight, normed)
                conv(normed.flatten(1), gated.flatten(1), state, conv_weight, out)
                advance(normed.flatten(1), state, state)

            def validate():
                raw = ((key * query).sum(-1, keepdim=True) / h**0.5).to(dtype)
                transformed = raw.abs().clamp_min(1e-6).sqrt() * raw.sign()
                expected_gate = (torch.sigmoid(transformed.float()).to(dtype) * value[:, None]).to(
                    dtype
                )
                gf = gated.float()
                expected_norm = (
                    gf * torch.rsqrt(gf.square().mean(-1, keepdim=True) + 1e-6) * (1 + weight)
                ).to(dtype)
                joined = torch.cat((initial, normed.flatten(1)))
                expected_conv = (
                    torch.nn.functional.conv1d(
                        joined.float().T[None],
                        conv_weight.to(dtype).float()[:, None],
                        groups=c,
                        dilation=3,
                    )
                    .squeeze(0)
                    .T.to(dtype)
                )
                expected_out = gated.flatten(1) + torch.nn.functional.silu(
                    expected_conv.float()
                ).to(dtype)
                metrics = {
                    name: error(actual, expected)
                    for name, actual, expected in (
                        ("gate", gated, expected_gate),
                        ("norm", normed, expected_norm),
                        ("conv", out, expected_out),
                    )
                }
                for name, metric in metrics.items():
                    assert metric["finite"] and metric["relative_l2"] < 0.004, (
                        dtype_name,
                        m,
                        name,
                        metric,
                    )
                assert torch.equal(state, joined[-history:])
                return metrics

            run()
            torch.cuda.synchronize()
            metrics = validate()
            timing, graph = benchmark(run, repetitions=8)
            saved_out, saved_state = out.clone(), state.clone()
            out.fill_(float("nan"))
            state.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, saved_out) and torch.equal(state, saved_state)
            original = query.clone()
            query.copy_(torch.randn_like(query))
            graph.replay()
            torch.cuda.synchronize()
            changed = validate()
            query.copy_(original)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, saved_out) and torch.equal(state, saved_state)

            # Split the already-normalized sequence, preserving all nine history
            # rows. Interleave another request to catch accidental shared state.
            chunked_state = initial.clone()
            other_state = -initial.clone()
            pieces = []
            position = 0
            for size in (1, 2, 5, m):
                size = min(size, m - position)
                if size <= 0:
                    continue
                x = normed.flatten(1)[position : position + size].contiguous()
                v = gated.flatten(1)[position : position + size].contiguous()
                y = torch.empty((size, c), device="cuda", dtype=dtype)
                chunk_conv = ple_conv(size, c, dtype=dtype_name)
                chunk_advance = ple_history(size, c, history, dtype_name)
                snapshot = chunked_state.clone()
                chunk_conv(x, v, chunked_state, conv_weight, y)
                chunk_advance(x, chunked_state, chunked_state)
                replay = torch.empty_like(y)
                restored = snapshot.clone()
                # Work on another request before restoring the first one.
                chunk_conv(x, v, other_state, conv_weight, replay)
                chunk_advance(x, other_state, other_state)
                chunk_conv(x, v, restored, conv_weight, replay)
                chunk_advance(x, restored, restored)
                assert torch.equal(y, replay) and torch.equal(restored, chunked_state)
                pieces.append(y)
                position += size
            assert position == m
            assert torch.equal(torch.cat(pieces), saved_out)
            assert torch.equal(chunked_state, saved_state)
            # Disjoint state outputs must agree with the in-place update.
            next_state = torch.empty_like(initial)
            advance(normed.flatten(1), initial, next_state)
            assert torch.equal(next_state, saved_state)
            case = {
                "dtype": dtype_name,
                "rows": m,
                "hidden": h,
                "timing": timing,
                "errors": metrics,
                "changed_input_errors": changed,
                "graph_checks": ["poison", "changed_input", "restore"],
                "state_checks": ["chunking", "snapshot_restore", "request_isolation", "inplace"],
            }
            report["cases"].append(case)
            write_json(args.output / "results.json", report)
            for name, kernel in [
                ("gate", gate),
                ("norm", norm),
                ("conv", conv),
                ("history", advance),
            ]:
                export_kernel(kernel, args.output / f"{dtype_name}-M{m}-H{h}" / name)
            print(dtype_name, m, h, timing["median_ms"], flush=True)
            del graph
            gc.collect()
    report["complete"] = True
    write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
