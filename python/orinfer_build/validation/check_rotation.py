"""Verify row-generic expert rotation, dispatch and exact A8 plane fusion."""

import argparse
from pathlib import Path

import tilelang
import torch

from kernels.cuda.model.e8p_exact import split_a8
from kernels.cuda.model.moe import expert_dispatch
from kernels.cuda.model.rotation_a8 import rotate_activation_a8, rotate_expert_a8
from kernels.shared.rows import explicit_rows
from orinfer_build.tuning.projection_tuning import profile_stream
from orinfer_build.compiler.cuda.common import configure, write_json


def compile_dynamic(fn):
    return profile_stream(
        tilelang.compile(
            explicit_rows(fn),
            out_idx=[],
            execution_backend="nvrtc",
            target={"kind": "cuda", "arch": "sm_87"},
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    torch.manual_seed(20261002)
    report = dict(complete=False, cases=[])
    geometries = [
        (None, 131, 384, False, 1, 1),
        (None, 131, 128, True, 1, 1),
        (None, 1280, 640, True, 1, 1),
        ("scatter", 128, 2560, False, 10, 512),
        ("shared", 1, 2560, False, 10, 512),
        ("shared", 17, 384, False, 3, 11),
        ("scatter", 17, 128, False, 2, 7),
        ("scatter", 17, 384, True, 3, 11),
    ]
    for mode, capacity, width, swiglu, top_k, experts in geometries:
        slots = capacity * top_k if mode else capacity
        candidate = compile_dynamic(
            rotate_expert_a8.__wrapped__(
                capacity,
                width,
                swiglu=swiglu,
                dynamic_rows=True,
                routing=mode,
                top_k=top_k,
                experts=experts,
            )
        )
        rotation = compile_dynamic(
            rotate_activation_a8.__wrapped__(capacity, width, swiglu=swiglu, dynamic_rows=True)
        )
        pack = compile_dynamic(split_a8.__wrapped__(slots, width, dynamic_rows=True))
        dispatch = (
            compile_dynamic(
                expert_dispatch.__wrapped__(
                    capacity, width, E=experts, top_k=top_k, scale_group=width, dynamic_rows=True
                )
            )
            if mode
            else None
        )
        x = torch.randn(capacity, width * (2 if swiglu else 1), device="cuda", dtype=torch.float16)
        x[0].zero_()
        signs = torch.randint(0, 2, (width,), device="cuda", dtype=torch.int8) * 2 - 1
        q = torch.empty(capacity, width, device="cuda", dtype=torch.int8)
        scale = torch.empty(capacity, 1, device="cuda", dtype=torch.float16)
        actual_q, actual_scale = torch.empty_like(q), torch.empty_like(scale)
        expected = [
            torch.empty(slots, width // 8, device="cuda", dtype=torch.uint32),
            torch.empty(slots, width // 8, device="cuda", dtype=torch.uint32),
            torch.empty(slots, device="cuda", dtype=torch.int32),
        ]
        actual = [torch.empty_like(t) for t in expected]
        routed = torch.empty(slots, width, device="cuda", dtype=torch.int8)
        for rows in sorted({1, min(3, capacity), max(1, capacity - 1), capacity}):
            plane_rows = rows * top_k if mode == "scatter" else rows
            routing = []
            if mode:
                ids = (
                    torch.rand(rows, experts, device="cuda")
                    .topk(top_k, dim=1)
                    .indices.to(torch.int32)
                )
                flat = ids.flatten().long()
                counts = torch.bincount(flat, minlength=experts)
                offsets = (counts.cumsum(0) - counts).to(torch.int32)
                one_hot = torch.nn.functional.one_hot(flat, experts).to(torch.int32)
                relative = (
                    one_hot.cumsum(0, dtype=torch.int32).gather(1, flat[:, None]).flatten() - 1
                ).reshape(rows, top_k)
                expected_ds = torch.empty(slots, 1, device="cuda", dtype=torch.float16)
                actual_ds = torch.empty_like(expected_ds)
                expected_map, actual_map = torch.empty_like(ids), torch.empty_like(ids)
                routing = [ids, relative, offsets, actual_ds, actual_map]

            def reference():
                rotation.adapter.func(x, signs, q, scale, rows)
                if mode:
                    dispatch.adapter.func(
                        q, scale, ids, relative, offsets, routed, expected_ds, expected_map, rows
                    )
                pack.adapter.func(routed if mode == "scatter" else q, *expected, plane_rows)

            def fused():
                candidate.adapter.func(x, signs, actual_q, actual_scale, *actual, *routing, rows)

            def reset():
                for value in [actual_q, actual_scale, *actual] + ([actual_ds] if mode else []):
                    value.fill_(42)

            def check():
                torch.cuda.synchronize()
                assert torch.equal(q[:rows], actual_q[:rows])
                assert torch.equal(scale[:rows], actual_scale[:rows])
                assert bool((actual_q[rows:] == 42).all())
                assert bool((actual_scale[rows:] == 42).all())
                for a, b in zip(actual, expected, strict=True):
                    assert torch.equal(a[:plane_rows], b[:plane_rows]), (mode, rows)
                    assert bool((a[plane_rows:] == 42).all())
                if mode:
                    live = rows * top_k
                    assert torch.equal(actual_ds[:live], expected_ds[:live])
                    assert torch.equal(actual_map, expected_map)
                    assert bool((actual_ds[live:] == 42).all())

            reset()
            reference()
            fused()
            check()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused()
            x.mul_(-0.75)
            reset()
            reference()
            graph.replay()
            check()
            report["cases"].append(
                dict(
                    mode=mode,
                    width=width,
                    swiglu=swiglu,
                    top_k=top_k,
                    experts=experts,
                    rows=rows,
                    exact=True,
                    tail_intact=True,
                    replay=True,
                )
            )
            print("PASS EXPERT FUSION", mode, width, rows, flush=True)
    report["complete"] = True
    write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
