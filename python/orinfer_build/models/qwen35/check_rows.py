"""Dynamic shared-row kernel checks, including tails and changed-input replay."""

import argparse
import json
from pathlib import Path


def rotation_checks():
    import torch
    from kernels.cuda.model.rotation_a8 import (
        rotate_activation_a8,
        rotate_activation_tiles,
        quantize_rotation_tiles,
    )
    from orinfer_build.models.qwen35.build import primitive
    from orinfer_build.compiler.cuda.common import benchmark, SEED

    torch.manual_seed(SEED)
    results = []

    def stream():
        return torch.cuda.current_stream().cuda_stream

    for k, swiglu in [
        (512, True),
        (640, True),
        (2048, False),
        (2560, False),
        (4096, False),
        (5120, False),
        (17408, True),
    ]:
        reference = primitive(rotate_activation_a8, 4096, k, swiglu=swiglu, dynamic_rows=True)
        register = primitive(
            rotate_activation_a8, 4096, k, swiglu=swiglu, dynamic_rows=True, registers=True
        )
        for live in [1, 17, 129, 512]:
            x = torch.randn((live, k * (2 if swiglu else 1)), device="cuda", dtype=torch.float16)
            signs = torch.randint(0, 2, (k,), device="cuda", dtype=torch.int8) * 2 - 1
            storage = torch.full((live + 1, k), 91, device="cuda", dtype=torch.int8)
            scales = torch.full((live + 1,), 91, device="cuda", dtype=torch.float16)
            q, s = storage[:live], scales[:live]
            eq, es = torch.empty_like(q), torch.empty_like(s)

            def run_register():
                register.adapter.func(x, signs, q, s, live, stream=stream())

            def check_register():
                reference.adapter.func(x, signs, eq, es, live, stream=stream())
                assert torch.equal(q, eq) and torch.equal(s, es), (k, live, "register rotation")
                assert bool(torch.all(storage[live:] == 91)) and bool(
                    torch.all(scales[live:] == 91)
                )

            run_register()
            check_register()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_register()
            x.neg_()
            signs.neg_()
            graph.replay()
            check_register()
            x.zero_()
            graph.replay()
            check_register()
            assert bool(torch.all(q == 0)) and bool(torch.all(s == 1))
            results.append(
                dict(kind="register_rotation", k=k, rows=live, exact=True, changed_input_graph=True)
            )
        for tile in [512, 1024]:
            first = primitive(rotate_activation_tiles, 4096, k, tile, swiglu, dynamic_rows=True)
            second = primitive(quantize_rotation_tiles, 4096, k, tile, dynamic_rows=True)
            for live in [1, 3, 17]:
                x = torch.randn(
                    (live, k * (2 if swiglu else 1)), device="cuda", dtype=torch.float16
                )
                signs = torch.randint(0, 2, (k,), device="cuda", dtype=torch.int8) * 2 - 1
                rotated = torch.empty((live, k), device="cuda", dtype=torch.float16)
                maxima = torch.empty((live, (k + tile - 1) // tile), device="cuda")
                storage = torch.full((live + 1, k), 91, device="cuda", dtype=torch.int8)
                scales = torch.full((live + 1,), 91, device="cuda", dtype=torch.float16)
                q, s = storage[:live], scales[:live]
                eq, es = torch.empty_like(q), torch.empty_like(s)

                def run():
                    first.adapter.func(x, signs, rotated, maxima, live, stream=stream())
                    second.adapter.func(rotated, maxima, q, s, live, stream=stream())

                def check():
                    reference.adapter.func(x, signs, eq, es, live, stream=stream())
                    assert torch.equal(q, eq), (
                        k,
                        tile,
                        live,
                        int((q.int() - eq.int()).abs().max()),
                    )
                    assert torch.equal(s, es), (k, tile, live, "scale")
                    assert bool(torch.all(storage[live:] == 91)) and bool(
                        torch.all(scales[live:] == 91)
                    )

                run()
                check()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                x.neg_()
                q.fill_(91)
                graph.replay()
                check()
                x.zero_()
                graph.replay()
                check()
                x.normal_()
                run()
                check()
                timing, _ = benchmark(run, repetitions=8)
                results.append(
                    dict(
                        kind="rotation",
                        k=k,
                        swiglu=swiglu,
                        tile=tile,
                        rows=live,
                        exact=True,
                        changed_input_graph=True,
                        ms=timing["median_ms"],
                    )
                )
                print("ROTATION", results[-1], flush=True)
    return results


def residual_rotation_checks():
    import torch
    from kernels.cuda.model.rotation_a8 import residual_norm_rotate_a8, rotate_activation_a8
    from kernels.cuda.operators.op02_residual_norm import residual_norm
    from orinfer_build.compiler.compile import primitive
    from orinfer_build.compiler.cuda.common import SEED

    torch.manual_seed(SEED)
    results = []
    reference = primitive(residual_norm, None)
    rotation = primitive(rotate_activation_a8, 4096, 5120, dynamic_rows=True)
    for dynamic in (False, True):
        for rows in [1, 129] if not dynamic else [1, 2, 3, 4, 8, 17, 129, 512, 4096]:
            kernel = primitive(
                residual_norm_rotate_a8, 4096 if dynamic else rows, dynamic_rows=dynamic
            )
            x = torch.randn((rows, 5120), device="cuda", dtype=torch.float16)
            r = torch.randn((rows, 5120), device="cuda")
            w = torch.randn((5120,), device="cuda", dtype=torch.float16) * 0.1
            signs = torch.randint(0, 2, (5120,), device="cuda", dtype=torch.int8) * 2 - 1
            storage = [
                torch.full((rows + 1, 5120), 91, device="cuda", dtype=dtype)
                for dtype in (torch.float16, torch.float32, torch.int8)
            ] + [torch.full((rows + 1,), 91, device="cuda", dtype=torch.float16)]
            y, ro, q, scale = [v[:rows] for v in storage]
            ey, er, eq, es = [torch.empty_like(v[:rows]) for v in storage]

            def launch(fn, *args):
                fn.adapter.func(*args, stream=torch.cuda.current_stream().cuda_stream)

            def run():
                launch(kernel, x, r, w, y, ro, signs, q, scale, *([rows] if dynamic else []))

            def check():
                launch(reference, x, r, w, ey, er)
                launch(rotation, ey, signs, eq, es, rows)
                for out, expected in zip((y, ro, q, scale), (ey, er, eq, es)):
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                assert all(bool(torch.all(v[rows:] == 91)) for v in storage)

            run()
            check()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            x.normal_()
            r.normal_()
            signs.neg_()
            for out in storage:
                out.fill_(91)
            graph.replay()
            check()
            x.zero_()
            r.zero_()
            graph.replay()
            check()
            assert bool(torch.all(q == 0)) and bool(torch.all(scale == 1))
            results.append(dict(rows=rows, dynamic=dynamic, exact=True, changed_input_graph=True))
            print("RESIDUAL ROTATION", results[-1], flush=True)
    return results


def kv_loader_checks():
    """Thread-stripe mapping versus independent packed-byte/FP16 arithmetic."""
    import torch
    import tilelang.language as T
    from kernels.cuda.model.kv_int8 import HALF2_SOURCE, int8_kv_tile
    from orinfer_build.compiler.compile import primitive
    from orinfer_build.compiler.cuda.common import orin_jit

    results = []
    for bn, dimension in [(32, 64), (32, 256), (64, 256)]:
        for threads in [128, 256]:
            load = int8_kv_tile(bn, dimension, threads)

            @orin_jit
            def probe():
                @T.prim_func
                def main(
                    K: T.Tensor((1, bn, dimension // 4), T.uint32),
                    V: T.Tensor((1, bn, dimension // 4), T.uint32),
                    KS: T.Tensor((1, bn, dimension // 64), T.float16),
                    VS: T.Tensor((1, bn, dimension // 64), T.float16),
                    Y: T.Tensor((bn, dimension), T.float16),
                    Z: T.Tensor((bn, dimension), T.float16),
                    valid: T.int32,
                ):
                    with T.Kernel(1, threads=threads):
                        T.import_source(HALF2_SOURCE)
                        k = T.alloc_shared((bn, dimension), T.float16)
                        v = T.alloc_shared((bn, dimension), T.float16)
                        load(K, V, KS, VS, k, v, 0, 0, 0, valid)
                        T.sync_threads()
                        for j, d in T.Parallel(bn, dimension):
                            Y[j, d] = k[j, d]
                            Z[j, d] = v[j, d]

                return main

            kernel = primitive(probe)
            codes = [
                torch.arange(bn * dimension, device="cuda", dtype=torch.int32)
                .remainder(256)
                .sub(128)
                .to(torch.int8)
                .reshape(1, bn, dimension)
                for _ in range(2)
            ]
            codes[1] = codes[1].flip(-1).contiguous()
            scales = [
                torch.rand((1, bn, dimension // 64), device="cuda", dtype=torch.float16) * 0.01
                + 0.0001
                for _ in range(2)
            ]
            scales[0][:, ::3] = 32768
            storage = [
                torch.full((bn + 1, dimension), 91, device="cuda", dtype=torch.float16)
                for _ in range(2)
            ]
            for valid in [0, bn - 1, bn]:
                kernel.adapter.func(
                    *(x.view(torch.uint32) for x in codes),
                    *scales,
                    *storage,
                    valid,
                    stream=torch.cuda.current_stream().cuda_stream,
                )
                torch.cuda.synchronize()
                for c, scale, out in zip(codes, scales, storage):
                    expected = (
                        (
                            c.float().reshape(bn, dimension // 64, 64)
                            * scale.float().reshape(bn, dimension // 64, 1)
                        )
                        .to(torch.float16)
                        .clamp(-65504, 65504)
                        .reshape(bn, dimension)
                    )
                    expected[valid:] = 0
                    torch.testing.assert_close(out[:bn], expected, rtol=0, atol=0)
                    assert bool(torch.all(out[bn:] == 91))
                results.append(
                    dict(block_n=bn, dimension=dimension, threads=threads, valid=valid, exact=True)
                )
    return results


def attention_checks(context=1024, kv_length=None):
    import torch
    from kernels.cuda.model.attention_rows import causal_attention_int8, merge_attention_int8
    from orinfer_build.validation.attention_reference import attention_prefill_int8
    from orinfer_build.models.qwen35.build import primitive
    from orinfer_build.compiler.cuda.common import SEED

    torch.manual_seed(SEED)

    def stream():
        return torch.cuda.current_stream().cuda_stream

    result = []
    configurations = [
        (m, n, scale, group, 8, 128, 0)
        for m, n in [(16, 32), (16, 64), (32, 64), (64, 32)]
        for scale in [1.0, 8.0]
        for group in [1, 6]
    ] + [(32, 32, scale, 1, 1, 128, 0) for scale in [1.0, 8.0]]
    configurations += [(64, 64, scale, 1, 8, 256, 1) for scale in [1.0, 8.0]]
    for bm, bn, q_scale, head_group, splits, threads, stages in configurations:
        target = 16
        valid = (context if kv_length is None else kv_length) - 5
        if not 256 <= valid + 5 <= context:
            raise ValueError("Expected 256 <= live KV length <= context capacity")
        first = primitive(
            causal_attention_int8,
            4096,
            context,
            bm,
            bn,
            target,
            splits,
            head_group=head_group,
            threads=threads,
            stages=stages,
        )
        second = primitive(merge_attention_int8, 4096, bm, target, splits, head_group=head_group)
        reference = primitive(attention_prefill_int8, 1, 4096, context, dynamic_rows=True)
        part = torch.empty((24 // head_group, 2 * target, bm, 256), device="cuda")
        maximum = torch.empty((24 // head_group, 2 * target, bm), device="cuda")
        denominator = torch.empty_like(maximum)
        for live in [1, 3, 8, 17, 129]:
            q = torch.randn((1, live, 24, 256), device="cuda", dtype=torch.float16) * q_scale
            gate = torch.randn_like(q)
            kv = [
                # Reserve no readable tail beyond live KV: run under memcheck
                # with the Torch allocator cache disabled to check prefetches.
                torch.randint(-127, 128, (1, valid, 4, 256), device="cuda", dtype=torch.int8)
                for _ in range(2)
            ]
            scales = [
                torch.rand((1, valid, 4, 4), device="cuda", dtype=torch.float16) * 0.003 + 0.00001
                for _ in range(2)
            ]
            pos = (torch.arange(live, device="cuda", dtype=torch.int32) + valid - live)[None]
            length = torch.tensor([valid], device="cuda", dtype=torch.int32)
            storage = torch.full((1, live + 1, 24, 256), 91, device="cuda", dtype=torch.float16)
            out, expected = storage[:, :live], torch.empty_like(q)

            def run():
                first.adapter.func(
                    q,
                    *kv,
                    *scales,
                    gate,
                    pos,
                    length,
                    out,
                    part,
                    maximum,
                    denominator,
                    live,
                    stream=stream(),
                )
                second.adapter.func(part, maximum, denominator, gate, out, live, stream=stream())

            def check():
                reference.adapter.func(
                    q, *kv, gate, pos, length, expected, *scales, live, stream=stream()
                )
                torch.testing.assert_close(out, expected, atol=4e-5, rtol=0.005)
                assert bool(torch.all(storage[:, live:] == 91))

            run()
            check()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            q.neg_()
            out.fill_(91)
            graph.replay()
            check()
            max_abs = float((out - expected).abs().max())
            pos.fill_(-1)
            graph.replay()
            check()
            assert bool(torch.all(out == 0))
            result.append(
                dict(
                    kind="causal_attention",
                    bm=bm,
                    bn=bn,
                    rows=live,
                    changed_input_graph=True,
                    q_scale=q_scale,
                    head_group=head_group,
                    max_splits=splits,
                    threads=threads,
                    stages=stages,
                    context_capacity=context,
                    valid_context=valid,
                    max_abs=max_abs,
                    empty_context_zero=True,
                )
            )
            print("ATTENTION", result[-1], flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--context-capacity", type=int, default=1024)
    p.add_argument("--kv-length", type=int)
    p.add_argument("--attention-only", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    report = dict(
        rotation=[] if a.attention_only else rotation_checks(),
        residual_rotation=[] if a.attention_only else residual_rotation_checks(),
        kv_loader=kv_loader_checks(),
        attention=attention_checks(a.context_capacity, a.kv_length),
    )
    (a.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
