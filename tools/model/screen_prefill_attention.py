"""Validate and time SM87 dense attention variants at actual history depths."""

import argparse
from pathlib import Path
import torch
from kernels.model.attention_prefill_staged import attention_prefill_staged
from tools.operators.common import configure, benchmark, error, export_kernel, write_json


def run(output):
    configure()
    result = {"seed": 20261002, "cases": []}
    for rows, context in [(64, 2048), (512, 8192), (2048, 32768)]:
        q = torch.randn(1, rows, 6144, device="cuda", dtype=torch.float16)
        gate = torch.randn_like(q)
        k = torch.randn(1, context, 1024, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        pos = torch.arange(context - rows, context, device="cuda", dtype=torch.int32)[None, :]
        length = torch.tensor([context], device="cuda", dtype=torch.int32)
        reference = torch.empty(1, rows, 24, 256, device="cuda", dtype=torch.float16)
        baseline = attention_prefill_staged(
            1, rows, context, kv_layout="token_major", block_m=32 if rows == 512 else 64
        ).torch_function
        baseline(q, k, v, gate, pos, length, reference)
        original_q = q.clone()
        original_k, original_v = k.clone(), v.clone()
        for bm, bn, stages, interior in [
            (32 if rows == 512 else 64, 32, 0, False),
            (32, 32, 0, True),
            (64, 32, 0, True),
            (32, 32, 1, True),
            (64, 32, 1, True),
        ]:
            name = f"q{rows}-kv{context}-m{bm}-n{bn}-s{stages}-mask{int(interior)}"
            print("compile", name, flush=True)
            kernel = attention_prefill_staged(
                1,
                rows,
                context,
                kv_layout="token_major",
                block_m=bm,
                block_n=bn,
                num_stages=stages,
                interior_mask=interior,
                contiguous_queries=stages > 0,
            )
            export_kernel(kernel, output / name)
            y = torch.empty_like(reference)
            q.copy_(original_q)
            k.copy_(original_k)
            v.copy_(original_v)
            length.fill_(context)
            pos.copy_(
                torch.arange(context - rows, context, device="cuda", dtype=torch.int32)[None, :]
            )
            baseline(q, k, v, gate, pos, length, reference)

            def call():
                kernel.torch_function(q, k, v, gate, pos, length, y)

            call()
            torch.cuda.synchronize()
            observed = error(y, reference)
            assert observed["finite"] and observed["relative_l2"] < 0.002, (name, observed)
            hot, graph = benchmark(call, repetitions=3)
            for n in [0, 1, 145, 1023, 1025, context]:
                k[:, n:].fill_(float("nan"))
                v[:, n:].fill_(float("nan"))
                length.fill_(n)
                pos.copy_(
                    torch.arange(
                        max(0, n - rows), max(0, n - rows) + rows, device="cuda", dtype=torch.int32
                    )[None, :]
                )
                pos[0, 0] = -1
                q.mul_(0.97)
                y.fill_(float("nan"))
                if stages:
                    k[:, n : min(context, ((n + bn - 1) // bn) * bn)].zero_()
                    v[:, n : min(context, ((n + bn - 1) // bn) * bn)].zero_()
                graph.replay()
                baseline(q, k, v, gate, pos, length, reference)
                torch.cuda.synchronize()
                e = error(y, reference)
                assert e["finite"] and e["relative_l2"] < 0.002, (name, n, e)
                assert bool((y[0, 0] == 0).all()), (name, "empty row")
                # Restore finite padding before the next length/depth.
                k[:, n:].normal_()
                v[:, n:].normal_()
            result["cases"].append(
                dict(name=name, rows=rows, context=context, error=observed, hot=hot)
            )
            write_json(output / "result.json", result)
            print(name, hot["median_ms"], observed, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
