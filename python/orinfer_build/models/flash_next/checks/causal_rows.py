"""Guard dynamic history/factor strides, hidden-ring wrap and actual graph replay."""

import argparse
from pathlib import Path
import torch
import tilelang
from kernels.cuda.model import flash_mtp, flash_control, speculation, speculative_factors
from kernels.cuda.model.causal_rows import copy as copy_rows
from kernels.shared.rows import explicit_rows
from orinfer_build.tuning.projection_tuning import profile_stream
from orinfer_build.compiler.cuda.common import configure, write_json


def compile_rows(fn):
    return profile_stream(
        tilelang.compile(
            explicit_rows(fn),
            out_idx=[],
            execution_backend="nvrtc",
            target={"kind": "cuda", "arch": "sm_87"},
        )
    )


def checked(kernel, inputs, outputs, rows, reference, change):
    def launch():
        kernel.adapter.func(*inputs, *[o for o, _ in outputs], rows)

    launch()
    torch.cuda.synchronize()
    for (actual, guard), expected in zip(outputs, reference()):
        assert torch.equal(actual, expected)
        assert bool((guard == 91).all())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    change()
    for out, _ in outputs:
        out.fill_(91)
    graph.replay()
    torch.cuda.synchronize()
    for (actual, guard), expected in zip(outputs, reference()):
        assert torch.equal(actual, expected)
        assert bool((guard == 91).all())


def output(shape, dtype=torch.float16):
    n = 1
    for dim in shape:
        n *= dim
    storage = torch.full((n + 64,), 91, dtype=dtype, device="cuda")
    return storage[:n].view(shape), storage[n:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    configure()
    torch.manual_seed(20261002)
    evidence = []
    for width in [2560, 10240]:
        kernel = compile_rows(copy_rows.__wrapped__(512, width))
        for n in [1, 3, 17, 129, 257, 511, 512]:
            x = torch.randn(n, width, dtype=torch.float16, device="cuda")
            checked(kernel, [x], [output((n, width))], n, lambda: [x], lambda: x.neg_())
            evidence.append(dict(kind="row-copy", rows=n, width=width, exact=True, replay=True))
    for history in [3, 9]:
        kernel = compile_rows(
            flash_mtp.history_prefix.__wrapped__(512, 10240, history, dynamic_rows=True)
        )
        for n in [1, 5, 8, 17, 129, 257, 512]:
            x = torch.randn(n, 10240, dtype=torch.float16, device="cuda")
            state = torch.randn(history, 10240, dtype=torch.float16, device="cuda")

            def reference():
                joined = torch.cat([state, x])
                return [torch.stack([joined[t + 1 : t + 1 + history] for t in range(n)])]

            checked(
                kernel, [x, state], [output((n, history, 10240))], n, reference, lambda: x.mul_(-1)
            )
            evidence.append(dict(kind="history", rows=n, history=history, exact=True, replay=True))
    ring = torch.randn(4096, 10240, dtype=torch.float16, device="cuda")
    gather = compile_rows(
        speculation.gather_target_hidden.__wrapped__(512, 10240, 4096, ring=True, dynamic_rows=True)
    )
    last = compile_rows(flash_control.last_hidden.__wrapped__(512, 10240, dynamic_rows=True))
    for n in [1, 5, 17, 129, 257, 512]:
        position = torch.tensor([4093], dtype=torch.int32, device="cuda")
        checked(
            gather,
            [ring, position],
            [output((n, 10240))],
            n,
            lambda: [ring[(torch.arange(n, device="cuda") + position[0]) % 4096]],
            lambda: position.add_(4094),
        )
        x = torch.randn(n, 10240, dtype=torch.float16, device="cuda")
        checked(last, [x], [output((1, 10240))], n, lambda: [x[n - 1 : n]], lambda: x.mul_(-1))
        evidence.append(dict(kind="hidden-ring", rows=n, exact=True, replay=True))
    save = compile_rows(speculative_factors.save_gdn_factors.__wrapped__(8, dynamic_rows=True))
    for n in range(2, 9):
        k = torch.randn(16, n, 128, dtype=torch.float16, device="cuda")
        g = torch.randn(n, 48, dtype=torch.float32, device="cuda")
        checked(
            save,
            [k, g],
            [output(k.shape), output(g.shape, torch.float32)],
            n,
            lambda: [k, g],
            lambda: k.mul_(-1),
        )
        evidence.append(dict(kind="gdn-factors", rows=n, exact=True, replay=True))
    write_json(a.output / "results.json", dict(complete=True, cases=evidence))


if __name__ == "__main__":
    main()
