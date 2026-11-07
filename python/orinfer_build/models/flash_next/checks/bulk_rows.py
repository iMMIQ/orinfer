"""Check bulk projection tiles, partial rows/columns and changed-input replay."""

import argparse
from pathlib import Path
import tilelang
import torch
from orinfer_build.tuning.projection_tuning import ProjectionTuner, make_prim, profile_stream
from orinfer_build.compiler.cuda.common import configure, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    configure()
    evidence = []
    for kind, n, k, dtype in [
        ("q4", 6144, 2560, "float16"),
        ("w8", 48, 2560, "float16"),
        ("w8", 640, 2560, "float16"),
        ("w8", 2560, 6144, "float16"),
    ]:
        spec = dict(kind=kind, rows=1024, outputs=n, inputs=k, dtype=dtype, dynamic=True)
        config = dict(
            algorithm="tensor_core",
            block_m=64,
            block_n=64 if n < 128 else 128,
            block_k=128,
        )
        kernel = profile_stream(
            tilelang.compile(
                make_prim(spec, config),
                out_idx=[],
                execution_backend="nvrtc",
                target={"kind": "cuda", "arch": "sm_87"},
            )
        )
        for rows in [129, 257, 513]:
            ProjectionTuner._validate(kernel, spec, config, rows)
            evidence.append(
                dict(kind=kind, outputs=n, inputs=k, rows=rows, exact=True, replay=True)
            )
    from kernels.cuda.model.e8p_prefill import e8p_prefill_grouped

    for n, k in [(1280, 2560), (2560, 640)]:
        spec = dict(
            kind="expert",
            rows=1024,
            outputs=n,
            inputs=k,
            dtype="float16",
            dynamic=True,
            experts=512,
        )
        tiles = (1024 * 10 + 15) // 16 + 512
        from kernels.shared.rows import explicit_rows
        from orinfer_build.tuning.projection_tuning import EXACT_PASSES

        kernel = profile_stream(
            tilelang.compile(
                explicit_rows(
                    e8p_prefill_grouped.__wrapped__(
                        10240, 512, tiles, n, k, dynamic_rows=True, row_major=True
                    )
                ),
                out_idx=[],
                execution_backend="nvrtc",
                target={"kind": "cuda", "arch": "sm_87"},
                pass_configs=EXACT_PASSES,
            )
        )
        group_kernel = profile_stream(
            tilelang.compile(
                explicit_rows(
                    e8p_prefill_grouped.__wrapped__(10240, 512, tiles, n, k, dynamic_rows=True)
                ),
                out_idx=[],
                execution_backend="nvrtc",
                target={"kind": "cuda", "arch": "sm_87"},
                pass_configs=EXACT_PASSES,
            )
        )
        for rows in [129, 257, 513]:
            ProjectionTuner._validate(kernel, spec, dict(block_m=16), rows)
            supply, _, check = ProjectionTuner._inputs(spec, rows)
            inputs = supply(kernel.get_profiler()._get_params(with_output=False))
            assignments = rows * 10
            inputs[:2] = [
                plane.view(torch.int8).view(assignments, k // 64, 32).transpose(0, 1).contiguous()
                for plane in inputs[:2]
            ]
            group_kernel.adapter.func(*inputs)
            torch.cuda.synchronize()
            check()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                group_kernel.adapter.func(*inputs)
            inputs[12].fill_(91)
            graph.replay()
            torch.cuda.synchronize()
            check()
            inputs[0].zero_()
            group_kernel.adapter.func(*inputs)
            torch.cuda.synchronize()
            changed = inputs[12].clone()
            inputs[12].fill_(91)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(inputs[12], changed)
            evidence.append(
                dict(kind="expert", outputs=n, inputs=k, rows=rows, exact=True, replay=True)
            )
    write_json(a.output / "results.json", dict(complete=True, cases=evidence))


if __name__ == "__main__":
    main()
