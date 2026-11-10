"""Measure real-weight routed W4A8 and shared bit-plane expert chains."""

import argparse
import ctypes
import gc
import json
from pathlib import Path

from orinfer_build.common.files import atomic_json
from orinfer_build.common.shared_json import decode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--routing", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=7)
    p.add_argument("--layers", type=int, nargs="+", default=[0])
    p.add_argument("--hints", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_npu  # noqa: F401
    from safetensors import safe_open
    from tests.ascend.grouped import GroupedMatmul
    from orinfer_build.tuning.ascend import npu_benchmark

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    model = decode(json.loads(args.plan.read_text()))
    rank = model["ranks"][0]
    defs = {v["name"]: v for v in rank["buffers"]}
    old = next(v for v in rank["packets"] if v.get("min_rows") == 128)
    new = next(v for v in rank["packets"] if v.get("min_rows") == 32)
    routing = json.loads(args.routing.read_text())["routing"]
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    libraries = []
    results, graph_checks = [], []

    def entry(step):
        path = args.plan.parent / step["manifest"]
        spec = json.loads(path.read_text())
        lib = ctypes.CDLL(str(path.parent / spec["library"]["file"]))
        libraries.append(lib)
        call = lib.orinfer_ascend_kernel_v2
        call.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        call.restype = ctypes.c_int32

        def run(values, rows):
            ptr = (ctypes.c_void_p * len(values))(*[v.data_ptr() for v in values])
            status = call(ptr, len(values), rows, torch.npu.current_stream()._as_parameter_)
            if status:
                raise RuntimeError(f"{spec['op']}: {status}")

        return run

    def read(name):
        tensor = defs[name]["prepared_tensor"]
        with safe_open(
            args.model_dir / index[tensor["name"]], framework="pt", device="cpu"
        ) as file:
            return file.get_tensor(tensor["name"])

    for layer in args.layers:
        oi = next(
            i
            for i, s in enumerate(old["body"])
            if s["op"] == "w4" and s["name"] == f"m2048.L{layer}.up"
        )
        ni = next(
            i
            for i, s in enumerate(new["body"])
            if s["op"] == "w4_a4" and s["name"] == f"m2048.L{layer}.up"
        )
        og, up, activation = old["body"][oi - 1 : oi + 2]
        ng, nu, nf = new["body"][ni - 1 : ni + 2]
        capacity = old["rows"]
        m, k, n, e = [up[x] for x in ["rows", "input", "output", "experts"]]
        w = n // 2
        weight = read(up["buffers"][1]).view(torch.int8).npu()
        scale = read(up["buffers"][2]).view(torch.int64).npu()
        assist = read(up["buffers"][3]).reshape(e, n).npu()
        table = torch.randint(-128, 128, (capacity, k), dtype=torch.int8, device="npu")
        scales = torch.rand(capacity, device="npu") * 0.02 + 0.001
        ids = torch.full((m,), -1, dtype=torch.int32, device="npu")
        groups = torch.zeros(e, dtype=torch.int64, device="npu")
        doubled = torch.zeros_like(groups)
        gathered = torch.zeros((m, k), dtype=torch.int8, device="npu")
        token_scales = torch.zeros(m, device="npu")
        packed = torch.zeros((m * 2, k // 2), dtype=torch.int8, device="npu")
        packed_scales = torch.zeros(m, device="npu")
        errors = torch.empty(m, dtype=torch.int32, device="npu")
        ones = torch.ones(m * 2, device="npu")
        raw = torch.empty((m, n), dtype=torch.bfloat16, device="npu")
        partial = torch.empty((m * 2, n), dtype=torch.float16, device="npu")
        out = torch.full((m + 1, w), 17, dtype=torch.int8, device="npu")
        out_scale = torch.full((m + 1,), 17.0, device="npu")
        expected = torch.empty_like(out)
        expected_scale = torch.empty_like(out_scale)
        active = torch.zeros(capacity + 1, dtype=torch.int32, device="npu")
        gather, act, split, finish = [entry(s) for s in [og, activation, ng, nf]]
        baseline = GroupedMatmul(
            gathered,
            weight,
            scale,
            assist,
            token_scales,
            groups,
            raw,
            rows=m,
            reduction=k,
            columns=n,
            experts=e,
            expected_rows=up["expected_expert_rows"],
        )
        if nu["expected_expert_rows"] not in args.hints:
            args.hints.append(nu["expected_expert_rows"])
        reference_hint = nu["expected_expert_rows"]
        plans = {
            hint: GroupedMatmul(
                packed,
                weight,
                scale,
                assist,
                ones,
                doubled,
                partial,
                rows=m * 2,
                reduction=k,
                columns=n,
                experts=e,
                split=True,
                expected_rows=hint,
            )
            for hint in args.hints
        }
        graphs = []
        for rows in [96, 127, 128, 192, 2048]:
            sample = next(s for s in routing if s["rows"] == rows and not s["decode"])
            counts = next(g["counts"] for g in sample["ranks"][0] if g["buffer"] == up["groups"])
            live = sum(counts)
            if rows == 2048 and not live:
                continue
            groups.copy_(torch.tensor(counts, dtype=torch.int64))
            ids.fill_(-1)
            ids[:live] = (torch.arange(live, dtype=torch.int32, device="npu") * 13 + 7) % rows
            active.zero_()
            active[rows] = live
            table.neg_()

            def original():
                gather([table, scales, ids, gathered, token_scales, errors], rows)
                baseline()
                act([raw, expected, expected_scale, active], rows)

            def fused(hint):
                split([table, scales, ids, groups, packed, packed_scales, errors, doubled], rows)
                plans[hint]()
                finish([partial, assist, packed_scales, groups, out, out_scale], rows)

            original()
            torch.npu.synchronize()
            timing, graph = npu_benchmark(original, repetitions=7, warmup=2, calls_per_replay=8)
            graphs.append(graph)
            record = dict(
                layer=layer, rows=rows, live=live, counts=counts, baseline=timing, candidates={}
            )
            for hint in args.hints:
                out.fill_(17)
                out_scale.fill_(17.0)
                fused(hint)
                torch.npu.synchronize()
                mismatch = int((out[:live] != expected[:live]).sum())
                scale_abs = (
                    float((out_scale[:live] - expected_scale[:live]).abs().max()) if live else 0.0
                )
                if mismatch or scale_abs:
                    raise AssertionError(
                        dict(rows=rows, hint=hint, mismatch=mismatch, scale_abs=scale_abs)
                    )
                assert bool((out[live:] == 17).all()) and bool((out_scale[live:] == 17).all())
                timing, graph = npu_benchmark(
                    lambda: fused(hint), repetitions=7, warmup=2, calls_per_replay=8
                )
                graphs.append(graph)
                record["candidates"][str(hint)] = timing
            for name, call in [
                (
                    "gather",
                    lambda: gather([table, scales, ids, gathered, token_scales, errors], rows),
                ),
                ("gmm", baseline),
                ("activation", lambda: act([raw, expected, expected_scale, active], rows)),
                (
                    "split",
                    lambda: split(
                        [table, scales, ids, groups, packed, packed_scales, errors, doubled], rows
                    ),
                ),
                ("a4_gmm", plans[reference_hint]),
                (
                    "finish",
                    lambda: finish([partial, assist, packed_scales, groups, out, out_scale], rows),
                ),
            ]:
                timing, graph = npu_benchmark(call, repetitions=7, warmup=2, calls_per_replay=8)
                graphs.append(graph)
                record[name] = timing
            results.append(record)
            atomic_json(args.output / "result.json", dict(cases=results))
            print(record, flush=True)
        # Captured commands must consume changed inputs, including no routes
        # and an isolated route to the final expert, without touching tails.
        rows = 192
        fused(reference_hint)
        changed_graph = torch.npu.NPUGraph()
        with torch.npu.graph(changed_graph):
            fused(reference_hint)
        changed_graph.replay()
        for live in [0, 1]:
            groups.zero_()
            groups[-1] = live
            ids.fill_(-1)
            if live:
                ids[0] = 7
            active.zero_()
            active[rows] = live
            table.neg_()
            out.fill_(17)
            out_scale.fill_(17.0)
            original()
            changed_graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(out[:live], expected[:live], atol=0, rtol=0)
            torch.testing.assert_close(out_scale[:live], expected_scale[:live], atol=0, rtol=0)
            assert bool((out[live:] == 17).all()) and bool((out_scale[live:] == 17).all())
            graph_checks.append(
                dict(
                    layer=layer,
                    rows=rows,
                    live=live,
                    changed_graph=True,
                    exact=True,
                    tail_preserved=True,
                )
            )
        del changed_graph
        torch.npu.synchronize()
        graphs.clear()
        del graph
        gc.collect()
        for plan in [baseline, *plans.values()]:
            plan.close()
        torch.npu.empty_cache()

    atomic_json(
        args.output / "result.json", dict(cases=results, graph_checks=graph_checks, seed=20261002)
    )


if __name__ == "__main__":
    main()
