"""Check real-weight indexer pipelines, nonaligned rows and changed-input Graphs."""

import argparse
import ctypes
import json
import math
from pathlib import Path

from tools.build.files import atomic_json
from tools.model.deepseek_v41.local_indexer import query_chain
from tools.model.row_contract import resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--rows", type=int, nargs="+", default=[17, 129, 512, 544, 1921, 2048])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_npu  # noqa: F401
    from tools.model.deepseek_v41.tune import npu_benchmark, read_bank

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    kinds = dict(
        i8=torch.int8, bf16=torch.bfloat16, f16=torch.float16, f32=torch.float32, i32=torch.int32
    )
    baseline, candidate = (json.loads(p.read_text()) for p in (args.baseline, args.plan))
    checkpoint = json.loads((args.model_dir / baseline["index"]["file"]).read_text())["weight_map"]
    libraries, reports = [], []
    for rank_id in (0, 7):
        original = baseline["ranks"][rank_id]
        packet = next(p for p in original["packets"] if p["min_rows"] == 512)

        def describe(step):
            return json.loads((args.baseline.parent / step["manifest"]).read_text())

        chain = next(
            c
            for i in range(len(packet["body"]))
            if (c := query_chain(packet["body"], i, describe)) is not None
        )
        new_rank = candidate["ranks"][rank_id]
        new_packet = next(p for p in new_rank["packets"] if p["min_rows"] == 512)
        start = next(
            i
            for i, s in enumerate(new_packet["body"])
            if s.get("buffers", [None, None])[0] == chain[0]["buffers"][0]
            and s.get("buffers", [None, None])[1] == "batch.indexer.local.m2048.activation"
        )
        local = new_packet["body"][start : start + 9]
        if len(local) != 9 or local[-1]["buffers"][4] != chain[-1]["buffers"][1]:
            raise ValueError("Local indexer producer does not match the original query boundary")
        engines = []
        for model_rank, root, steps in [
            (original, args.baseline.parent, chain),
            (new_rank, args.plan.parent, local),
        ]:
            definitions = {b["name"]: b for b in model_rank["buffers"]}
            raw = {}

            def allocation(name):
                if name not in raw:
                    definition = definitions[name]
                    view = definition.get("view")
                    raw[name] = (
                        allocation(view["parent"])[
                            view["offset"] : view["offset"] + definition["bytes"]
                        ]
                        if view
                        else torch.zeros(definition["bytes"], dtype=torch.uint8, device="npu")
                    )
                return raw[name]

            calls = []
            for step in steps:
                path = root / step["manifest"]
                spec = json.loads(path.read_text())
                values = []
                for name, port in zip(step["buffers"], spec["operands"], strict=True):
                    value = (
                        allocation(name)
                        .view(kinds[port["dtype"]])[: math.prod(port["shape"])]
                        .view(port["shape"])
                    )
                    definition = definitions[name]
                    if "weight" in definition or "prepared_weight" in definition:
                        value.copy_(
                            read_bank(
                                root,
                                args.model_dir,
                                checkpoint,
                                definition,
                                port["dtype"],
                                port["shape"],
                            ).npu()
                        )
                    else:
                        root_name = name
                        while "view" in definitions[root_name]:
                            root_name = definitions[root_name]["view"]["parent"]
                        if root_name.startswith("ones"):
                            value.fill_(1)
                    values.append(value)
                library = ctypes.CDLL(str((path.parent / spec["library"]["file"]).resolve()))
                libraries.append(library)
                function = library.orinfer_ascend_kernel_v2
                function.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_void_p,
                ]
                function.restype = ctypes.c_int32
                pointers = (ctypes.c_void_p * len(values))(*[v.data_ptr() for v in values])
                calls.append((function, pointers, step))
            engines.append((raw, calls))

        inputs = [
            chain[0]["buffers"][0],
            chain[0]["buffers"][2],
            chain[3]["buffers"][0],
            *chain[1]["buffers"][1:3],
        ]
        data = [
            torch.randint(-63, 64, (2048, 1280), dtype=torch.int8, device="npu"),
            torch.rand(2048, dtype=torch.float32, device="npu") * 0.02 + 0.001,
            torch.randn(2048, 5120, device="npu").bfloat16(),
            torch.randn(2048, 32, device="npu").cos(),
            torch.randn(2048, 32, device="npu").sin(),
        ]
        for rows in args.rows:
            if not 1 <= rows <= 2048:
                raise ValueError("Rows must fit the published packet capacity")

            def initialize(sign):
                for raw, _ in engines:
                    for i, (name, value) in enumerate(zip(inputs, data, strict=True)):
                        target = raw[name].view(value.dtype).view(value.shape)
                        target.copy_(value * sign if i in (0, 2) else value)

            def run(calls):
                for function, pointers, step in calls:
                    live = (
                        resolve(step["row_expression"], rows) if "row_expression" in step else rows
                    )
                    status = function(
                        pointers, len(pointers), live, torch.npu.current_stream()._as_parameter_
                    )
                    if status:
                        raise RuntimeError(f"AOT indexer launch failed: {status}")

            initialize(1)
            measurements, graphs = [], []
            for _, calls in engines:
                timing, graph = npu_benchmark(lambda: run(calls), 9, 2)
                measurements.append(timing["median_ms"])
                graphs.append(graph)
            valid = max(0, (rows - rank_id + 7) // 8)
            query_name = chain[-3]["buffers"][1]
            before = [
                raw[query_name].view(torch.int8).view(256, 4096)[:valid].clone()
                for raw, _ in engines
            ]
            initialize(-1)
            for graph in graphs:
                graph.replay()
            torch.npu.synchronize()
            if valid:
                for (raw, _), prior in zip(engines, before, strict=True):
                    actual_query = raw[query_name].view(torch.int8).view(256, 4096)[:valid]
                    assert actual_query.abs().max().item() > 0, "Query projection did not execute"
                    assert not torch.equal(actual_query, prior), "Graph ignored changed input"
            for part, width, dtype in zip(
                chain[-3:], (4096, 32, 32), (torch.int8, torch.float16, torch.bfloat16), strict=True
            ):
                name = part["buffers"][1]
                expected, actual = (
                    raw[name].view(dtype).view(256, width)[:valid] for raw, _ in engines
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for graph in graphs:
                graph.reset()
            result = dict(
                rank=rank_id,
                rows=rows,
                valid_query_rows=valid,
                real_weights=True,
                exact_changed_graph=True,
                baseline_ms=measurements[0],
                candidate_ms=measurements[1],
                speedup=measurements[0] / measurements[1],
            )
            reports.append(result)
            print(json.dumps(result), flush=True)
    atomic_json(
        args.output / "validation.json",
        dict(
            scope="Only valid owner query rows are compared; unowned padded scale lanes may contain the quantizer minimum instead of zero and are never consumed.",
            cases=reports,
        ),
    )


if __name__ == "__main__":
    main()
