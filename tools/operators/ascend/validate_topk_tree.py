"""Compare fused TopK trees with the published three passes, including ties."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json
from tools.model.deepseek_v41.fuse_topk import merge_chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 16, 17, 64, 68, 128, 256])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    import torch_npu  # noqa: F401
    from tools.model.deepseek_v41.tune import npu_benchmark

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    original, model = (json.loads(p.read_text()) for p in (args.baseline, args.plan))
    before = next(p for p in original["ranks"][0]["packets"] if p["min_rows"] == 512)
    after = next(p for p in model["ranks"][0]["packets"] if p["min_rows"] == 512)
    descriptions, trees, fused, libraries, reports = {}, {}, {}, [], []

    def describe(step):
        key = step["manifest"]
        if key not in descriptions:
            descriptions[key] = json.loads((args.baseline.parent / key).read_text())
        return descriptions[key]

    for i in range(len(before["body"])):
        if (chain := merge_chain(before["body"], i, describe)) is not None:
            trees[describe(chain[0])["parameters"]["sources"]] = chain
    for step in after["body"]:
        if step["op"] == "kernel":
            spec = json.loads((args.plan.parent / step["manifest"]).read_text())
            if spec["op"] == "topk.merge_tree":
                fused[spec["parameters"]["sources"]] = step
    if set(fused) != {32, 64} or not set(fused) <= set(trees):
        raise ValueError("Require the actual32/64-source published merge trees")

    for sources in sorted(fused):
        spec = describe(trees[sources][0])
        capacity, selected = (
            spec["parameters"]["_row_capacity"],
            spec["parameters"].get("selected", 512),
        )
        candidate = torch.empty(spec["operands"][0]["shape"], dtype=torch.float32, device="npu")
        lengths_gpu = torch.empty((capacity, 1), dtype=torch.int32, device="npu")
        engines = []
        for root, steps in [
            (args.baseline.parent, trees[sources]),
            (args.plan.parent, [fused[sources]]),
        ]:
            tensors, calls = [candidate], []
            for step in steps:
                path = root / step["manifest"]
                desc = json.loads(path.read_text())
                out = torch.empty(desc["operands"][-1]["shape"], dtype=torch.float32, device="npu")
                library = ctypes.CDLL(str((path.parent / desc["library"]["file"]).resolve()))
                libraries.append(library)
                function = library.orinfer_ascend_kernel_v2
                function.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_uint64,
                    ctypes.c_uint64,
                    ctypes.c_void_p,
                ]
                function.restype = ctypes.c_int32
                operands = [tensors[-1], out]
                if desc["op"] == "topk.merge_tree":
                    operands.insert(1, lengths_gpu)
                pointers = (ctypes.c_void_p * len(operands))(*(t.data_ptr() for t in operands))
                calls.append((function, pointers))
                tensors.append(out)
            engines.append((tensors, calls))

        def inputs(rows, phase):
            row = np.arange(rows, dtype=np.int32)[:, None, None]
            leaf = np.arange(sources, dtype=np.int32)[None, :, None]
            lane = np.arange(selected, dtype=np.int32)[None, None, :]
            values = (
                np.broadcast_to(
                    (selected - 1 - lane) // (16 if phase == 0 else 8)
                    + leaf % (4 + phase)
                    + row % 3
                    - 20,
                    (rows, sources, selected),
                )
                .astype(np.float32)
                .copy()
            )
            labels = (
                np.broadcast_to(
                    leaf * 4096 + (lane if phase == 0 else selected - 1 - lane), values.shape
                )
                .astype(np.float32)
                .copy()
            )
            patterns = [513, 0, 8193, sources * 4096 - 4000, sources * 4096, 65536, 65537]
            lengths = np.array(patterns, dtype=np.int32)[(np.arange(rows) + phase) % len(patterns)]
            valid = np.maximum(0, np.minimum(selected, lengths[:, None, None] - leaf * 4096))
            mask = np.broadcast_to(lane >= valid, values.shape)
            values[mask] = -np.inf
            labels[mask] = -1
            pairs = np.stack([values, labels], axis=-1).reshape(rows, sources, 2 * selected)
            candidate[:rows].copy_(torch.from_numpy(pairs).npu())
            lengths_gpu[:rows, 0].copy_(torch.from_numpy(lengths).npu())

        for rows in args.rows:
            if not 1 <= rows <= capacity:
                raise ValueError("Query rows exceed the actual AOT capacity")

            def run(calls):
                for function, pointers in calls:
                    result = function(
                        pointers, len(pointers), rows, torch.npu.current_stream()._as_parameter_
                    )
                    if result:
                        raise RuntimeError(f"TopK tree launch failed: {result}")

            for tensors, _ in engines:
                for tensor in tensors[1:]:
                    tensor.fill_(17)
            inputs(rows, 0)
            timings, graphs = [], []
            for _, calls in engines:
                timing, graph = npu_benchmark(lambda: run(calls), 9, 2)
                timings.append(timing["median_ms"])
                graphs.append(graph)
            prior = [tensors[-1][:rows].clone() for tensors, _ in engines]
            torch.testing.assert_close(
                prior[1].view(torch.int32), prior[0].view(torch.int32), rtol=0, atol=0
            )
            inputs(rows, 1)
            for graph in graphs:
                graph.replay()
            torch.npu.synchronize()
            old, new = (tensors[-1] for tensors, _ in engines)
            torch.testing.assert_close(
                new[:rows].view(torch.int32), old[:rows].view(torch.int32), rtol=0, atol=0
            )
            for (tensors, _), saved in zip(engines, prior, strict=True):
                assert not torch.equal(
                    tensors[-1][:rows].view(torch.int32), saved.view(torch.int32)
                ), "Graph ignored changed candidates"
                for tensor in tensors[1:]:
                    assert bool((tensor[rows:] == 17).all()), "TopK wrote beyond actual query rows"
            for graph in graphs:
                graph.reset()
            result = dict(
                rows=rows,
                sources=sources,
                exact_bits=True,
                ties=True,
                inf_padding=True,
                changed_graph=True,
                tail_unchanged=True,
                baseline_ms=timings[0],
                candidate_ms=timings[1],
                speedup=timings[0] / timings[1],
            )
            reports.append(result)
            print(json.dumps(result), flush=True)
    atomic_json(args.output / "validation.json", dict(cases=reports))


if __name__ == "__main__":
    main()
