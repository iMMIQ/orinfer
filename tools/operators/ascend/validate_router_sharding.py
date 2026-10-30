"""Compare sharded FP32 router arithmetic with its original full projection."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json, file_hash
from tools.model.deepseek_v41.router_sharding import column_spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(8)
    torch.npu.set_device(args.device)
    torch.manual_seed(20261002)
    libraries = []

    def descriptors(plan):
        model = json.loads(plan.read_text())
        packet = next(p for p in model["ranks"][0]["packets"] if p.get("batch_decode"))
        found = {}
        for step in packet["body"]:
            if step["op"] != "kernel":
                continue
            path = plan.parent / step["manifest"]
            spec = json.loads(path.read_text())
            if spec["op"] in ("dense.fp32_partials", "dense.fp32_reduce", "memory.gather_columns"):
                found[spec["op"]] = (path, spec)
            if spec["op"] == "routing.route":
                break
        return found

    old, new = descriptors(args.baseline), descriptors(args.plan)
    capacity = new["dense.fp32_partials"][1]["parameters"]["_row_capacity"]
    physical, starts = column_spans(384, 8)
    x = torch.empty((capacity, 5120), dtype=torch.float32, device="npu")
    weight = (torch.randn(384, 5120) * 0.02).bfloat16().float().npu()
    full = torch.empty((capacity + 1, 384), device="npu")
    result = torch.empty_like(full)
    partial = torch.empty((capacity, 12, 8, 32), device="npu")
    parts = [torch.empty((capacity, 2, 8, 32), device="npu") for _ in range(8)]
    outputs = [torch.empty((capacity, physical), device="npu") for _ in range(8)]
    gathered = torch.empty(8 * capacity * physical, device="npu")

    def bind(description, tensors, rows):
        path, spec = description
        binary = path.parent / spec["library"]["file"]
        if file_hash(binary) != spec["library"]["sha256"]:
            raise ValueError("Router validation library identity differs")
        library = ctypes.CDLL(str(binary.resolve()))
        libraries.append(library)
        entry = library.orinfer_ascend_kernel_v2
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        entry.restype = ctypes.c_int32
        pointers = (ctypes.c_void_p * len(tensors))(*[t.data_ptr() for t in tensors])

        def launch():
            status = entry(pointers, len(tensors), rows, torch.npu.current_stream()._as_parameter_)
            if status:
                raise RuntimeError(f"Router projection launch: {status}")

        return launch

    reports = []
    for rows in (1, 3, 17, capacity):
        full_part = bind(old["dense.fp32_partials"], [x, weight, partial], rows)
        full_reduce = bind(old["dense.fp32_reduce"], [partial, full], rows)
        producers = [
            (
                bind(
                    new["dense.fp32_partials"],
                    [x, weight[first : first + physical], parts[r]],
                    rows,
                ),
                bind(new["dense.fp32_reduce"], [parts[r], outputs[r]], rows),
            )
            for r, first in enumerate(starts)
        ]
        restore = bind(new["memory.gather_columns"], [gathered, result], rows)

        def run():
            full_part()
            full_reduce()
            for rank, (project, reduce) in enumerate(producers):
                project()
                reduce()
                gathered[rank * rows * physical : (rank + 1) * rows * physical].copy_(
                    outputs[rank][:rows].flatten()
                )
            restore()

        x.normal_()
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        for step in range(2):
            x.normal_()
            if step == 0:
                x.zero_()
            full.fill_(12345)
            result.fill_(12345)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(result[:rows], full[:rows], atol=0, rtol=0)
            assert bool((result[rows:] == 12345).all()), "Router restore wrote the tail"
        graph.reset()
        reports.append(dict(rows=rows, exact_original_fp32=True, changed_input_graph=True))
        print(reports[-1], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, reports)


if __name__ == "__main__":
    main()
