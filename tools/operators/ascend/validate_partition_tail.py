"""Check production short collective tails and changed-input Graph replay."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    model = json.loads(args.plan.read_text())
    paths = {
        s["manifest"]
        for p in model["ranks"][0]["packets"]
        for s in p["body"]
        if s["op"] == "kernel"
    }
    reports = []
    for relative in sorted(paths):
        path = args.plan.parent / relative
        spec = json.loads(path.read_text())
        if spec["op"] != "memory.zero_row_tail" or spec["parameters"].get("partitions") != 8:
            continue
        parameters = spec["parameters"]
        capacity, width = parameters["capacity"], parameters["width"]
        dtype = dict(bf16=torch.bfloat16, f32=torch.float32)[spec["operands"][1]["dtype"]]
        positions = torch.zeros(capacity, dtype=torch.int32, device="npu")
        output = torch.empty((capacity, width), dtype=dtype, device="npu")
        lib = ctypes.CDLL(str(path.parent / spec["library"]["file"]))
        fn = lib.orinfer_ascend_kernel_v2
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int32
        pointers = (ctypes.c_void_p * 2)(positions.data_ptr(), output.data_ptr())

        def launch(rows):
            status = fn(pointers, 2, rows, torch.npu.current_stream()._as_parameter_)
            if status:
                raise RuntimeError(f"Partition tail launch failed: {status}")

        samples = sorted(
            {
                r
                for r in [
                    1,
                    2,
                    7,
                    8,
                    9,
                    17,
                    63,
                    64,
                    65,
                    127,
                    128,
                    129,
                    511,
                    512,
                    513,
                    capacity - 1,
                    capacity,
                ]
                if r <= capacity
            }
        )
        for rows in samples:
            end = (rows + 7) // 8 * 8
            for captured, sentinel in [(False, 17), (True, 19)]:
                if captured:
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        launch(rows)
                    positions.add_(1)
                output.fill_(sentinel)
                if captured:
                    graph.replay()
                else:
                    launch(rows)
                torch.npu.synchronize()
                expected = torch.full((capacity, width), sentinel, dtype=dtype)
                expected[rows:end] = 0
                assert torch.equal(output.cpu(), expected), (relative, rows, captured)
                if captured:
                    del graph
        reports.append(
            dict(manifest=relative, parameters=parameters, rows=samples, changed_graph=True)
        )
    if not reports:
        raise ValueError("No production partition-tail kernels found")
    atomic_json(args.output, dict(seed=20261002, checks=reports))
    print(json.dumps(dict(kernels=len(reports))))


if __name__ == "__main__":
    main()
