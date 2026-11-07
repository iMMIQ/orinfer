"""Check shared DSpark block packing, physical tails and changed-input Graphs."""

import argparse
import ctypes
import json
from pathlib import Path

from orinfer_build.common.files import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from orinfer_build.tuning.ascend import npu_benchmark

    torch.manual_seed(20261002)
    torch.npu.set_device(7)
    model = json.loads(args.plan.read_text())
    paths = set()
    for packet in model["ranks"][0]["packets"]:
        for step in packet["body"]:
            if step["op"] == "kernel":
                path = args.plan.parent / step["manifest"]
                spec = json.loads(path.read_text())
                if spec["op"] == "memory.pad_blocks":
                    paths.add(path)
    if not paths:
        raise ValueError("Plan has no shared-stride draft boundary")
    reports = []
    for path in sorted(paths):
        spec = json.loads(path.read_text())
        p = spec["parameters"]
        width, capacity, stride = p["width"], p["table_rows"], p["block_size"]
        owners = p["rows"]
        dtype = dict(bfloat16=torch.bfloat16, float32=torch.float32, int32=torch.int32)[p["dtype"]]
        library = ctypes.CDLL(str(path.parent / spec["library"]["file"]))
        entry = library.orinfer_ascend_kernel_v2
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        entry.restype = ctypes.c_int32
        source = torch.randint(-127, 128, (capacity + 1, width)).to(dtype).npu()
        endpoints = torch.zeros(owners, dtype=torch.int32, device="npu")
        out = torch.full((owners * stride + 1, width), 17, dtype=dtype, device="npu")
        pointers = (ctypes.c_void_p * 3)(source.data_ptr(), endpoints.data_ptr(), out.data_ptr())
        for batch in (1, 3, owners):

            def launch():
                status = entry(pointers, 3, batch, torch.npu.current_stream()._as_parameter_)
                if status:
                    raise RuntimeError(f"Block packing launch failed: {status}")

            endpoints[:batch].copy_(
                torch.arange(1, batch + 1, dtype=torch.int32, device="npu") * min(5, stride) - 1
            )
            launch()
            _, graph = npu_benchmark(launch, 2, 1)
            for block in sorted({min(5, stride), min(8, stride), stride}):
                source.neg_()
                lengths = [block - (i % 2) for i in range(batch)]
                ends = torch.tensor(lengths, dtype=torch.int32).cumsum(0).int() - 1
                endpoints[:batch].copy_(ends)
                out.fill_(17)
                graph.replay()
                torch.npu.synchronize()
                wanted = torch.full(out.shape, 17, dtype=dtype)
                values, first = source.cpu(), 0
                for owner, count in enumerate(lengths):
                    wanted[owner * stride : owner * stride + count] = values[first : first + count]
                    first += count
                torch.testing.assert_close(out.cpu(), wanted, atol=0, rtol=0)
                reports.append(
                    dict(
                        width=width,
                        dtype=p["dtype"],
                        batch=batch,
                        block=block,
                        changed_input_graph=True,
                    )
                )
                print(json.dumps(reports[-1]), flush=True)
    atomic_json(args.output, reports)


if __name__ == "__main__":
    main()
