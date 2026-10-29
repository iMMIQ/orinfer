"""Check FP32 compressor projection UB reuse across bounded dynamic row grids."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.build.files import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=7)
    p.add_argument("--allow-failure", action="store_true")
    args = p.parse_args()
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    model = json.loads(args.plan.read_text())
    packet = next(p for p in model["ranks"][0]["packets"] if p.get("batch_decode"))
    entries = {}
    for step in packet["body"]:
        if step["op"] != "kernel":
            continue
        path = args.plan.parent / step["manifest"]
        s = json.loads(path.read_text())
        if (
            s["op"] in ("dense.fp32_partials", "dense.fp32_reduce")
            and s["parameters"]["output"] == 512
        ):
            lib = ctypes.CDLL(str((path.parent / "kernel.so").resolve()))
            fn = lib.orinfer_ascend_kernel_v2
            fn.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            fn.restype = ctypes.c_int32
            entries[s["op"]] = (lib, fn)
    assert len(entries) == 2

    def call(op, tensors, rows):
        pointers = (ctypes.c_void_p * len(tensors))(*[x.data_ptr() for x in tensors])
        status = entries[op][1](
            pointers, len(tensors), rows, torch.npu.current_stream()._as_parameter_
        )
        if status:
            raise RuntimeError(f"{op}: {status}")

    cap = 128
    weight = torch.randn(512, 5120) * 0.01
    activation = torch.randn(cap, 5120).bfloat16()
    w = weight.npu()
    x = activation.npu()
    partial = torch.zeros((cap, 16, 8, 32), device="npu")
    out = torch.zeros((cap, 512), device="npu")
    reference_partial = torch.zeros_like(partial)
    reference_out = torch.zeros_like(out)
    reports = []
    for rows in [1, 2, 3, 7, 31, 128]:
        partial.fill_(23)
        out.fill_(29)
        call("dense.fp32_partials", [x, w, partial], rows)
        call("dense.fp32_reduce", [partial, out], rows)
        torch.npu.synchronize()
        batch = out[:rows].cpu()
        reference = []
        for row in range(rows):
            call("dense.fp32_partials", [x[row:], w, reference_partial], 1)
            call("dense.fp32_reduce", [reference_partial, reference_out], 1)
            torch.npu.synchronize()
            reference.append(reference_out[0].cpu())
        serial = torch.stack(reference)
        want = activation[:rows].float() @ weight.T
        item = dict(
            rows=rows,
            exact_serial=torch.equal(batch, serial),
            max_serial=float((batch - serial).abs().max()),
            max_torch=float((batch - want).abs().max()),
            tail_intact=bool(torch.all(out[rows:] == 29)),
        )
        reports.append(item)
        print(item, flush=True)
    atomic_json(args.output, reports)
    if not args.allow_failure:
        assert all(
            r["exact_serial"] and r["max_torch"] < 1e-4 and r["tail_intact"] for r in reports
        )


if __name__ == "__main__":
    main()
