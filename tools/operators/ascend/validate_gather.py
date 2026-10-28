"""Stress dynamic KV gather masks, DMA reuse and changed-input Graph replay."""

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
    args = p.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.operators.ascend import model_ops
    import importlib

    importlib.reload(model_ops)
    from tools.operators.ascend.build import compile_kernel
    from tools.model.deepseek_v41.tune import npu_benchmark

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    torch.npu.set_device(args.device)
    model = json.loads(args.plan.read_text())
    chosen = {}
    for packet in model["ranks"][0]["packets"]:
        for step in packet["body"]:
            if step["op"] != "kernel":
                continue
            path = args.plan.parent / step["manifest"]
            spec = json.loads(path.read_text())
            if spec["op"] == "model_ops.gather_attention":
                chosen[(packet["rows"] == 1, spec["parameters"]["compressed"])] = (path, spec)
    reports = []
    libraries = []
    for (static, compressed), (old_path, spec) in chosen.items():
        parameters = dict(spec["parameters"])
        row_capacity = parameters.pop("_row_capacity", None)
        cap = row_capacity or 1
        limit = parameters.pop("_grid_limit", 64)
        parameters.pop("rows")
        path = compile_kernel(
            model_ops.gather_attention(row_count(cap, not static), **parameters),
            spec["op"],
            spec["operands"],
            dict(rows=cap, **parameters),
            args.output / "cache",
            row_capacity=row_capacity,
            grid_limit=limit,
        )
        c, lc = parameters["capacity"], parameters["compressed_capacity"]
        selected = 640 if compressed else 128
        swa = torch.randn(c, 512).bfloat16()
        long = torch.randn(lc, 512).bfloat16()
        a = swa.npu()
        b = long.npu()
        sid = torch.empty(cap, 128, dtype=torch.int32, device="npu")
        lid = torch.empty(cap, 512, dtype=torch.int32, device="npu")
        out = torch.empty(cap + 1, selected, 512, dtype=torch.bfloat16, device="npu")
        mask = torch.empty(cap + 1, selected, dtype=torch.int32, device="npu")
        pointers = (ctypes.c_void_p * 6)(*[t.data_ptr() for t in (a, b, sid, lid, out, mask)])
        entries = []
        for file in (old_path, path):
            lib = ctypes.CDLL(str(file.parent / "kernel.so"))
            libraries.append(lib)
            call = getattr(
                lib, "orinfer_ascend_kernel_v1" if static else "orinfer_ascend_kernel_v2"
            )
            call.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            if static:
                call.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64, ctypes.c_void_p]
            call.restype = ctypes.c_int32
            entries.append(call)
        for rows in (1,) if static else (16, 17, 32, 129, 2048):
            if rows > cap:
                continue
            positions = torch.arange(rows) % 127
            si = torch.arange(128).expand(rows, -1).clone()
            si[si > positions[:, None]] = -1
            li = torch.arange(512).expand(rows, -1).clone()
            li[li > (positions[:, None] + 1) // 2] = -1
            sid[:rows].copy_(si.int())
            lid[:rows].copy_(li.int())
            old_errors = 0
            for i, call in enumerate(entries):
                for repetition in range(5):

                    def run():
                        stream = torch.npu.current_stream()._as_parameter_
                        status = (
                            call(pointers, 6, stream) if static else call(pointers, 6, rows, stream)
                        )
                        if status:
                            raise RuntimeError(f"KV gather launch failed: {status}")

                    out[rows : rows + 1].fill_(17)
                    mask[rows:].fill_(17)
                    run()
                    torch.npu.synchronize()
                    ids = torch.cat((si, li), 1) if compressed else si
                    wanted = (ids >= 0).int() - 1
                    # Valid flags are zero, invalid flags are minus one.
                    got = mask[:rows].cpu()
                    bad = int((got != wanted).sum())
                    samples = sorted({0, rows // 2, rows - 1})
                    sample_ids = ids[samples]
                    want = swa[si[samples].clamp_min(0)]
                    if compressed:
                        want = torch.cat((want, long[li[samples].clamp_min(0)]), dim=1)
                    want = want.masked_fill((sample_ids < 0)[:, :, None], 0)
                    actual = torch.stack([out[r].cpu() for r in samples])
                    bad += int((actual != want).sum())
                    if i == 0:
                        old_errors += bad
                    else:
                        assert bad == 0, (
                            f"KV gather mismatch rows={rows}, repetition={repetition}, bad={bad}"
                        )
                        assert bool((out[rows : rows + 1].cpu() == 17).all()) and bool(
                            (mask[rows:].cpu() == 17).all()
                        ), "KV gather wrote tail"
                if i == 1:
                    _, graph = npu_benchmark(run, 2, 1)
                    si = si.flip(0).contiguous()
                    li = li.flip(0).contiguous()
                    sid[:rows].copy_(si.int())
                    lid[:rows].copy_(li.int())
                    graph.replay()
                    torch.npu.synchronize()
                    ids = torch.cat((si, li), 1) if compressed else si
                    torch.testing.assert_close(
                        mask[:rows].cpu(), (ids >= 0).int() - 1, atol=0, rtol=0
                    )
            reports.append(
                dict(
                    rows=rows,
                    static=static,
                    compressed=compressed,
                    old_mismatches=old_errors,
                    fenced_exact=True,
                )
            )
            print(reports[-1], flush=True)
        del a, b, sid, lid, out, mask
        torch.npu.empty_cache()
    atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
