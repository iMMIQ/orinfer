"""Real FFN paired strict W8 expansion, original vs native I8 packed W4.

Full GPU output equality, independent CPU sentinel-row code reference, lossless
packed inverse, and graph input mutation. Isolated reader, not model TPS.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from common import benchmark, configure, environment, export_kernel, identity, write_json
from kernels.model.w4_i8_to_temporary_w8 import w4_i8_to_temporary_w8
from kernels.operators.op29_w4_to_temporary_w8 import w4_warp_to_temporary_w8
from tools.quantization.w4_i8_pack import pack_array


def sentinel_codes(pp, s, z, ws, rows):
    """Original F16 layout decoded by scalar row/K equations in NumPy."""
    k = s.shape[1] * 128
    j = np.arange(k, dtype=np.uint32)
    result = []
    for row in rows:
        i = row % 64
        lane = (i // 16) * 32 + (i % 8) * 4 + ((j % 8) // 2)
        shift = ((i % 16 // 8) * 4 + ((j % 16) // 8) * 2 + (j % 2)) * 4
        code = (
            (pp.view(np.uint32)[row // 64, j // 128, lane, (j % 128) // 16] >> shift) & 15
        ).astype(np.int32)
        # Exact original half source boundary, then FP32 division and RNE.
        weight = (
            (code - z[row, j // 128].astype(np.int32)).astype(np.float16) * s[row, j // 128]
        ).astype(np.float16)
        ratio = weight.astype(np.float32) / ws[row].astype(np.float32)
        result.append(np.clip(np.rint(ratio), -127, 127).astype(np.int8))
    return np.stack(result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--validation-only", action="store_true")
    ap.add_argument("--rows", type=int)
    args = ap.parse_args()
    configure()
    assert args.rows is None or args.rows > 0 and args.rows % 64 == 0
    report = dict(
        status="running",
        environment=environment(),
        model=identity(args.model),
        cases=[],
        sources=[],
        validation_only=args.validation_only,
        scope="Actual L0 weights, paired lossless layouts; complete expansion, not model performance or quality",
    )
    for name in [
        "kernels/model/w4_i8_to_temporary_w8.py",
        "kernels/operators/op29_w4_to_temporary_w8.py",
        "kernels/operators/op03_ffn_gate_up.py",
        "tools/quantization/w4_i8_pack.py",
    ]:
        p = Path(name)
        shutil.copyfile(p, args.output / p.name)
        report["sources"].append(identity(p))
    model = json.loads(args.model.read_text())
    buffers = {b["name"]: b for b in model["buffers"]}
    for name, bk in [("L0_GateUp", 512), ("L0_Down", 256)]:
        arrays = []
        for suffix, dtype in [
            ("_P", np.int32),
            ("_S", np.float16),
            ("_Z", np.int8),
            ("_WS", np.float16),
        ]:
            b = buffers[name + suffix]
            arrays.append(
                np.fromfile(args.model.parent / b["data"]["file"], dtype=dtype).reshape(b["shape"])
            )
        original, s, z, ws = arrays
        n, k = s.shape[0], s.shape[1] * 128
        if args.rows is not None:
            assert args.rows <= n
            n = args.rows
            original = original[: n // 64].copy()
            s = s[:n].copy()
            z = z[:n].copy()
            ws = ws[:n].copy()
        native = pack_array(original, verify=True)
        rows = sorted({0, 7, 8, 15, 16, 31, 32, 63, min(64, n - 1), n - 1})
        expected = sentinel_codes(original, s, z, ws, rows)
        old_p, new_p = (torch.from_numpy(a.view(np.int32)).cuda() for a in (original, native))
        scales, zeros, row_scale = (torch.from_numpy(a).cuda() for a in (s, z, ws))
        out = torch.empty((n, k), device="cuda", dtype=torch.int8)
        reference = torch.empty_like(out)
        old = w4_warp_to_temporary_w8(n, k, BK=bk)
        new = w4_i8_to_temporary_w8(n, k, BK=bk)

        def baseline():
            old.adapter.func(
                old_p,
                scales,
                zeros,
                row_scale,
                reference,
                stream=torch.cuda.current_stream().cuda_stream,
            )

        def run():
            new.adapter.func(
                new_p, scales, zeros, row_scale, out, stream=torch.cuda.current_stream().cuda_stream
            )

        if args.validation_only:
            baseline()
            run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            timing = bt = None
        else:
            bt, _ = benchmark(baseline, repetitions=12)
            timing, graph = benchmark(run, repetitions=12)
        assert torch.equal(out, reference), {"mismatches": int((out != reference).sum())}
        assert np.array_equal(out[rows].cpu().numpy(), expected)
        saved = scales.clone()
        golden = out.clone()
        scales.zero_()
        out.fill_(71)
        graph.replay()
        torch.cuda.synchronize()
        assert bool((out == 0).all())
        scales.copy_(saved)
        out.fill_(71)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, golden)
        export_kernel(new, args.output / name)
        row = dict(
            weight=name,
            N=n,
            K=k,
            BK=bk,
            timing=timing,
            baseline_timing=bt,
            outputs_identical=True,
            independent_cpu_rows=rows,
            cpu_codes_identical=True,
            full_packed_roundtrip=True,
            graph_zero_restore=True,
            packed_bytes=new_p.numel() * 4,
        )
        report["cases"].append(row)
        write_json(args.output / "result.json", report)
        print(json.dumps(row), flush=True)
        old_p = None
        new_p = None
        scales = None
        zeros = None
        row_scale = None
        out = None
        reference = None
        del saved
        del golden
    report["status"] = "passed"
    write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
