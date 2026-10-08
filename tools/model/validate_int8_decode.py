"""Real-weight decode INT8 screening, independent references and graph replay.

Captured activations may be supplied; random inputs only test implementation.
Projection error against current W4 is diagnostic, not BF16/FP8 model acceptance.
"""

import argparse
import gc
import hashlib
import json
import mmap
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import numpy as np
    import torch
    from kernels.model.w4a8_decode import w4a8_decode
    from kernels.model.w4_small_m import w4_small_m
    from kernels.model.w4_i8_to_temporary_w8 import w4_i8_to_temporary_w8
    from kernels.operators.op30_activation_quantization import activation_quantization
    from tools.quantization.w4_i8_pack import pack_array
    from tools.operators.common import configure, benchmark, error, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 3, 4, 8, 16, 17, 32, 65, 128])
    parser.add_argument("--layers", type=int, nargs="+", default=[0])
    parser.add_argument("--tile-n", type=int, default=64, choices=[64, 128])
    parser.add_argument("--tile-m", type=int, default=16, choices=[16, 32, 64])
    parser.add_argument("--split-down", type=int, default=8, choices=[1, 2, 4, 8])
    parser.add_argument("--activations", type=Path)
    parser.add_argument("--modes", nargs="+", choices=["row", "group"], default=["row", "group"])
    parser.add_argument("--dynamic-rows", action="store_true")
    parser.add_argument("--group-activation", action="store_true")
    parser.add_argument(
        "--families", nargs="+", choices=["In", "Out", "GateUp", "Down"], default=["GateUp", "Down"]
    )
    args = parser.parse_args()
    if args.group_activation and args.modes != ["group"]:
        parser.error("--group-activation requires --modes group")
    configure()
    weights = args.model / "cache/weights"
    metadata = json.loads((args.model / "cache/model.json").read_text())["metadata"]
    buffers = {b["name"]: b for b in metadata["buffers"]}
    index = json.loads((weights / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(name, device="cuda"):
        with (
            (weights / index[name]).open("rb") as f,
            mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as raw,
        ):
            size = int.from_bytes(raw[:8], "little")
            spec = json.loads(raw[8 : 8 + size])[name]
            start, end = spec["data_offsets"]
            dtype = {"I32": np.int32, "F16": np.float16, "I8": np.int8}[spec["dtype"]]
            data = (
                np.frombuffer(raw[8 + size + start : 8 + size + end], dtype=dtype)
                .copy()
                .reshape(spec["shape"])
            )
            if hashlib.sha256(data.tobytes()).hexdigest() != buffers[name]["data"]["sha256"]:
                raise ValueError("Tensor payload digest mismatch: " + name)
        value = torch.from_numpy(data)
        return value.cuda() if device == "cuda" else value

    report = dict(
        status="running",
        seed=20261002,
        cases=[],
        scope="W4 projection implementation and incremental quantization error; not BF16/FP8 model quality.",
    )
    for layer in args.layers:
        for family in args.families:
            prefix = f"L{layer}_{family}"
            pp, s, z, ws = [tensor(prefix + suffix) for suffix in ["_P", "_S", "_Z", "_WS"]]
            layout = buffers[prefix + "_P"]["layout"]
            if layout not in ("u4_warp_n64_k128_mma_i8", "u4_warp_n64_k128_mma_f16"):
                raise ValueError("Unsupported packed W4 layout: " + layout)
            weight_layout = "i8" if layout.endswith("mma_i8") else "f16"
            native = (
                pp
                if weight_layout == "i8"
                else torch.from_numpy(
                    pack_array(tensor(prefix + "_P", "cpu").numpy()).view(np.int32)
                ).cuda()
            )
            n, groups = s.shape
            k = groups * 128
            split = args.split_down if family in ("Down", "Out") else 1
            out_dtype = "float32" if split > 1 else "float16"
            w8 = torch.empty((n, k), device="cuda", dtype=torch.int8)
            expand = w4_i8_to_temporary_w8(n, k, BK=128)
            expand.adapter.func(
                native, s, z, ws, w8, stream=torch.cuda.current_stream().cuda_stream
            )
            wf = w8.float()
            # Independent inverse index mapping to logical U4, not the MMA reader.
            j = torch.arange(k, device="cuda")
            i = torch.arange(n, device="cuda")[:, None]
            lanes = (i % 64 // 16) * 32 + (i % 8) * 4 + (j % 16 // 4)
            words = (j % 128 // 32) * 2 + (i % 16 // 8)
            shifts = (j % 32 // 16) * 16 + (j % 4) * 4
            q4 = ((native[i // 64, j // 128, lanes, words].to(torch.int64) >> shifts) & 15).to(
                torch.int8
            )
            centered = q4.reshape(n, groups, 128).float() - z[:, :, None].float()
            w_group = (centered * s[:, :, None].float()).reshape(n, k)
            del centered, q4
            quant = activation_quantization(k, 128 if args.group_activation else None)
            mask = torch.zeros(k, device="cuda", dtype=torch.uint8)
            for rows in args.rows:
                if args.activations:
                    path = args.activations / f"L{layer}_{family}.f16"
                    array = np.fromfile(path, dtype=np.float16).reshape(-1, k)
                    assert len(array) > 0
                    x = torch.from_numpy(
                        np.tile(array, ((rows + len(array) - 1) // len(array), 1))[:rows].copy()
                    ).cuda()
                    origin = str(path)
                else:
                    x = torch.randn((rows, k), device="cuda", dtype=torch.float16) * 0.1
                    origin = "random implementation input"
                aq = torch.empty((rows, k), device="cuda", dtype=torch.int8)
                asc = torch.empty(
                    (rows, groups if args.group_activation else 1),
                    device="cuda",
                    dtype=torch.float16,
                )

                def quantize():
                    quant.adapter.func(
                        x, mask, aq, asc, stream=torch.cuda.current_stream().cuda_stream
                    )

                quantize()
                a32 = aq.view(torch.int32)
                ordinary = w4_small_m(
                    rows,
                    n,
                    k,
                    split,
                    out_dtype,
                    TILE_N=128 if family in ("In", "Out", "GateUp") else 64,
                    weight_layout=weight_layout,
                    byte_permute=True,
                    vector_words=4,
                )
                golden = torch.empty(
                    (split, rows, n), device="cuda", dtype=getattr(torch, out_dtype)
                )

                def baseline():
                    ordinary.adapter.func(
                        x, pp, s, z, golden, stream=torch.cuda.current_stream().cuda_stream
                    )

                base_time, _ = benchmark(baseline, repetitions=8)
                for mode in args.modes:
                    kernel = w4a8_decode(
                        None if args.dynamic_rows else rows,
                        n,
                        k,
                        split,
                        mode=mode,
                        TILE_N=args.tile_n,
                        TILE_M=args.tile_m,
                        output_dtype=out_dtype,
                        activation_group=128 if args.group_activation else None,
                        weight_layout=weight_layout,
                    )
                    guarded = torch.full(
                        (split * rows * n + 256,),
                        123.0,
                        device="cuda",
                        dtype=getattr(torch, out_dtype),
                    )
                    out = guarded[:-256].view(split, rows, n)

                    def candidate():
                        quantize()
                        kernel.adapter.func(
                            a32,
                            pp,
                            s,
                            z,
                            ws,
                            asc,
                            out,
                            stream=torch.cuda.current_stream().cuda_stream,
                        )

                    timing, graph = benchmark(candidate, repetitions=8)
                    references = []
                    for part in range(split):
                        lo, hi = part * k // split, (part + 1) * k // split
                        weight = wf[:, lo:hi] if mode == "row" else w_group[:, lo:hi]
                        if args.group_activation:
                            dequant_a = (
                                aq[:, lo:hi].reshape(rows, -1, 128).float()
                                * asc[:, lo // 128 : hi // 128, None].float()
                            ).reshape(rows, -1)
                            ref = dequant_a @ weight.T
                        else:
                            ref = aq[:, lo:hi].float() @ weight.T
                            if mode == "row":
                                ref = ref * asc.float() * ws[None, :].float()
                            else:
                                ref = ref * asc.float()
                        references.append(ref.to(getattr(torch, out_dtype)))
                    reference = torch.stack(references)
                    implementation = error(out, reference)
                    assert implementation["finite"] and implementation["relative_l2"] < 2e-5, (
                        family,
                        rows,
                        mode,
                        implementation,
                    )
                    if weight_layout == "f16":
                        native_kernel = w4a8_decode(
                            None if args.dynamic_rows else rows,
                            n,
                            k,
                            split,
                            mode=mode,
                            TILE_N=args.tile_n,
                            TILE_M=args.tile_m,
                            output_dtype=out_dtype,
                            activation_group=128 if args.group_activation else None,
                        )
                        native_out = torch.empty_like(out)
                        native_kernel.adapter.func(
                            a32,
                            native,
                            s,
                            z,
                            ws,
                            asc,
                            native_out,
                            stream=torch.cuda.current_stream().cuda_stream,
                        )
                        torch.cuda.synchronize()
                        assert torch.equal(native_out, out), (
                            "Register shuffle differs from lossless offline permutation"
                        )
                        del native_kernel, native_out
                    assert bool((guarded[-256:] == 123.0).all()), "Output tail overwritten"
                    value = out.clone()
                    saved = x.clone()
                    x.zero_()
                    out.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    assert bool((out == 0).all()), "Graph ignored changed input"
                    x.copy_(saved)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(out, value), "Graph restoration differs"
                    assert bool((guarded[-256:] == 123.0).all())
                    diagnostic = error(out.sum(0), golden.sum(0))
                    entry = dict(
                        layer=layer,
                        family=family,
                        rows=rows,
                        mode=mode,
                        weight_layout=weight_layout,
                        fragment_permutation_equal=True,
                        tile_n=args.tile_n,
                        tile_m=args.tile_m,
                        split=split,
                        activation_group=128 if args.group_activation else None,
                        dynamic_rows=args.dynamic_rows,
                        activation_origin=origin,
                        implementation=implementation,
                        w4_error=diagnostic,
                        baseline=base_time,
                        candidate=timing,
                        tail_safe=True,
                        graph_zero_restore=True,
                    )
                    report["cases"].append(entry)
                    write_json(args.output / "result.json", report)
                    print(
                        layer,
                        family,
                        rows,
                        mode,
                        "ms",
                        timing["median_ms"],
                        "relative_l2",
                        diagnostic["relative_l2"],
                        flush=True,
                    )
                    kernel = None
                    del graph
                    out = None
                    del guarded
                    del references
                    del reference
                    del value
                    del saved
                ordinary = None
                golden = None
                x = None
                aq = None
                a32 = None
                asc = None
            pp = None
            del native
            s = None
            z = None
            ws = None
            del w8
            del wf
            del w_group
            del expand
            gc.collect()
    report["status"] = "passed"
    write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
