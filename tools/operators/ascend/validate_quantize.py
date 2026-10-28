"""Check batched A2 quantization against scalar kernels and FP64 division."""

import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.build.files import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--capacity", type=int, default=2049)
    parser.add_argument("--tile-rows", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--multipliers", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--scale-types", nargs="+", choices=["f16", "f32"], default=["f16", "f32"])
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.capacity <= 4096 or any(m not in (1, 32) for m in args.multipliers):
        parser.error("Require capacity 2..4096 and multipliers 1 or 32")
    import torch
    import torch_npu  # noqa: F401
    from kernels.model.rows import row_count
    from tools.model.deepseek_v41.tune import npu_benchmark
    from tools.operators.ascend.build import LOWER_LOCK, compile_kernel, operand
    from tools.operators.ascend.vector import quantize

    torch.set_num_threads(4)
    torch.manual_seed(20261002)
    if not args.compile_only:
        torch.npu.set_device(args.device)
    width, capacity = 128, args.capacity
    configs = sorted({1, *args.tile_rows})
    reports = []
    for multiplier, static in [(m, False) for m in args.multipliers] + [(32, True)]:
        physical = 32 if static else capacity * multiplier
        for scale_type in args.scale_types:
            half = scale_type == "f16"
            specs = [
                operand("x", "bf16", [physical, width]),
                operand("out", "i8", [physical, width], "write"),
                operand("scales", scale_type, [physical], "write"),
            ]

            current_configs = [1] if static else configs

            def compile_one(tile):
                with LOWER_LOCK:
                    function = quantize(
                        physical if static else row_count(capacity, True) * multiplier,
                        width,
                        half,
                        tile,
                    )
                path = compile_kernel(
                    function,
                    "vector.quantize",
                    specs,
                    dict(rows=physical, width=width, fp16_query_scale=half, tile_rows=tile),
                    args.output / "cache",
                    row_capacity=None if static else capacity,
                )
                print(
                    dict(
                        compiled=tile,
                        multiplier=multiplier,
                        static=static,
                        scale_type=scale_type,
                        manifest=str(path),
                    ),
                    flush=True,
                )
                return path

            with ThreadPoolExecutor(max_workers=3) as pool:
                paths = list(pool.map(compile_one, current_configs))
            if args.compile_only:
                continue
            x = torch.empty(physical, width, dtype=torch.bfloat16, device="npu")
            outputs = [
                torch.empty(physical + 32, width, dtype=torch.int8, device="npu") for _ in paths
            ]
            scales = [
                torch.empty(
                    physical + 32, dtype=torch.float16 if half else torch.float32, device="npu"
                )
                for _ in paths
            ]
            libraries = [ctypes.CDLL(str(p.parent / "kernel.so")) for p in paths]
            entries, pointers = [], []
            for lib, out, scale in zip(libraries, outputs, scales, strict=True):
                entry = lib.orinfer_ascend_kernel_v1 if static else lib.orinfer_ascend_kernel_v2
                entry.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_uint64,
                    *([] if static else [ctypes.c_uint64]),
                    ctypes.c_void_p,
                ]
                entry.restype = ctypes.c_int32
                entries.append(entry)
                pointers.append(
                    (ctypes.c_void_p * 3)(x.data_ptr(), out.data_ptr(), scale.data_ptr())
                )
            random = (
                torch.randn(physical, width)
                * torch.pow(2.0, torch.randint(-20, 12, (physical, 1)).float())
            ).bfloat16()
            boundary = (torch.arange(width).float() - 63.5)[None].repeat(physical, 1)
            boundary[:, 0] = 127
            boundary *= torch.pow(2.0, torch.arange(physical).remainder(28).float() - 24)[:, None]
            boundary = boundary.bfloat16()
            measured_rows = (
                [1]
                if static
                else sorted({r for r in (1, 2, 17, 31, 32, 129, 2048, capacity) if r <= capacity})
            )
            for rows in measured_rows:
                count = rows * multiplier
                for pattern in ("random", "half_integer", "zero", "tiny", "subnormal_scale"):
                    source = (random if pattern == "random" else boundary).clone()
                    if pattern == "zero":
                        source.zero_()
                    elif pattern == "tiny":
                        source = (random.float() * 2.0**-40).bfloat16()
                    elif pattern == "subnormal_scale":
                        # The unrounded scale becomes zero in FP16; the exact
                        # 2^-24 floor must still round half-integer codes evenly.
                        source = (
                            ((torch.arange(width).remainder(16).float() - 7.5) * 2.0**-24)[None]
                            .repeat(physical, 1)
                            .bfloat16()
                        )
                    x.copy_(source)
                    for out, scale in zip(outputs, scales, strict=True):
                        out.fill_(17)
                        scale.fill_(17)

                    def run(index):
                        status = entries[index](
                            pointers[index],
                            3,
                            *([] if static else [rows]),
                            torch.npu.current_stream()._as_parameter_,
                        )
                        if status:
                            raise RuntimeError(f"Quantization candidate failed: {status}")

                    def check():
                        torch.npu.synchronize()
                        values = x[:count].cpu().double()
                        maximum = values.abs().amax(1).float().clamp_min(1e-12)
                        expected_scale = (maximum.double() / 127).float()
                        if half:
                            expected_scale = expected_scale.half().float().clamp_min(2.0**-24)
                        expected = (values / expected_scale.double()[:, None]).float()
                        expected = expected.clamp(-127, 127).round().to(torch.int8)
                        for i, (out, scale) in enumerate(zip(outputs, scales, strict=True)):
                            torch.testing.assert_close(
                                out[:count],
                                outputs[0][:count],
                                atol=0,
                                rtol=0,
                                msg=f"{pattern} m={rows} tile={current_configs[i]} legacy codes",
                            )
                            torch.testing.assert_close(
                                scale[:count],
                                scales[0][:count],
                                atol=0,
                                rtol=0,
                                msg=f"{pattern} m={rows} tile={current_configs[i]} legacy scales",
                            )
                            actual = out[:count].cpu()
                            if not torch.equal(actual, expected):
                                bad = (actual != expected).nonzero()[:8].tolist()
                                print(
                                    dict(
                                        pattern=pattern,
                                        rows=rows,
                                        tile=current_configs[i],
                                        mismatches=[
                                            dict(
                                                row=a,
                                                column=b,
                                                value=float(values[a, b]),
                                                scale=float(scale[a].cpu()),
                                                reference_scale=float(expected_scale[a]),
                                                actual=int(actual[a, b]),
                                                expected=int(expected[a, b]),
                                            )
                                            for a, b in bad
                                        ],
                                    ),
                                    flush=True,
                                )
                            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                            torch.testing.assert_close(
                                scale[:count].cpu().float(), expected_scale, atol=0, rtol=0
                            )
                            assert bool((out[count:] == 17).all()), (
                                current_configs[i],
                                "output tail",
                            )
                            assert bool((scale[count:] == 17).all()), (
                                current_configs[i],
                                "scale tail",
                            )
                            torch.testing.assert_close(
                                out[:count], outputs[0][:count], atol=0, rtol=0
                            )
                            torch.testing.assert_close(
                                scale[:count], scales[0][:count], atol=0, rtol=0
                            )

                    for i in range(len(entries)):
                        run(i)
                    check()
                    timings, graphs = [], []
                    for i in range(len(entries)):
                        timing, graph = npu_benchmark(lambda: run(i), 3, 1)
                        timings.append(timing["median_ms"])
                        graphs.append(graph)
                    x.neg_()
                    for graph in graphs:
                        graph.replay()
                    check()
                    x.mul_(1.25)
                    for graph in graphs:
                        graph.replay()
                    check()
                    report = dict(
                        rows=rows,
                        multiplier=multiplier,
                        static=static,
                        scale_type=scale_type,
                        pattern=pattern,
                        tiles=current_configs,
                        milliseconds=timings,
                        exact=True,
                    )
                    reports.append(report)
                    print(report, flush=True)
            torch.npu.synchronize()
            graphs = graph = entries = pointers = libraries = None
            x = outputs = scales = out = scale = None
            torch.npu.empty_cache()
    if not args.compile_only:
        atomic_json(args.output / "validation.json", reports)


if __name__ == "__main__":
    main()
