"""Offline checkpoint repacking and TileLang AOT full-model plan generation.

No Python/PyTorch compute is used by the resulting online Rust executable.
Initial diagnostic retains original dense embedding/head (FP16 cast); this is
explicitly above the ~4bit budget. Dense U4 is a separately evaluated candidate.
"""

import argparse
import gc
import hashlib
import importlib
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from common import configure, environment, export_kernel, write_json, ROOT
from abi import parse_host, evaluate
from tools.quantization.w4_warp_pack import LAYOUT as WARP_LAYOUT, pack_array

from tools.model.checkpoint import Checkpoint

H, F, V, C, MAXPOS, NP = 5120, 17408, 248320, 512, 8704, 68
# 8704 / 128 = 68, and capacity includes all 8192+255 model inputs.


class Builder:
    def __init__(
        self,
        out,
        reuse_aot=None,
        reuse_weights=None,
        dense_u4=False,
        prefill_w4a8=False,
        w8_expand_mode="dynamic",
        gdn_math="simt",
        prefill_tokens=512,
        prefill_grid_order="auto",
        decode_register_mma=False,
        swiglu_a8_mode="strict",
        decode_register_scope="ffn",
        decode_state_mode="immutable",
        decode_attention_mode="original",
        prefill_norm_a8=False,
        gdn_wy_mode="simt",
        prefill_attention_mode="original",
        gdn_solve_mode="registers",
        checkpoint=None,
    ):
        self.checkpoint = Path(checkpoint).resolve(strict=True)
        with Checkpoint(self.checkpoint) as source:
            source.validate_backbone()
            self.checkpoint_identity = source.identity
        self.out = out
        self.dense_u4 = dense_u4
        self.prefill_w4a8 = prefill_w4a8
        self.w8_expand_mode = w8_expand_mode
        self.gdn_math = gdn_math
        assert decode_state_mode in ("immutable", "inplace")
        assert decode_attention_mode in ("original", "staged")
        self.decode_state_mode = decode_state_mode
        self.decode_attention_mode = decode_attention_mode
        assert gdn_wy_mode in ("simt", "compensated")
        assert prefill_attention_mode in ("original", "staged64")
        self.prefill_norm_a8 = prefill_norm_a8
        self.gdn_wy_mode = gdn_wy_mode
        self.prefill_attention_mode = prefill_attention_mode
        assert gdn_solve_mode in ("registers", "columns-register")
        self.gdn_solve_mode = gdn_solve_mode
        if prefill_norm_a8 and not prefill_w4a8:
            raise ValueError("Fused norm/A8 requires W4A8 prefill")
        self.decode_register_mma = decode_register_mma
        assert decode_register_scope in ("ffn", "all")
        self.decode_register_scope = decode_register_scope
        if decode_register_scope == "all" and not (decode_register_mma and dense_u4):
            raise ValueError(
                "All-projection register MMA requires --decode-register-mma and --dense-u4"
            )
        self.warp_weights = {}
        if decode_register_mma:
            for i in range(64):
                self.warp_weights.update({f"L{i}_GateUp": (2 * F, H), f"L{i}_Down": (H, F)})
                if decode_register_scope == "all":
                    self.warp_weights.update(
                        {f"L{i}_In": (14336 if i % 4 == 3 else 16384, H), f"L{i}_Out": (H, 6144)}
                    )
            if decode_register_scope == "all":
                self.warp_weights["Head"] = (V, H)
        assert swiglu_a8_mode in ("strict", "guarded-reciprocal", "lut")
        self.swiglu_a8_mode = swiglu_a8_mode
        if decode_register_mma and not prefill_w4a8:
            raise ValueError(
                "Warp-packed decode currently requires W4A8 prefill to read the same weight copy"
            )
        assert prefill_tokens in (512, 2048, 8192)
        self.prefill_tokens = prefill_tokens
        assert prefill_grid_order in ("auto", "nfirst", "mfirst")
        # M-first large-M screening lost activation reuse and was slower.
        # Keep it explicit for comparisons; auto selects measured N-first.
        self.i8_grid_order = "nfirst" if prefill_grid_order == "auto" else prefill_grid_order
        self.reuse_aot = reuse_aot
        self.reuse_weights = reuse_weights
        self.buffers, self.kernels, self.exports, self.reset = [], [], {}, []
        self.weight_parameters = 0
        self.started = time.perf_counter()
        self.report = {
            "status": "building",
            "environment": environment(),
            "layers": [],
            "scale_casts": [],
        }
        if reuse_weights:
            self.report["scale_casts"] = json.loads((reuse_weights / "progress.json").read_text())[
                "scale_casts"
            ]
            self.reused_report = json.loads((reuse_weights / "build-report.json").read_text())
            if self.reused_report.get("checkpoint_identity") != self.checkpoint_identity:
                raise ValueError("Reused weights belong to a different checkpoint identity")
            self.reused_buffers = {
                b["name"]: b
                for b in json.loads((reuse_weights / "model.json").read_text())["buffers"]
            }
            warp_names = {
                name for name, b in self.reused_buffers.items() if b["layout"] == WARP_LAYOUT
            }
            allowed_warp = {name + "_P" for name in self.warp_weights}
            if warp_names and (not decode_register_mma or not warp_names <= allowed_warp):
                raise ValueError(
                    "Reused warp-packed weights need matching decode-register-mma support"
                )
        self.sources = []
        for src in sorted((ROOT / "kernels").rglob("*.py")) + [
            Path(__file__).resolve(),
            ROOT / "tools/operators/common.py",
            ROOT / "tools/operators/abi.py",
            ROOT / "tools/quantization/w4_warp_pack.py",
        ]:
            dst = out / "measurement-source" / src.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            self.sources.append(
                {
                    "file": str(src.relative_to(ROOT)),
                    "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                }
            )

    def identity(self, path):
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return {"file": str(path.relative_to(self.out)), "sha256": digest.hexdigest()}

    def buffer(self, name, dtype, shape, data=None, weight=False, reset=False):
        item = {
            "name": name,
            "dtype": dtype,
            "shape": list(shape),
            "layout": "contiguous",
            "alignment": 256,
            "access": "read" if weight else "read_write",
            "data": None,
        }
        if data is not None:
            path = self.out / "weights" / f"{name}.bin"
            path.parent.mkdir(exist_ok=True)
            previous = (
                self.reuse_weights / "weights" / path.name
                if self.reuse_weights and weight
                else None
            )
            if previous is not None and previous.exists():
                assert previous.stat().st_size == data.numel() * data.element_size()
                os.link(previous, path)
            else:
                data.contiguous().view(torch.uint8).numpy().tofile(path)
            item["data"] = self.identity(path)
        self.buffers.append(item)
        if reset:
            self.reset.append(name)
        return name

    def tensor(self, model, name, key, dtype="f16", shape=None):
        if (
            self.reuse_weights is not None
            and (self.reuse_weights / "weights" / f"{name}.bin").exists()
        ):
            source_shape = model.get_slice(key).get_shape()
            self.weight_parameters += int(np.prod(source_shape))
            self.buffer(name, dtype, shape or source_shape, weight=True)
            path = self.out / "weights" / f"{name}.bin"
            path.parent.mkdir(exist_ok=True)
            os.link(self.reuse_weights / "weights" / path.name, path)
            self.buffers[-1]["data"] = self.identity(path)
            return name
        tensor = model.get_tensor(key)
        self.weight_parameters += tensor.numel()
        tensor = tensor.float() if dtype == "f32" else tensor.half()
        return self.buffer(name, dtype, shape or tensor.shape, tensor, weight=True)

    def packed(self, model, name, keys):
        # Lossless physical conversion of source INT4. Bounded row slices.
        paths = {kind: self.out / "weights" / f"{name}_{kind}.bin" for kind in ("P", "S", "Z")}
        for p in paths.values():
            p.parent.mkdir(exist_ok=True)
        if self.reuse_weights is not None:
            shapes = [model.get_slice(key + ".weight_scale").get_shape() for key in keys]
            n_total = sum(shape[0] for shape in shapes)
            k = shapes[0][1] * 128
            assert all(shape[1] * 128 == k for shape in shapes)
            self.weight_parameters += n_total * k
            for kind, dtype, shape in (
                ("P", "u8", (n_total, k // 2)),
                ("S", "f16", (n_total, k // 128)),
                ("Z", "i8", (n_total, k // 128)),
            ):
                previous = self.reuse_weights / "weights" / paths[kind].name
                assert (
                    previous.stat().st_size
                    == int(np.prod(shape)) * {"u8": 1, "f16": 2, "i8": 1}[dtype]
                )
                os.link(previous, paths[kind])
                self.buffer(name + "_" + kind, dtype, shape, weight=True)
                self.buffers[-1]["data"] = self.identity(paths[kind])
            return {k: name + "_" + k for k in ("P", "S", "Z")}
        handles = {kind: p.open("wb") for kind, p in paths.items()}
        n_total, k = 0, None
        try:
            for key in keys:
                source = model.get_tensor(key + ".weight_packed")
                scales = model.get_tensor(key + ".weight_scale")
                zeros = model.get_tensor(key + ".weight_zero_point")
                n, ng = scales.shape
                this_k = ng * 128
                assert source.shape == (n, this_k // 8) and zeros.shape == (n // 8, ng)
                assert k is None or this_k == k
                k = this_k
                cast = scales.half().float()
                delta = (cast - scales.float()).abs()
                self.report["scale_casts"].append(
                    {
                        "tensor": key,
                        "changed": int((cast != scales.float()).sum()),
                        "underflows": int(((cast == 0) & (scales != 0)).sum()),
                        "max_abs": float(delta.max()),
                    }
                )
                shifts = torch.arange(8, dtype=torch.int32) * 4
                for row in range(0, n, 1024):
                    count = min(1024, n - row)
                    q = (
                        ((source[row : row + count, :, None] >> shifts) & 15)
                        .to(torch.uint8)
                        .reshape(count, k)
                    )
                    p = q[:, 0::2] | (q[:, 1::2] << 4)
                    z = (
                        (
                            (zeros[row // 8 : (row + count) // 8, None, :] >> shifts[None, :, None])
                            & 15
                        )
                        .to(torch.int8)
                        .reshape(count, ng)
                    )
                    p.numpy().tofile(handles["P"])
                    scales[row : row + count].half().numpy().tofile(handles["S"])
                    z.numpy().tofile(handles["Z"])
                n_total += n
                self.weight_parameters += n * k
                del source, scales, zeros
        finally:
            for f in handles.values():
                f.close()
        for kind, dtype, shape in (
            ("P", "u8", (n_total, k // 2)),
            ("S", "f16", (n_total, k // 128)),
            ("Z", "i8", (n_total, k // 128)),
        ):
            self.buffer(name + "_" + kind, dtype, shape, weight=True)
            self.buffers[-1]["data"] = self.identity(paths[kind])
        return {k: name + "_" + k for k in ("P", "S", "Z")}

    def dense_quantize(self, model, name, key):
        n, k = model.get_slice(key).get_shape()
        assert k % 128 == 0
        self.weight_parameters += n * k
        paths = {kind: self.out / "weights" / f"{name}_{kind}.bin" for kind in ("P", "S", "Z")}
        for p in paths.values():
            p.parent.mkdir(exist_ok=True)
        if self.reuse_weights is not None and all(
            (self.reuse_weights / "weights" / path.name).exists() for path in paths.values()
        ):
            old = self.reused_report
            quant = next(q for q in old["dense_quantization"] if q["tensor"] == key)
            self.report.setdefault("dense_quantization", []).append(quant)
            for kind, dtype, shape in (
                ("P", "u8", (n, k // 2)),
                ("S", "f16", (n, k // 128)),
                ("Z", "i8", (n, k // 128)),
            ):
                source = self.reuse_weights / "weights" / paths[kind].name
                assert (
                    source.stat().st_size
                    == int(np.prod(shape)) * {"u8": 1, "f16": 2, "i8": 1}[dtype]
                )
                os.link(source, paths[kind])
                self.buffer(name + "_" + kind, dtype, shape, weight=True)
                self.buffers[-1]["data"] = self.identity(paths[kind])
            return
        handles = {kind: path.open("wb") for kind, path in paths.items()}
        numerator = denominator = 0.0
        maximum = 0.0
        try:
            for row in range(0, n, 1024):
                original = (
                    model.get_slice(key)[row : min(row + 1024, n)]
                    .float()
                    .reshape(-1, k // 128, 128)
                )
                lo = original.amin(-1).clamp(max=0)
                hi = original.amax(-1).clamp(min=0)
                scale = ((hi - lo) / 15).clamp(min=2**-24).half()
                zero = torch.round(-lo / scale.float()).clamp(0, 15).to(torch.int8)
                codes = (
                    torch.round(original / scale.float()[:, :, None] + zero[:, :, None])
                    .clamp(0, 15)
                    .to(torch.uint8)
                    .reshape(-1, k)
                )
                packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
                actual = (
                    (
                        (codes.reshape(-1, k // 128, 128).float() - zero[:, :, None].float())
                        * scale[:, :, None].float()
                    )
                    .half()
                    .float()
                )
                diff = (actual - original).double()
                numerator += float((diff * diff).sum())
                denominator += float(original.double().square().sum())
                maximum = max(maximum, float(diff.abs().max()))
                packed.numpy().tofile(handles["P"])
                scale.numpy().tofile(handles["S"])
                zero.numpy().tofile(handles["Z"])
        finally:
            for handle in handles.values():
                handle.close()
        for kind, dtype, shape in (
            ("P", "u8", (n, k // 2)),
            ("S", "f16", (n, k // 128)),
            ("Z", "i8", (n, k // 128)),
        ):
            self.buffer(name + "_" + kind, dtype, shape, weight=True)
            self.buffers[-1]["data"] = self.identity(paths[kind])
        self.report.setdefault("dense_quantization", []).append(
            {
                "tensor": key,
                "parameters": n * k,
                "relative_l2": (numerator / denominator) ** 0.5,
                "max_abs": maximum,
                "method": "group128 asymmetric RNE, FP16 scale, numeric I8 zero, includes zero in extrema; uncalibrated candidate",
            }
        )
        print("quantized", name, flush=True)

    def w8_row_scales(self, name):
        """Offline WS from actual packed codes, bounded blocks, FP16 W math.

        The extrema of the actual codes in each group suffice because W(q) is
        monotone. No full decoded weight or resident W8 copy is constructed.
        """
        weight = next(b for b in self.buffers if b["name"] == name + "_P")
        n, khalf = weight["shape"]
        k = khalf * 2
        ng = k // 128
        if (
            self.reuse_weights is not None
            and self.reused_buffers[name + "_P"]["layout"] == WARP_LAYOUT
            and not (self.reuse_weights / "weights" / f"{name}_WS.bin").exists()
        ):
            raise ValueError(
                "Warp-packed weights require compatible logical-row WS; cannot scan them as adjacent U4"
            )
        if (
            self.reuse_weights is not None
            and (self.reuse_weights / "weights" / f"{name}_WS.bin").exists()
        ):
            old_buffers = self.reused_buffers
            for suffix in ("_P", "_S", "_Z"):
                actual = next(b for b in self.buffers if b["name"] == name + suffix)
                assert actual["data"]["sha256"] == old_buffers[name + suffix]["data"]["sha256"]
            source = self.reuse_weights / "weights" / f"{name}_WS.bin"
            assert source.stat().st_size == n * 2
            old_report = self.reused_report
            record = next(r for r in old_report["w8_row_scales"] if r["weight"] == name)
            assert record["N"] == n and record["K"] == k
            self.buffer(name + "_WS", "f16", (n,), weight=True)
            path = self.out / "weights" / source.name
            path.parent.mkdir(exist_ok=True)
            os.link(source, path)
            self.buffers[-1]["data"] = self.identity(path)
            assert self.buffers[-1]["data"]["sha256"] == old_buffers[name + "_WS"]["data"]["sha256"]
            self.report.setdefault("w8_row_scales", []).append(record)
            return
        packed = np.memmap(
            self.out / "weights" / f"{name}_P.bin", dtype=np.uint8, mode="r", shape=(n, ng, 64)
        )
        scale = np.memmap(
            self.out / "weights" / f"{name}_S.bin", dtype=np.float16, mode="r", shape=(n, ng)
        )
        zero = np.memmap(
            self.out / "weights" / f"{name}_Z.bin", dtype=np.int8, mode="r", shape=(n, ng)
        )
        ws = np.empty(n, dtype=np.float16)
        for row in range(0, n, 256):
            p = packed[row : row + 256]
            lo = p & 15
            hi = p >> 4
            minimum = np.minimum(lo.min(-1), hi.min(-1)).astype(np.int16) - zero[row : row + 256]
            maximum = np.maximum(lo.max(-1), hi.max(-1)).astype(np.int16) - zero[row : row + 256]
            low = (minimum.astype(np.float16) * scale[row : row + 256]).astype(np.float16)
            high = (maximum.astype(np.float16) * scale[row : row + 256]).astype(np.float16)
            amax = np.maximum(np.abs(low), np.abs(high)).max(-1).astype(np.float32)
            ws[row : row + 256] = np.where(
                amax > 0, np.maximum(amax / np.float32(127), np.float32(2**-24)), np.float32(1)
            ).astype(np.float16)
        assert np.isfinite(ws).all() and (ws > 0).all()
        self.buffer(name + "_WS", "f16", (n,), torch.from_numpy(ws), weight=True)
        self.report.setdefault("w8_row_scales", []).append(
            {
                "weight": name,
                "N": n,
                "K": k,
                "bytes": n * 2,
                "minimum": float(ws.min()),
                "maximum": float(ws.max()),
                "method": "actual-code group extrema, W=half((q-zero)*scale); WS=half(max(abs(W))/127), floor2^-24, zero-row1",
            }
        )

    def repack_warp_weights(self):
        """Replace only this output's hardlink with one lossless physical copy.

        WS is prepared/reused before this conversion and retains its logical
        row meaning. Never modify the reused artifact's inode in place.
        """
        for base, (n, k) in self.warp_weights.items():
            name = base + "_P"
            item = next(b for b in self.buffers if b["name"] == name)
            old = self.reused_buffers.get(name) if self.reuse_weights else None
            shape = [n // 64, k // 128, 128, 8]
            if old and old["layout"] == WARP_LAYOUT:
                assert old["dtype"] == "i32" and old["shape"] == shape
                item.update(dtype="i32", shape=shape, layout=WARP_LAYOUT)
                self.report.setdefault("warp_packing", []).append(
                    dict(
                        weight=name,
                        N=n,
                        K=k,
                        reused=True,
                        source_sha256=item["data"]["sha256"],
                        packed_sha256=item["data"]["sha256"],
                        bytes=n * k // 2,
                    )
                )
                continue
            path = self.out / item["data"]["file"]
            source_sha = item["data"]["sha256"]
            source = np.memmap(path, dtype=np.uint8, mode="r", shape=(n, k // 2))
            temp = path.with_suffix(".warp.tmp")
            with temp.open("xb") as handle:
                for row in range(0, n, 256):
                    packed = pack_array(source[row : row + 256], verify=True)
                    packed.tofile(handle)
            del source
            assert temp.stat().st_size == path.stat().st_size == n * k // 2
            os.replace(temp, path)
            item.update(dtype="i32", shape=shape, layout=WARP_LAYOUT, data=self.identity(path))
            self.report.setdefault("warp_packing", []).append(
                dict(
                    weight=name,
                    N=n,
                    K=k,
                    reused=False,
                    source_sha256=source_sha,
                    packed_sha256=item["data"]["sha256"],
                    bytes=n * k // 2,
                    full_roundtrip=True,
                )
            )
            print("warp packed", base, flush=True)
            write_json(self.out / "progress.json", self.report)

    def compile(self, name, factory):
        # Fixed-M exports must not reuse a cubin compiled for another chunk
        # size. Dynamic/shared shape exports keep their existing identities.
        fixed_prefill = name.startswith(("prefill_", "i8_", "attentionprefill"))
        qualified = (
            f"{name}_m{self.prefill_tokens}"
            if fixed_prefill and self.prefill_tokens != 512
            else name
        )
        if name.startswith("i8_") and self.i8_grid_order != (
            "mfirst" if self.prefill_tokens > 512 else "nfirst"
        ):
            qualified += "_" + self.i8_grid_order
        dest = self.out / "aot" / qualified
        print("compile", qualified, flush=True)
        if (
            self.reuse_aot is not None
            and not (self.dense_u4 and name in ("embedding", "head", "topk"))
            and (self.reuse_aot / "aot" / qualified / "kernel.cubin").exists()
        ):
            shutil.copytree(self.reuse_aot / "aot" / qualified, dest)
        else:
            kernel = factory()
            export_kernel(kernel, dest)
        abi = parse_host((dest / "host.txt").read_text())
        assert len(abi) == 1, (name, abi)
        self.exports[name] = {
            **abi[0],
            "module": self.identity(dest / "kernel.cubin"),
            "source": self.identity(dest / "kernel.cu"),
            "host_abi": self.identity(dest / "host.txt"),
        }

    def op(self, program, export, pointers, scalars):
        abi = self.exports[export]
        launch = abi["launch_expressions"]
        args = []
        for arg in abi["ordered_arguments"]:
            value, ctype = arg["value"], arg["ctype"]
            if ctype in ("ctypes.c_void_p", "c_void_p"):
                assert value.endswith(".data_ptr()"), (export, value)
                value = value.removesuffix(".data_ptr()")
                assert value in pointers, (export, value, pointers)
                args.append({"kind": "buffer", "name": pointers[value]})
            else:
                types = {
                    "ctypes.c_int": "i32",
                    "ctypes.c_int32": "i32",
                    "ctypes.c_uint32": "u32",
                    "ctypes.c_int64": "i64",
                    "ctypes.c_uint64": "u64",
                    "ctypes.c_float": "f32",
                }
                assert ctype in types, (export, arg)
                args.append({"kind": types[ctype], "value": evaluate(value, scalars)})
        name = f"{export}_{len(self.kernels)}"
        k = {k: abi[k] for k in ("module", "source", "host_abi", "symbol")}
        k.update(
            name=name,
            grid=[evaluate(launch["gridDim" + a], scalars) for a in "XYZ"],
            block=[evaluate(launch["blockDim" + a], scalars) for a in "XYZ"],
            shared_memory_bytes=evaluate(launch["sharedMemBytes"], scalars),
            cooperative=False,
            args=args,
        )
        self.kernels.append(k)
        program.append({"kind": "kernel", "name": name})

    @staticmethod
    def copy(program, source, destination, n):
        program.append({"kind": "copy", "source": source, "destination": destination, "bytes": n})

    def compile_all(self):
        C = self.prefill_tokens

        def op(n):
            return importlib.import_module("kernels.operators.op" + n)

        from kernels.model import control
        from kernels.model.w4_decode_register_mma import w4_decode_register_mma
        from kernels.projections.candidates import fp16_gemm

        for m, phase in ((C, "prefill"), (1, "decode")):
            self.compile(phase + "_prepare", lambda m=m: control.prepare(m))
            self.compile(phase + "_advance", lambda m=m: control.advance(m))
            if m > 1 and self.prefill_norm_a8:
                from kernels.model.residual_norm_a8 import residual_norm_a8

                self.compile("norm_a8_fused256", lambda: residual_norm_a8(threads=256))
                self.exports["prefill_norm"] = self.exports["norm_a8_fused256"]
            else:
                self.compile(
                    phase + "_norm",
                    lambda m=m: op("02_residual_norm").residual_norm(
                        m, residual_dtype="float32", output_residual_dtype="float32"
                    ),
                )
            if not (m > 1 and self.prefill_w4a8):
                if m == 1 and self.decode_register_mma:
                    self.compile(
                        "decode_gateup_regmma", lambda: w4_decode_register_mma(2 * F, H, TILE_N=128)
                    )
                    self.exports["decode_gateup"] = self.exports["decode_gateup_regmma"]
                else:
                    self.compile(
                        phase + "_gateup",
                        lambda m=m: op("03_ffn_gate_up").ffn_gate_up(
                            m, implementation="register", BM=64 if m > 1 else 16
                        ),
                    )
                if m == 1 and self.decode_register_mma and self.decode_register_scope == "all":
                    self.compile(
                        "decode_qkvz_regmma",
                        lambda: w4_decode_register_mma(16384, H, TILE_N=128, output_layout="qkvz"),
                    )
                    self.exports["decode_qkvz"] = self.exports["decode_qkvz_regmma"]
                else:
                    self.compile(
                        phase + "_qkvz",
                        lambda m=m: op("06_gdn_qkvz").gdn_qkvz(
                            m, implementation="register", BM=64 if m > 1 else 16
                        ),
                    )
            self.compile(
                phase + "_ab",
                lambda m=m: op("07_gdn_ab").gdn_ab_tensorcore(
                    m, BM=32 if m > 1 else 16, output_dtype="float16"
                ),
            )
            self.compile(
                phase + "_conv",
                lambda m=m: op("08_gdn_conv_prep").gdn_conv_prep(
                    B=1, tokens=m, tile_tokens=16 if m > 1 else 1
                ),
            )
            if not (m > 1 and self.prefill_w4a8):
                if m == 1 and self.decode_register_mma and self.decode_register_scope == "all":
                    self.compile(
                        "decode_fullproj_regmma",
                        lambda: w4_decode_register_mma(14336, H, TILE_N=128),
                    )
                    self.exports["decode_fullproj"] = self.exports["decode_fullproj_regmma"]
                    self.compile(
                        "decode_outproj_regmma",
                        lambda: w4_decode_register_mma(H, 6144, 8, "float32", TILE_N=128),
                    )
                    self.exports["decode_outproj"] = self.exports["decode_outproj_regmma"]
                else:
                    self.compile(
                        phase + "_fullproj",
                        lambda m=m: op("19_full_qgatekv").full_qgatekv(
                            m, implementation="register", BM=64 if m > 1 else 16
                        ),
                    )
                    self.compile(
                        phase + "_outproj",
                        lambda m=m: (
                            op("18_mixer_out").mixer_out_full(m)
                            if m > 1
                            else op("18_mixer_out").mixer_out_partial(m)
                        ),
                    )
                if m == 1:
                    # Separate export identity avoids accidentally reusing the
                    # older shared-dequant cubin under --reuse-aot.
                    if self.decode_register_mma:
                        self.compile(
                            "decode_down_regmma",
                            lambda: w4_decode_register_mma(H, F, 8, "float32", TILE_N=64),
                        )
                        self.exports["decode_down"] = self.exports["decode_down_regmma"]
                    else:
                        self.compile(
                            "decode_down_register",
                            lambda: op("05_ffn_down").ffn_down_partial(
                                1, implementation="register"
                            ),
                        )
                        self.exports["decode_down"] = self.exports["decode_down_register"]
                else:
                    self.compile(phase + "_down", lambda m=m: op("05_ffn_down").ffn_down_full(m))
        if self.prefill_w4a8:
            from kernels.projections.candidates import int8_gemm
            from kernels.model.w4a8 import gdn_qkvz_int8

            if self.w8_expand_mode == "aligned":
                for n, k in ((16384, H), (14336, H), (H, 6144), (2 * F, H), (H, F)):
                    self.compile(
                        f"w8_expand_{n}_{k}",
                        lambda n=n, k=k: op("29_w4_to_temporary_w8").w4_to_temporary_w8_aligned(
                            n, k, BK=512 if k == H else 256
                        ),
                    )
            else:
                self.compile("w8_expand", lambda: op("29_w4_to_temporary_w8").w4_to_temporary_w8())
            if self.decode_register_mma:
                for n, k in sorted(
                    set(shape for name, shape in self.warp_weights.items() if name != "Head")
                ):
                    self.compile(
                        f"w8_expand_warp_{n}_{k}",
                        lambda n=n, k=k: op("29_w4_to_temporary_w8").w4_warp_to_temporary_w8(
                            n, k, BK=512 if k == H else 256
                        ),
                    )
            for k in (H, 6144):
                self.compile(
                    "a8_" + str(k),
                    lambda k=k: op("30_activation_quantization").activation_quantization(k),
                )
            if self.swiglu_a8_mode == "lut":
                from kernels.model.swiglu_lut_a8 import swiglu_lut_activation_quantization

                self.compile(
                    "swiglu_a8_lut512", lambda: swiglu_lut_activation_quantization(F, threads=512)
                )
                self.exports["swiglu_a8"] = self.exports["swiglu_a8_lut512"]
            elif self.swiglu_a8_mode == "guarded-reciprocal":
                from kernels.model.a8_reciprocal import activation_quantization_reciprocal

                self.compile(
                    "swiglu_a8_guarded_recip512",
                    lambda: activation_quantization_reciprocal(F, fused=True, threads=512),
                )
                self.exports["swiglu_a8"] = self.exports["swiglu_a8_guarded_recip512"]
            else:
                self.compile(
                    "swiglu_a8",
                    lambda: op("30_activation_quantization").swiglu_activation_quantization(F),
                )
            order = self.i8_grid_order
            self.compile("i8_qkvz", lambda: gdn_qkvz_int8(C, grid_order=order))
            for name, n, k in (
                ("gateup", 2 * F, H),
                ("down", H, F),
                ("fullproj", 14336, H),
                ("outproj", H, 6144),
            ):
                self.compile(
                    "i8_" + name,
                    lambda n=n, k=k: int8_gemm(C, n, k, 256, 128, 128, 2, 256, grid_order=order),
                )
        self.compile(
            "embedding",
            lambda: (
                op("01_embedding").embedding_u4()
                if self.dense_u4
                else op("01_embedding").embedding_gather()
            ),
        )
        self.compile("swiglu", lambda: op("04_swiglu").swiglu())
        self.compile(
            "gates", lambda: op("09_gdn_gates").gdn_gates(packed_ab=True, beta_round_fp16=True)
        )
        self.compile("gatednorm", lambda: op("17_gdn_gated_norm").gdn_gated_norm())
        self.compile("cumsum", lambda: op("11_gdn_chunk_cumsum").gdn_pack_chunk_cumsum())
        self.compile(
            "matrices", lambda: op("12_gdn_chunk_matrices").gdn_chunk_matrices(q_scale=128**-0.5)
        )
        if self.gdn_solve_mode == "columns-register":
            from kernels.model.gdn_solve_columns_register import gdn_chunk_solve_columns_register

            self.compile("solve_columns_register", lambda: gdn_chunk_solve_columns_register())
            self.exports["solve"] = self.exports["solve_columns_register"]
        else:
            self.compile("solve", lambda: op("13_gdn_chunk_solve").gdn_chunk_solve())
        if self.gdn_wy_mode == "compensated":
            from kernels.model.gdn_compensated import gdn_chunk_wy_compensated

            self.compile("wy_compensated32", lambda: gdn_chunk_wy_compensated(value_tile=32))
            self.exports["wy"] = self.exports["wy_compensated32"]
        else:
            self.compile("wy", lambda: op("14_gdn_chunk_wy").gdn_chunk_wy())
        if self.gdn_math == "compensated":
            from kernels.model.gdn_compensated import (
                gdn_chunk_state_compensated,
                gdn_chunk_output_compensated,
            )

            self.compile("state_compensated", lambda: gdn_chunk_state_compensated())
            self.compile("output_compensated", lambda: gdn_chunk_output_compensated())
            self.exports["state"] = self.exports["state_compensated"]
            self.exports["output"] = self.exports["output_compensated"]
        elif self.gdn_math == "factored":
            from kernels.model.gdn_factored import (
                gdn_chunk_state_factored,
                gdn_chunk_output_factored,
            )

            self.compile("state_factored", lambda: gdn_chunk_state_factored())
            self.compile("output_factored", lambda: gdn_chunk_output_factored())
            self.exports["state"] = self.exports["state_factored"]
            self.exports["output"] = self.exports["output_factored"]
        else:
            self.compile("state", lambda: op("15_gdn_chunk_state").gdn_chunk_state(value_tile=32))
            self.compile(
                "output",
                lambda: op("16_gdn_chunk_output").gdn_chunk_output(
                    q_scale=128**-0.5, token_tile=8, value_tile=16, output_layout="tokenmajor"
                ),
            )
        if self.decode_state_mode == "inplace":
            from kernels.model.gdn_recurrent_inplace import gdn_recurrent_inplace

            self.compile(
                "recurrent_inplace", lambda: gdn_recurrent_inplace(q_scale=128**-0.5, value_tile=32)
            )
            self.exports["recurrent"] = self.exports["recurrent_inplace"]
        else:
            self.compile(
                "recurrent",
                lambda: op("10_gdn_recurrent").gdn_recurrent(q_scale=128**-0.5, value_tile=32),
            )
        self.compile("splitmerge", lambda: op("32_split_k_merge").split_k_merge(1))
        self.compile(
            "fullprepare",
            lambda: op("20_full_prepare").full_prepare(1, NP, NP, max_position=MAXPOS),
        )
        self.compile("gather", lambda: op("27_state_lifecycle").paged_kv_gather())
        if self.prefill_attention_mode == "staged64":
            from kernels.model.attention_prefill_staged import attention_prefill_staged

            bm = 64 if C >= 2048 else 32
            name = f"attentionprefill_staged{bm}x32"
            self.compile(
                name,
                lambda: attention_prefill_staged(1, C, MAXPOS, kv_layout="token_major", block_m=bm),
            )
            self.exports["attentionprefill"] = self.exports[name]
        else:
            self.compile(
                "attentionprefill",
                lambda: op("21_attention_prefill").attention_prefill(
                    1, C, MAXPOS, kv_layout="token_major"
                ),
            )
        if self.decode_attention_mode == "staged":
            from kernels.model.attention_decode_staged import paged_attention_partials_gqa_staged

            self.compile(
                "attentiondecode_staged",
                lambda: paged_attention_partials_gqa_staged(NP, NP, nsplits=8),
            )
            self.exports["attentiondecode"] = self.exports["attentiondecode_staged"]
        else:
            self.compile(
                "attentiondecode",
                lambda: op("22_attention_decode").paged_attention_partials_gqa(NP, NP, nsplits=8),
            )
        self.compile(
            "attentionmerge", lambda: op("28_attention_split_merge").attention_split_merge(splits=8)
        )
        self.compile("finalnorm", lambda: op("23_final_norm").final_norm())
        if self.decode_register_mma and self.decode_register_scope == "all":
            self.compile(
                "head_regmma",
                lambda: w4_decode_register_mma(V, H, output_dtype="float32", TILE_N=128),
            )
            self.exports["head"] = self.exports["head_regmma"]
        else:
            self.compile(
                "head",
                lambda: (
                    op("24_lm_head").lm_head(1)
                    if self.dense_u4
                    else fp16_gemm(1, V, H, BM=16, BN=64, BK=64)
                ),
            )
        self.compile(
            "topk",
            lambda: op("25_token_selection").topk_partials(
                k=1, dtype="float32" if self.dense_u4 else "float16"
            ),
        )
        self.compile("topkmerge", lambda: op("25_token_selection").topk_merge(k=1))

    def weights(self):
        if self.prefill_w4a8 and self.swiglu_a8_mode == "lut":
            from kernels.model.swiglu_lut_a8 import initialize_swish_lut

            started = time.perf_counter()
            codes = torch.arange(65536, device="cuda", dtype=torch.int32)
            lut = torch.empty(65536, device="cuda", dtype=torch.float32)
            generator = initialize_swish_lut()
            generator.adapter.func(codes, lut, stream=torch.cuda.current_stream().cuda_stream)
            data = lut.cpu()
            digest = hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()
            if self.reuse_weights and "SwiGLULUT" in self.reused_buffers:
                assert self.reused_buffers["SwiGLULUT"]["data"]["sha256"] == digest, (
                    "Reused SwiGLU table changed"
                )
            self.buffer("SwiGLULUT", "f32", data.shape, data, weight=True)
            self.report["swiglu_lut"] = dict(
                bytes=data.numel() * data.element_size(),
                sha256=digest,
                generation_s=time.perf_counter() - started,
                scope="Offline table generation, not native model load",
            )
            del codes, lut, data, generator
        with Checkpoint(self.checkpoint) as model:
            if self.dense_u4:
                self.dense_quantize(model, "Embedding", "model.language_model.embed_tokens.weight")
                self.dense_quantize(model, "Head", "lm_head.weight")
            else:
                self.tensor(model, "Embedding", "model.language_model.embed_tokens.weight")
                self.tensor(model, "Head", "lm_head.weight")
            self.tensor(model, "FinalWeight", "model.language_model.norm.weight")
            for i in range(64):
                p = f"model.language_model.layers.{i}."
                l = f"L{i}_"
                self.tensor(model, l + "PreWeight", p + "input_layernorm.weight")
                self.tensor(model, l + "PostWeight", p + "post_attention_layernorm.weight")
                self.packed(model, l + "GateUp", [p + "mlp.gate_proj", p + "mlp.up_proj"])
                self.packed(model, l + "Down", [p + "mlp.down_proj"])
                if i % 4 != 3:
                    a = p + "linear_attn."
                    self.packed(model, l + "In", [a + "in_proj_qkv", a + "in_proj_z"])
                    self.packed(model, l + "Out", [a + "out_proj"])
                    ab = torch.cat(
                        [
                            model.get_tensor(a + "in_proj_a.weight"),
                            model.get_tensor(a + "in_proj_b.weight"),
                        ]
                    ).half()
                    self.weight_parameters += ab.numel()
                    self.buffer(l + "ABWeight", "f16", ab.shape, ab, weight=True)
                    self.tensor(model, l + "ConvWeight", a + "conv1d.weight", shape=(10240, 4))
                    self.tensor(model, l + "Al", a + "A_log", dtype="f32")
                    self.tensor(model, l + "Dt", a + "dt_bias", dtype="f32")
                    self.tensor(model, l + "GatedWeight", a + "norm.weight")
                    self.buffer(l + "History", "f16", (1, 3, 10240), reset=True)
                    self.buffer(l + "State", "f32", (1, 48, 128, 128), reset=True)
                else:
                    a = p + "self_attn."
                    self.packed(model, l + "In", [a + "q_proj", a + "k_proj", a + "v_proj"])
                    self.packed(model, l + "Out", [a + "o_proj"])
                    self.tensor(model, l + "QWeight", a + "q_norm.weight")
                    self.tensor(model, l + "KWeight", a + "k_norm.weight")
                    self.buffer(l + "KPages", "f16", (NP, 128, 4, 256), reset=True)
                    self.buffer(l + "VPages", "f16", (NP, 128, 4, 256), reset=True)
                if self.prefill_w4a8:
                    for name in ("GateUp", "Down", "In", "Out"):
                        self.w8_row_scales(l + name)
                print("packed layer", i, flush=True)
                self.report["layers"].append(i)
                write_json(self.out / "progress.json", self.report)
                gc.collect()

    def workspace(self):
        C = self.prefill_tokens
        specs = {
            "Input": ("i32", (C,)),
            "Hidden": ("f16", (C, H)),
            "R0": ("f32", (C, H)),
            "R1": ("f32", (C, H)),
            "Norm": ("f16", (C, H)),
            "Mix": ("f16", (C, H)),
            "GateUp": ("f16", (C, 2 * F)),
            "Activated": ("f16", (C, F)),
            "QKV": ("f16", (C, 10240)),
            "Zout": ("f16", (C, 6144)),
            "AB": ("f16", (C, 96)),
            "g": ("f32", (C, 48)),
            "Beta": ("f32", (C, 48)),
            "Q": ("f16", (16, C, 128)),
            "K": ("f16", (16, C, 128)),
            "V": ("f16", (48, C, 128)),
            "Ho": ("f16", (1, 3, 10240)),
            "Po": ("i32", (1,)),
            "Gcum": ("f32", (48, C // 64, 64)),
            "Bpad": ("f32", (48, C // 64, 64)),
            "System": ("f32", (48, C // 64, 64, 64)),
            "QK": ("f32", (48, C // 64, 64, 64)),
            "Transform": ("f32", (48, C // 64, 64, 64)),
            "W": ("f32", (48, C // 64, 64, 128)),
            "U": ("f32", (48, C // 64, 64, 128)),
            "Rchunk": ("f32", (48, C // 64, 64, 128)),
            "Senter": ("f32", (48, C // 64, 128, 128)),
            "Sout": ("f32", (48, 128, 128)),
            "Y": ("f16", (C, 6144)),
            "MixerIn": ("f16", (C, 6144)),
            "Partial": ("f32", (8, 1, H)),
            "FullX": ("f16", (C, 14336)),
            "FullQ": ("f16", (C, 24, 256)),
            "FullGate": ("f16", (C, 24, 256)),
            "Kcontig": ("f16", (MAXPOS, 4, 256)),
            "Vcontig": ("f16", (MAXPOS, 4, 256)),
            "AttM": ("f32", (24, 8)),
            "AttL": ("f32", (24, 8)),
            "AttO": ("f32", (24, 8, 256)),
            "LastHidden": ("f16", (1, H)),
            "Logits": ("f32" if self.dense_u4 else "f16", (1, V)),
            "PV": ("f32", (1, (V + 4095) // 4096, 1)),
            "PI": ("i32", (1, (V + 4095) // 4096, 1)),
            "Bad": ("i32", (1, (V + 4095) // 4096)),
            "Values": ("f32", (1, 1)),
            "IDs": ("i32", (1, 1)),
            "Token": ("i32", (1,)),
            "TokenStatus": ("i32", (1,)),
            "Positions": ("i32", (C,)),
            "SeqLength": ("i32", (1,)),
            "Step": ("i32", (1,)),
            "PrepareStatus": ("i32", (1,)),
        }
        for name, (dtype, shape) in specs.items():
            self.buffer(name, dtype, shape, reset=name in ("Step", "PrepareStatus"))
        if self.prefill_w4a8:
            self.buffer("TemporaryW8", "i8", (2 * F * H,))
            self.buffer("TemporaryA8", "i8", (C * F,))
            self.buffer("TemporaryAS", "f16", (C,))
            self.buffer("UnusedA8Mask", "u8", (F,))
        for name, value in (
            ("Req", torch.zeros(C, dtype=torch.int32)),
            ("Pages", torch.arange(NP, dtype=torch.int32).reshape(1, NP)),
            ("LengthPrefill", torch.tensor([C], dtype=torch.int32)),
            ("LengthDecode", torch.tensor([1], dtype=torch.int32)),
            ("IndexPrefill", torch.tensor([C - 1], dtype=torch.int32)),
            ("IndexDecode", torch.tensor([0], dtype=torch.int32)),
        ):
            self.buffer(name, "i32", value.shape, value)
        freq = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        angles = torch.arange(MAXPOS, dtype=torch.float32)[:, None] * freq[None, :]
        rotary = torch.cat((angles.cos(), angles.sin()), dim=-1).half()
        self.buffer("Rotary", "f16", rotary.shape, rotary)

    def program(self, phase, m):
        p = []
        d = {
            "M": m,
            "rows": m,
            "batch": 1,
            "tokens": m,
            "chunks": m // 64,
            "pages": NP,
            "table_width": NP,
        }

        def emit(export, **bind):
            self.op(p, export, bind, d)

        use_i8 = m > 1 and self.prefill_w4a8

        def projection(
            export, weight, input_buffer, n, k, output, fused_swiglu=False, already_a8=False
        ):
            # A single preallocated W8/A8 workspace is reused on this stream.
            aq = "swiglu_a8" if fused_swiglu else "a8_" + str(k)
            if not already_a8:
                quant_bind = dict(
                    X=input_buffer, Mask="UnusedA8Mask", Q="TemporaryA8", S="TemporaryAS"
                )
                if fused_swiglu and self.swiglu_a8_mode == "lut":
                    quant_bind["LUT"] = "SwiGLULUT"
                emit(aq, **quant_bind)
            expansion = f"w8_expand_{n}_{k}" if self.w8_expand_mode == "aligned" else "w8_expand"
            if weight in self.warp_weights:
                expansion = f"w8_expand_warp_{n}_{k}"
            self.op(
                p,
                expansion,
                dict(
                    P=weight + "_P",
                    PP=weight + "_P",
                    S=weight + "_S",
                    Z=weight + "_Z",
                    WS=weight + "_WS",
                    W8="TemporaryW8",
                ),
                dict(d, N=n, K=k),
            )
            bind = dict(A="TemporaryA8", B="TemporaryW8", AS="TemporaryAS", BS=weight + "_WS")
            bind.update(output)
            emit("i8_" + export, **bind)

        if m == 1:
            self.copy(p, "Token", "Input", 4)
        emit(phase + "_prepare", Step="Step", Positions="Positions", SeqLength="SeqLength")
        if self.dense_u4:
            emit(
                "embedding",
                P="Embedding_P",
                S="Embedding_S",
                Z="Embedding_Z",
                I="Input",
                Y="Hidden",
            )
        else:
            emit("embedding", W="Embedding", I="Input", Y="Hidden")
        p.append({"kind": "zero", "destination": "R0", "bytes": m * H * 4})
        for i in range(64):
            l = f"L{i}_"

            def w(n):
                return l + n

            emit(
                phase + "_norm",
                X="Hidden",
                R="R0",
                W=w("PreWeight"),
                Y="Norm",
                RO="R1",
                Q="TemporaryA8",
                S="TemporaryAS",
            )
            if i % 4 != 3:
                if use_i8:
                    projection(
                        "qkvz",
                        w("In"),
                        "Norm",
                        16384,
                        H,
                        dict(QKV="QKV", ZOUT="Zout"),
                        already_a8=self.prefill_norm_a8,
                    )
                else:
                    emit(
                        phase + "_qkvz",
                        A="Norm",
                        P=w("In_P"),
                        PP=w("In_P"),
                        S=w("In_S"),
                        Z=w("In_Z"),
                        QKV="QKV",
                        ZOUT="Zout",
                    )
                emit(phase + "_ab", X="Norm", W_ab=w("ABWeight"), Y="AB")
                emit("gates", A="AB", B="AB", Parameter=w("Al"), DtBias=w("Dt"), G="g", Beta="Beta")
                emit(
                    phase + "_conv",
                    X="QKV",
                    W=w("ConvWeight"),
                    HI=w("History"),
                    lengths="Length" + phase.title(),
                    positions="Step",
                    Q="Q",
                    K="K",
                    V="V",
                    HO="Ho",
                    positions_out="Po",
                )
                self.copy(p, "Ho", w("History"), 3 * 10240 * 2)
                if m > 1:
                    emit(
                        "cumsum",
                        G="g",
                        Beta="Beta",
                        Lengths="LengthPrefill",
                        CumulativeG="Gcum",
                        PaddedBeta="Bpad",
                    )
                    emit("matrices", Q="Q", K="K", G="Gcum", Beta="Bpad", L="System", QK="QK")
                    emit("solve", L="System", A="Transform")
                    emit("wy", A="Transform", K="K", V="V", G="Gcum", Beta="Bpad", W="W", U="U")
                    emit(
                        "state",
                        K="K",
                        G="Gcum",
                        W="W",
                        U="U",
                        Sin=w("State"),
                        Senter="Senter",
                        R="Rchunk",
                        Sfinal="Sout",
                    )
                    emit("output", Q="Q", G="Gcum", QK="QK", Senter="Senter", R="Rchunk", Y="Y")
                    self.copy(p, "Sout", w("State"), 48 * 128 * 128 * 4)
                elif self.decode_state_mode == "inplace":
                    emit(
                        "recurrent",
                        Q="Q",
                        K="K",
                        V="V",
                        G="g",
                        Beta="Beta",
                        State=w("State"),
                        Out="Y",
                    )
                else:
                    emit(
                        "recurrent",
                        Q="Q",
                        K="K",
                        V="V",
                        G="g",
                        Beta="Beta",
                        StateIn=w("State"),
                        StateOut="Sout",
                        Out="Y",
                    )
                    self.copy(p, "Sout", w("State"), 48 * 128 * 128 * 4)
                emit("gatednorm", X="Y", Z="Zout", W=w("GatedWeight"), Y="MixerIn")
            else:
                if use_i8:
                    projection(
                        "fullproj",
                        w("In"),
                        "Norm",
                        14336,
                        H,
                        dict(C="FullX"),
                        already_a8=self.prefill_norm_a8,
                    )
                else:
                    emit(
                        phase + "_fullproj",
                        A="Norm",
                        P=w("In_P"),
                        PP=w("In_P"),
                        S=w("In_S"),
                        Z=w("In_Z"),
                        C="FullX",
                        O="FullX",
                    )
                emit(
                    "fullprepare",
                    X="FullX",
                    WQ=w("QWeight"),
                    WK=w("KWeight"),
                    Cache="Rotary",
                    Req="Req",
                    Pos="Positions",
                    Pages="Pages",
                    Status="PrepareStatus",
                    Q="FullQ",
                    Gate="FullGate",
                    K=w("KPages"),
                    V=w("VPages"),
                )
                if m > 1:
                    gd = dict(d, tokens=MAXPOS)
                    self.op(
                        p,
                        "gather",
                        dict(
                            Kpages=w("KPages"),
                            Vpages=w("VPages"),
                            PageTable="Pages",
                            SeqLengths="SeqLength",
                            Kout="Kcontig",
                            Vout="Vcontig",
                        ),
                        gd,
                    )
                    emit(
                        "attentionprefill",
                        Q="FullQ",
                        K="Kcontig",
                        V="Vcontig",
                        Gate="FullGate",
                        Positions="Positions",
                        Lengths="SeqLength",
                        Y="MixerIn",
                    )
                else:
                    emit(
                        "attentiondecode",
                        Q="FullQ",
                        K=w("KPages"),
                        V=w("VPages"),
                        Pages="Pages",
                        SeqLen="SeqLength",
                        QueryPos="Positions",
                        M="AttM",
                        L="AttL",
                        O="AttO",
                    )
                    emit(
                        "attentionmerge",
                        M="AttM",
                        L="AttL",
                        O="AttO",
                        RawGate="FullGate",
                        Y="MixerIn",
                    )
            bind = dict(A="MixerIn", P=w("Out_P"), PP=w("Out_P"), S=w("Out_S"), Z=w("Out_Z"))
            bind["C" if m > 1 else "O"] = "Mix" if m > 1 else "Partial"
            if use_i8:
                projection("outproj", w("Out"), "MixerIn", H, 6144, dict(C="Mix"))
            else:
                self.op(p, phase + "_outproj", bind, d)
            if m == 1:
                emit("splitmerge", P="Partial", O="Mix")
            emit(
                phase + "_norm",
                X="Mix",
                R="R1",
                W=w("PostWeight"),
                Y="Norm",
                RO="R0",
                Q="TemporaryA8",
                S="TemporaryAS",
            )
            if use_i8:
                projection(
                    "gateup",
                    w("GateUp"),
                    "Norm",
                    2 * F,
                    H,
                    dict(C="GateUp"),
                    already_a8=self.prefill_norm_a8,
                )
            else:
                emit(
                    phase + "_gateup",
                    A="Norm",
                    P=w("GateUp_P"),
                    PP=w("GateUp_P"),
                    S=w("GateUp_S"),
                    Z=w("GateUp_Z"),
                    C="GateUp",
                    O="GateUp",
                )
                emit("swiglu", X="GateUp", Y="Activated")
            bind = dict(A="Activated", P=w("Down_P"), PP=w("Down_P"), S=w("Down_S"), Z=w("Down_Z"))
            bind["C" if m > 1 else "O"] = "Hidden" if m > 1 else "Partial"
            if use_i8:
                projection("down", w("Down"), "GateUp", H, F, dict(C="Hidden"), fused_swiglu=True)
            else:
                self.op(p, phase + "_down", bind, d)
            if m == 1:
                emit("splitmerge", P="Partial", O="Hidden")
        emit(phase + "_advance", Step="Step")
        return p

    def head(self, m):
        p = []
        d = {"rows": m, "batch": 1}
        self.op(
            p,
            "finalnorm",
            dict(
                X="Hidden",
                R="R0",
                I="IndexPrefill" if m > 1 else "IndexDecode",
                W="FinalWeight",
                Y="LastHidden",
            ),
            d,
        )
        head_bind = (
            dict(
                A="LastHidden",
                P="Head_P",
                PP="Head_P",
                S="Head_S",
                Z="Head_Z",
                Logits="Logits",
                O="Logits",
            )
            if self.dense_u4
            else dict(A="LastHidden", B="Head", C="Logits")
        )
        self.op(p, "head", head_bind, {})
        self.op(p, "topk", dict(X="Logits", PV="PV", PI="PI", Bad="Bad"), {"rows": 1})
        self.op(
            p,
            "topkmerge",
            dict(
                PV="PV",
                PI="PI",
                Bad="Bad",
                Values="Values",
                IDs="IDs",
                Token="Token",
                Status="TokenStatus",
            ),
            {"rows": 1},
        )
        return p

    def finish(self):
        C = self.prefill_tokens
        self.workspace()
        programs = {
            "prefill": self.program("prefill", C),
            "head": self.head(C),
            "decode": self.program("decode", 1) + self.head(1),
        }
        sizes = {"u8": 1, "i8": 1, "f16": 2, "f32": 4, "i32": 4}
        weight_bytes = sum(
            np.prod(b["shape"]).item() * sizes[b["dtype"]]
            for b in self.buffers
            if b["access"] == "read"
        )
        manifest = {
            "schema_version": 1,
            "target": "sm_87",
            "model": "Qwen3_5-text-27B",
            "chunk_tokens": C,
            "max_context": MAXPOS,
            "vocab": V,
            "toolchain": {k: str(v) for k, v in environment().items()},
            "buffers": self.buffers,
            "kernels": self.kernels,
            "programs": programs,
            "reset_buffers": self.reset,
            "input": "Input",
            "token": "Token",
            "status": "TokenStatus",
            "logits": "Logits",
            "position": "Step",
            "weight_bytes": weight_bytes,
            "weight_parameters": self.weight_parameters,
            "weight_scope": (
                "community W4 backbone + asymmetric group128 RNE U4 embedding/head candidate"
                if self.dense_u4
                else "community W4 backbone, original BF16 dense tables cast FP16; diagnostic above budget"
            )
            + (
                "; prefill temporary row-W8/per-token-A8 INT8, offline WS metadata counted; decode W4A16"
                if self.prefill_w4a8
                else ""
            )
            + (
                "; "
                + self.decode_register_scope
                + " projections use one losslessly warp-packed W4 copy shared by prefill/decode"
                if self.decode_register_mma
                else ""
            ),
        }
        write_json(self.out / "model.json", manifest)
        self.report.update(
            status="built",
            build_s=time.perf_counter() - self.started,
            weight_bytes=weight_bytes,
            effective_weight_bits=8 * weight_bytes / self.weight_parameters,
            sources=self.sources,
            checkpoint=str(self.checkpoint),
            checkpoint_identity=self.checkpoint_identity,
        )
        self.report["prefill_w4a8"] = self.prefill_w4a8
        self.report["prefill_tokens"] = C
        self.report["prefill_execution"] = (
            "One layer-major full graph per chunk; W8 expanded once per projection per chunk"
        )
        self.report["prefill_i8_grid_order"] = self.i8_grid_order
        self.report["w8_expand_mode"] = self.w8_expand_mode if self.prefill_w4a8 else None
        self.report["decode_down_implementation"] = (
            "warp-register MMA split-K8" if self.decode_register_mma else "register split-K8"
        ) + "; op32 FP32 merge"
        self.report["decode_register_mma"] = self.decode_register_mma
        self.report["decode_register_scope"] = (
            self.decode_register_scope if self.decode_register_mma else None
        )
        self.report["warp_packed_projection_tables"] = len(self.warp_weights)
        self.report["swiglu_a8_mode"] = self.swiglu_a8_mode if self.prefill_w4a8 else None
        if self.decode_register_mma:
            self.report["decode_ffn_layout"] = {
                "layout": WARP_LAYOUT,
                "gateup_tile_n": 128,
                "down_tile_n": 64,
                "single_packed_copy": True,
                "quantization_changed": False,
                "prefill_reader": "strict op29 warp-to-temporary-W8",
            }
        self.report["gdn_prefill_math"] = self.gdn_math
        self.report["decode_state_mode"] = self.decode_state_mode
        self.report["decode_attention_mode"] = self.decode_attention_mode
        self.report["prefill_norm_a8"] = self.prefill_norm_a8
        self.report["gdn_wy_mode"] = self.gdn_wy_mode
        self.report["gdn_solve_mode"] = self.gdn_solve_mode
        self.report["prefill_attention_mode"] = self.prefill_attention_mode
        if self.gdn_math == "compensated":
            self.report["gdn_precision"] = (
                "three FP16 products hi*hi/hi*lo/lo*hi, FP32 accumulation and persistent state; omits lo*lo and may underflow operand residuals; candidate requires model quality evaluation"
            )
        elif self.gdn_math == "factored":
            self.report["gdn_precision"] = (
                "state: compensated W@S, exact FP16 K.T @ split(decay*R); output: exact FP16 Q @ split(S), row q_scale*exp(G), compensated QK@R; FP32 accumulation and persistent state. Reassociation changes rounding and split residuals may underflow; requires model quality evaluation. WY selected independently."
            )
        if self.prefill_w4a8:
            self.report["w4a8_contract"] = {
                "prefill": "all 256 large backbone projections; per-call W4-to-rowwise-W8 + per-token-A8 + INT32 MMA, FP16 scale/output; SwiGLU FP16 boundary fused into A8",
                "decode": "existing W4A16",
                "retained_high_precision": [
                    "GDN a/b",
                    "norm/nonlinear",
                    "GDN FP32 state",
                    "FP16 attention/KV",
                ],
                "head": "existing last-position W4A16"
                if self.dense_u4
                else "FP16 original dense table",
                "weight_row_scale_bytes": sum(x["bytes"] for x in self.report["w8_row_scales"]),
                "extra_workspace_bytes": 2 * F * H + C * F + C * 2 + F,
                "quality": "unmeasured against BF16/FP8; token identity is not an acceptance gate",
            }
        write_json(self.out / "build-report.json", self.report)
        print("BUILD COMPLETE", self.report["build_s"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-aot")
    parser.add_argument("--reuse-weights")
    parser.add_argument("--dense-u4", action="store_true")
    parser.add_argument("--prefill-w4a8", action="store_true")
    parser.add_argument("--w8-expand-mode", choices=("dynamic", "aligned"), default="dynamic")
    parser.add_argument("--gdn-math", choices=("simt", "compensated", "factored"), default="simt")
    parser.add_argument("--prefill-tokens", type=int, choices=(512, 2048, 8192), default=512)
    parser.add_argument(
        "--prefill-grid-order", choices=("auto", "nfirst", "mfirst"), default="auto"
    )
    parser.add_argument("--decode-register-mma", action="store_true")
    parser.add_argument(
        "--swiglu-a8-mode", choices=("strict", "guarded-reciprocal", "lut"), default="strict"
    )
    parser.add_argument("--decode-register-scope", choices=("ffn", "all"), default="ffn")
    parser.add_argument(
        "--decode-state-mode", choices=("immutable", "inplace"), default="immutable"
    )
    parser.add_argument(
        "--decode-attention-mode", choices=("original", "staged"), default="original"
    )
    parser.add_argument("--prefill-norm-a8", action="store_true")
    parser.add_argument("--gdn-wy-mode", choices=("simt", "compensated"), default="simt")
    parser.add_argument(
        "--prefill-attention-mode", choices=("original", "staged64"), default="original"
    )
    parser.add_argument(
        "--gdn-solve-mode", choices=("registers", "columns-register"), default="registers"
    )
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("model.json", "weights", "aot")):
        parser.error("Output already contains model build products")
    b = Builder(
        out,
        Path(args.reuse_aot) if args.reuse_aot else None,
        Path(args.reuse_weights) if args.reuse_weights else None,
        args.dense_u4,
        args.prefill_w4a8,
        args.w8_expand_mode,
        args.gdn_math,
        args.prefill_tokens,
        args.prefill_grid_order,
        args.decode_register_mma,
        args.swiglu_a8_mode,
        args.decode_register_scope,
        args.decode_state_mode,
        args.decode_attention_mode,
        args.prefill_norm_a8,
        args.gdn_wy_mode,
        args.prefill_attention_mode,
        args.gdn_solve_mode,
        args.checkpoint,
    )
    configure()
    b.compile_all()
    b.weights()
    if b.decode_register_mma:
        b.repack_warp_weights()
    b.finish()


if __name__ == "__main__":
    main()
