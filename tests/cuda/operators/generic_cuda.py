"""On-device validation & benchmark for portable non-Orin CUDA TileLang operators.

Verifies numerical exactness, changed-input CUDA Graph replay, offline E8P
encoding, and AOT cubin export on SM80/86/89/90/100/120 GPUs without relying on
SM87-specific PTX or unified memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from kernels.cuda.generic import (
    e8p_dense_gemm,
    e8p_grouped_gemm,
    gdn_recurrent,
    plain_activation_a8,
    rotate_activation_a8,
    w8a8_gemm,
)
from orinfer_build.compiler.cuda.common import (
    benchmark,
    configure,
    cuda_arch,
    error,
    export_kernel,
    package_target,
)
from orinfer_build.contracts.abi import evaluate, parse_host, validate_parameter_count
from orinfer_build.quantization.e8p import basis
from orinfer_build.quantization.e8p_gpu import Encoder
from orinfer_build.quantization.vq import e8p_decode


def _rotate_block128_ref(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    y = x.half().float() * signs.float()
    k = y.shape[-1]
    y = y.reshape(-1, k // 128, 128)
    for stage in range(7):
        h = 1 << stage
        v = y.reshape(-1, 128 // (2 * h), 2, h)
        y = torch.stack((v[:, :, 0, :] + v[:, :, 1, :], v[:, :, 0, :] - v[:, :, 1, :]), dim=2)
        y = y.reshape(-1, k // 128, 128)
    return (y.reshape_as(x) * (128.0**-0.5)).half().float()


def _a8_ref(rot_f32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mx = rot_f32.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(mx > 0, (mx / 127.0).clamp_min(2.0**-24), torch.ones_like(mx)).half()
    q = torch.round(rot_f32 / scale.float()).clamp(-127.0, 127.0).to(torch.int8)
    return q, scale.squeeze(-1)


def _gdn_ref(q, k, v, decay, beta, state):
    b_dim, seq_len, hk, dk = k.shape
    hv, dv = v.shape[2], v.shape[3]
    qf = q.float()
    kf = k.float()
    vf = v.float()
    qf = qf * torch.rsqrt(qf.square().sum(-1, keepdim=True) + 1e-6)
    kf = kf * torch.rsqrt(kf.square().sum(-1, keepdim=True) + 1e-6)
    qf = qf.repeat_interleave(hv // hk, dim=2) / math.sqrt(dk)
    kf = kf.repeat_interleave(hv // hk, dim=2)
    ed = decay.float().exp()
    bf = beta.float()
    s = state.clone()
    out = torch.empty((b_dim, seq_len, hv, dv), dtype=q.dtype, device=q.device)
    for t in range(seq_len):
        s = s * ed[:, t, :, None, None]
        pred = torch.einsum("bhkv,bhk->bhv", s, kf[:, t])
        upd = bf[:, t, :, None] * (vf[:, t] - pred)
        s = s + kf[:, t, :, :, None] * upd[:, :, None, :]
        out[:, t] = torch.einsum("bhkv,bhk->bhv", s, qf[:, t]).to(q.dtype)
    return out, s


def _validate_rotation_and_plain_a8() -> dict:
    results = {}
    rng = np.random.default_rng(20261002)
    for label, rows, k, swiglu in [
        ("rotate_a8_k2560", 256, 2560, False),
        ("swiglu_rotate_a8_k640", 512, 640, True),
    ]:
        x = torch.randn((rows, 2 * k if swiglu else k), device="cuda", dtype=torch.float16) * 0.5
        signs = torch.from_numpy(rng.choice(np.array([-1, 1], np.int8), k)).cuda()
        q = torch.empty((rows, k), device="cuda", dtype=torch.int8)
        s = torch.empty((rows,), device="cuda", dtype=torch.float16)
        kern = rotate_activation_a8(rows, k, swiglu=swiglu, in_dtype="float16")

        def run():
            kern(x, signs, q, s)

        bench_res, graph = benchmark(run, repetitions=10, warmup=2)
        for delta in (0.0, 0.25):
            x.add_(delta)
            graph.replay()
            torch.cuda.synchronize()
            if swiglu:
                g, u = x.chunk(2, dim=-1)
                mid = (F.silu(g.float()) * u.float()).half()
                rot = _rotate_block128_ref(mid, signs)
            else:
                rot = _rotate_block128_ref(x, signs)
            q_ref, s_ref = _a8_ref(rot)
            max_q = int((q.int() - q_ref.int()).abs().max().item())
            max_s = float((s.float() - s_ref.float()).abs().max().item())
            assert max_q <= 1 and max_s <= 1e-3, f"{label} mismatch: max_q={max_q}, max_s={max_s}"
        results[label] = {"median_ms": round(bench_res["median_ms"], 4), "graph_replay": True}

    x = torch.randn((256, 2560), device="cuda", dtype=torch.float16) * 0.5
    q = torch.empty((256, 2560), device="cuda", dtype=torch.int8)
    s = torch.empty((256,), device="cuda", dtype=torch.float16)
    pkern = plain_activation_a8(256, 2560, in_dtype="float16")
    pkern(x, q, s)
    torch.cuda.synchronize()
    q_ref, s_ref = _a8_ref(x.half().float())
    assert int((q.int() - q_ref.int()).abs().max().item()) <= 1
    results["plain_a8_k2560"] = {"ok": True}
    return results


def _validate_w8a8(fixture_dir: Path | None = None) -> dict:
    m, n, k = 128, 256, 512
    a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    w = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    ws = (torch.rand((n,), device="cuda") * 0.01 + 0.001).half()
    as_ = (torch.rand((m,), device="cuda") * 0.02 + 0.002).half()
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    kern = w8a8_gemm(m, n, k, out_dtype="float16", block_m=64, block_n=64, block_k=128)

    def run():
        kern(a, w, ws, as_, out)

    bench_res, graph = benchmark(run, repetitions=10, warmup=2)
    for negate in (False, True):
        if negate:
            a.neg_()
        graph.replay()
        torch.cuda.synchronize()
        ref = (((a.float() @ w.float().T) * ws.float()[None, :]) * as_.float()[:, None]).half()
        err = error(out, ref)
        assert err["relative_l2"] < 1e-3, f"w8a8 error too high: {err}"

    if fixture_dir is not None:
        _export_rust_fixture(fixture_dir, kern, a, w, ws, as_, out)

    return {"median_ms": round(bench_res["median_ms"], 4), "relative_l2": err["relative_l2"]}


def _export_rust_fixture(
    out_dir: Path,
    kern,
    a: torch.Tensor,
    w: torch.Tensor,
    ws: torch.Tensor,
    as_: torch.Tensor,
    out: torch.Tensor,
):
    """Export a schema-1 AOT fixture verifiable directly by `orinfer-cli validate-fixture`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    export_kernel(kern, out_dir)
    (host,) = parse_host((out_dir / "host.txt").read_text())
    validate_parameter_count((out_dir / "kernel.cu").read_text(), host)

    ref = (((a.float() @ w.float().T) * ws.float()[None, :]) * as_.float()[:, None]).float()
    tensors = {
        "A": ("i8", a),
        "W": ("i8", w),
        "WS": ("f16", ws),
        "AS": ("f16", as_),
        "Out": ("f16", None),
    }

    def write_bin(fname: str, t: torch.Tensor) -> dict:
        raw = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        (out_dir / fname).write_bytes(raw)
        return {"file": fname, "sha256": hashlib.sha256(raw).hexdigest()}

    buffers = []
    for name, (dtype, tensor) in tensors.items():
        if tensor is None:
            buffers.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "shape": list(out.shape),
                    "layout": "row_major",
                    "alignment": 16,
                    "access": "write",
                    "data": None,
                }
            )
        else:
            buffers.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "shape": list(tensor.shape),
                    "layout": "row_major",
                    "alignment": 16,
                    "access": "read",
                    "data": write_bin(f"{name.lower()}.bin", tensor),
                }
            )

    ref_id = write_bin("ref.bin", ref)
    launch = host["launch_expressions"]
    args = []
    for arg in host["ordered_arguments"]:
        assert arg["ctype"] == "ctypes.c_void_p" and arg["value"].endswith(".data_ptr()")
        args.append({"kind": "buffer", "name": arg["value"].removesuffix(".data_ptr()")})

    def file_id(fname: str) -> dict:
        raw = (out_dir / fname).read_bytes()
        return {"file": fname, "sha256": hashlib.sha256(raw).hexdigest()}

    manifest = {
        "schema_version": 1,
        "target": package_target(),
        "toolchain": {"tilelang": "generic-cuda", "cuda_arch": cuda_arch()},
        "buffers": buffers,
        "kernels": [
            {
                "name": "generic_w8a8_gemm",
                "module": file_id("kernel.cubin"),
                "source": file_id("kernel.cu"),
                "host_abi": file_id("host.txt"),
                "symbol": host["symbol"],
                "grid": [evaluate(launch["gridDim" + axis], {}) for axis in "XYZ"],
                "block": [evaluate(launch["blockDim" + axis], {}) for axis in "XYZ"],
                "shared_memory_bytes": evaluate(launch["sharedMemBytes"], {}),
                "cooperative": False,
                "args": args,
            }
        ],
        "validation": {
            "output": "Out",
            "reference": ref_id,
            "relative_l2_tolerance": 0.002,
            "zero_input": "A",
            "repetitions": 20,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _validate_e8p_gemms() -> dict:
    table = basis()
    book = torch.from_numpy(table).cuda()
    grid = torch.from_numpy(
        e8p_decode(np.arange(65536, dtype=np.uint16), table).astype(np.float32)
    ).cuda()
    results = {}

    # 1. Grouped MoE E8P Q2A8 for BOTH packed_k=128 and packed_k=64
    for packed_k in (128, 64):
        e_cnt, n, k, bm = 16, 256, 640, 32
        counts = torch.tensor(
            [1, 0, 17, 32, 33, 5, 0, 64, 2, 19, 0, 8, 41, 15, 0, 11],
            device="cuda",
            dtype=torch.int32,
        )
        rows = int(counts.sum().item())
        offsets = torch.zeros(e_cnt, dtype=torch.int32, device="cuda")
        offsets[1:] = torch.cumsum(counts[:-1], dim=0, dtype=torch.int32)
        tiles_per_e = (counts + (bm - 1)) // bm
        total_tiles = int(tiles_per_e.sum().item())
        max_tiles = (rows + bm - 1) // bm + e_cnt
        tile_expert = torch.zeros(max_tiles, dtype=torch.int32, device="cuda")
        tile_row = torch.zeros(max_tiles, dtype=torch.int32, device="cuda")
        te_live = torch.repeat_interleave(
            torch.arange(e_cnt, dtype=torch.int32, device="cuda"), tiles_per_e
        )
        to = torch.zeros(e_cnt, dtype=torch.int32, device="cuda")
        to[1:] = torch.cumsum(tiles_per_e[:-1], dim=0, dtype=torch.int32)
        tr_live = (torch.arange(total_tiles, dtype=torch.int32, device="cuda") - to[te_live]) * bm
        tile_expert[:total_tiles] = te_live
        tile_row[:total_tiles] = tr_live
        tile_count = torch.tensor([total_tiles], dtype=torch.int32, device="cuda")

        a = torch.randint(-127, 128, (rows, k), device="cuda", dtype=torch.int8)
        codes = torch.randint(0, 65536, (e_cnt, n, k // 8), device="cuda", dtype=torch.int32).to(
            torch.uint16
        )
        p = codes.reshape(e_cnt, n, k // packed_k, packed_k // 8).permute(0, 2, 1, 3).contiguous()
        ws = (torch.rand((e_cnt, n), device="cuda") * 0.01 + 0.001).half()
        as_ = (torch.rand((rows,), device="cuda") * 0.02 + 0.002).half()
        out = torch.full((rows, n), 91.0, device="cuda", dtype=torch.float16)

        kern = e8p_grouped_gemm(
            rows, e_cnt, max_tiles, n, k, block_m=bm, block_n=64, packed_k=packed_k
        )
        bench_res, graph = benchmark(
            lambda: kern(
                a, p, book, ws, as_, counts, offsets, tile_expert, tile_row, tile_count, out
            ),
            repetitions=10,
            warmup=2,
        )
        for negate in (False, True):
            if negate:
                a.neg_()
            graph.replay()
            torch.cuda.synchronize()
            ref = torch.empty_like(out)
            for e in range(e_cnt):
                c = int(counts[e].item())
                if c == 0:
                    continue
                off = int(offsets[e].item())
                w_e = grid[codes[e].long()].reshape(n, k)
                dot = a[off : off + c].float() @ w_e.T
                ref[off : off + c] = (
                    (dot * ws[e].float()[None, :]) * as_[off : off + c].float()[:, None]
                ).half()
            err = error(out, ref)
            assert err["relative_l2"] < 1e-3, f"e8p_grouped packed_k={packed_k} failed: {err}"
        results[f"e8p_grouped_k{packed_k}"] = {
            "median_ms": round(bench_res["median_ms"], 4),
            "relative_l2": err["relative_l2"],
        }

    # 2. Dense E8P Q2A8
    rows, n, k = 128, 256, 640
    a = torch.randint(-127, 128, (rows, k), device="cuda", dtype=torch.int8)
    codes = torch.randint(0, 65536, (n, k // 8), device="cuda", dtype=torch.int32).to(torch.uint16)
    p = codes.reshape(n, k // 128, 16).permute(1, 0, 2).contiguous()
    ws = (torch.rand((n,), device="cuda") * 0.01 + 0.001).half()
    as_ = (torch.rand((rows,), device="cuda") * 0.02 + 0.002).half()
    out = torch.empty((rows, n), device="cuda", dtype=torch.float16)
    dkern = e8p_dense_gemm(rows, n, k, block_m=64, block_n=64)
    dkern(a, p, book, ws, as_, out)
    torch.cuda.synchronize()
    w_ref = grid[codes.long()].reshape(n, k)
    ref = (((a.float() @ w_ref.T) * ws.float()[None, :]) * as_.float()[:, None]).half()
    err = error(out, ref)
    assert err["relative_l2"] < 1e-3, f"e8p_dense failed: {err}"
    results["e8p_dense"] = {"relative_l2": err["relative_l2"]}
    return results


def _validate_gdn_recurrent() -> dict:
    b_dim, seq_len, hk, hv, dk, dv = 4, 128, 16, 48, 128, 128
    q = torch.randn((b_dim, seq_len, hk, dk), device="cuda", dtype=torch.bfloat16) * 0.3
    k = torch.randn((b_dim, seq_len, hk, dk), device="cuda", dtype=torch.bfloat16) * 0.3
    v = torch.randn((b_dim, seq_len, hv, dv), device="cuda", dtype=torch.bfloat16) * 0.3
    decay = -torch.rand((b_dim, seq_len, hv), device="cuda", dtype=torch.float32) * 0.5
    beta = torch.sigmoid(torch.randn((b_dim, seq_len, hv), device="cuda", dtype=torch.float32))
    init_state = torch.randn((b_dim, hv, dk, dv), device="cuda", dtype=torch.float32) * 0.02
    state = init_state.clone()
    out = torch.empty((b_dim, seq_len, hv, dv), device="cuda", dtype=torch.bfloat16)

    kern = gdn_recurrent(
        b_dim, seq_len, hk, hv, dk=dk, dv=dv, block_v=64, dtype="bfloat16", dynamic_shape=True
    )
    kern(q, k, v, decay, beta, state, out)
    torch.cuda.synchronize()
    ref_out, ref_state = _gdn_ref(q, k, v, decay, beta, init_state)
    err_out = error(out, ref_out)
    err_state = error(state, ref_state)
    assert err_out["relative_l2"] < 5e-3 and err_state["relative_l2"] < 5e-3

    # Verify stateful decode L=1 using the SAME compiled dynamic kernel!
    q1 = torch.randn((b_dim, 1, hk, dk), device="cuda", dtype=torch.bfloat16) * 0.3
    k1 = torch.randn((b_dim, 1, hk, dk), device="cuda", dtype=torch.bfloat16) * 0.3
    v1 = torch.randn((b_dim, 1, hv, dv), device="cuda", dtype=torch.bfloat16) * 0.3
    d1 = -torch.rand((b_dim, 1, hv), device="cuda", dtype=torch.float32) * 0.5
    b1 = torch.sigmoid(torch.randn((b_dim, 1, hv), device="cuda", dtype=torch.float32))
    out1 = torch.empty((b_dim, 1, hv, dv), device="cuda", dtype=torch.bfloat16)
    s1 = state.clone()
    kern(q1, k1, v1, d1, b1, s1, out1)
    torch.cuda.synchronize()
    ref_o1, ref_s1 = _gdn_ref(q1, k1, v1, d1, b1, state)
    assert error(out1, ref_o1)["relative_l2"] < 5e-3 and error(s1, ref_s1)["relative_l2"] < 5e-3
    return {
        "prefill_rel_l2": err_out["relative_l2"],
        "state_rel_l2": err_state["relative_l2"],
        "decode_step_ok": True,
    }


def _validate_offline_encoder() -> dict:
    enc = Encoder()
    rng = np.random.default_rng(20261002)
    raw = rng.normal(scale=0.03, size=(64, 256)).astype(np.float32)
    signs = rng.choice(np.array([-1, 1], np.int8), 256)
    fitted = enc.fit(raw, signs)
    assert fitted.indices.shape == (64, 32) and fitted.scales.shape == (64,)
    return {"ok": True, "shape": list(fitted.indices.shape)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-output", type=Path, default=None)
    args = parser.parse_args()
    t0 = time.perf_counter()
    configure()
    report = {
        "device": torch.cuda.get_device_name(0),
        "sm": list(torch.cuda.get_device_capability()),
        "cuda_arch": cuda_arch(),
        "package_target": package_target(),
        "rotation_a8": _validate_rotation_and_plain_a8(),
        "w8a8_gemm": _validate_w8a8(args.fixture_output),
        "e8p_gemm": _validate_e8p_gemms(),
        "gdn_recurrent": _validate_gdn_recurrent(),
        "offline_encoder": _validate_offline_encoder(),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
