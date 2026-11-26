"""8-NPU + 192-Core Activation-Calibrated SmoothQuant + Act-Order Hessian-GPTQ quantizer for DSv4.1.

Runs a multi-domain tokenized calibration corpus (e.g. 256 x 1024 = 262,144 tokens, strictly disjoint
from evaluation splits) layer-by-layer on the official MXFP4+FP8 reference trajectory across 8 Ascend
910B NPUs and all host CPU cores (8 NPU rank processes x 24 dedicated 1-thread CPU GPTQ worker
processes = 192 lock-free OS processes saturating 100% of host cores without OpenBLAS lock contention):

  - Exact Ascend A2 execution-graph SmoothQuant (zero HBM/kernel/latency cost):
      * q_norm -> wq_b + indexer.wq_b (all 40 layers)
      * attn_norm -> wq_a + wkv + indexer.weights_proj (36 non-KV-source W8A8 layers)
      * shared_experts w3/w2 SwiGLU SmoothQuant (exact 8x288 TP8 A8, clamp-exact |u_max| <= 8.5)
      * ffn_norm W4A8-gated SmoothQuant (W4 layers only where actual W4A8 MSE drops >= 3%)
  - Optimal exact-grid lossless W8 routed-expert layers (default: L15, L39 — top sensitivity within
    the 2-layer HBM budget so on-device prefix cache stays at full 4.6 GiB capacity).
  - Remaining W4 routed-expert layers:
      * Per-expert clamp-exact gated w3/w2 SwiGLU SmoothQuant (|u_max| <= 8.5, s in [1, 4])
      * Routing-weight-exact (wt^2) shrinkage-blended Act-Order Hessian-GPTQ across 192 CPU workers
        + instant NPU exact closed-form H-optimal scale (+ optional sparse Gauss-Seidel CD on top
        sensitive layers)
      * Direct Ascend Transposed-NZ packing and resumable per-(layer, rank) disk checkpoints.

Run inside the Ascend container with 1 thread per worker process (8 x 24 = 192 lock-free workers):
  bash tools/environments/ascend/run.sh \\
    env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python3 -m orinfer_build.models.deepseek_v41.calibrate_norot \\
    --official-dir /data/models/DeepSeek-V4.1-Flash-1017 \\
    --corpus /path/to/calib-corpus-256x1024.json \\
    --out-dir artifacts/dsv41-calibrated \\
    --w8-layers 15 39 --cd-layers 0 2 3 4 5 7 37 38 \\
    --num-seqs 256 --seq-len 1024 --workers-per-rank 24
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import importlib.util
import json
from multiprocessing import shared_memory
import os
from pathlib import Path
import shutil
import sys
import time
import types

import numpy as np

from orinfer_build.common.files import atomic_json
from orinfer_build.models.deepseek_v41.official import Checkpoint, Decoder, FP8_VALUES, decode_scale
from orinfer_build.models.deepseek_v41.quantize_norot import (
    GPTQQuantizer,
    KV_SOURCE_LAYERS,
    SSZQuantizer,
    pack_w4_nz_bank,
)
from orinfer_build.weights.safetensors_source import local_header

W8_DEFAULT = (15, 39)
CD_DEFAULT = (0, 2, 3, 4, 5, 7, 37, 38)
SWIGLU_SAFE_U_MAX = 8.5

_CPU_GPTQ = None
_CPU_SSZ = None


def _init_cpu_worker(cache_dir: str):
    """Initialize 1-thread pure-CPU GPTQ worker (never imports torch_npu or ACL)."""
    global _CPU_GPTQ, _CPU_SSZ
    import torch

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    _CPU_GPTQ = GPTQQuantizer(Path(cache_dir), n_threads=1)
    _CPU_SSZ = SSZQuantizer(Path(cache_dir))


def _cpu_factor_and_gptq(w_np: np.ndarray, h_np: np.ndarray, damp_frac: float):
    """Single-thread pure-CPU Act-Order Blocked GPTQ (zero OpenBLAS lock across 192 workers)."""
    import torch

    k = h_np.shape[0]
    h = np.array(h_np, dtype=np.float64, copy=True)
    dead = np.diag(h) <= 0.0
    h[dead, dead] = 1.0
    perm = np.argsort(-np.diag(h)).astype(np.int32)
    invperm = np.argsort(perm).astype(np.int32)
    hp = h[np.ix_(perm, perm)]
    mean_d = float(np.mean(np.diag(hp)))
    damp = max(damp_frac * mean_d, 1e-6)
    idx = np.arange(k)
    hinv_ch = None

    try:
        hd32 = np.ascontiguousarray(hp, dtype=np.float32)
        hd32[idx, idx] += np.float32(damp)
        l32 = torch.linalg.cholesky(torch.from_numpy(hd32))
        hi32 = torch.cholesky_inverse(l32)
        hinv_ch = torch.linalg.cholesky(hi32, upper=True).numpy()
        if not np.isfinite(hinv_ch).all():
            hinv_ch = None
    except RuntimeError:
        hinv_ch = None

    if hinv_ch is None:
        for mult in (1.0, 2.0, 5.0, 10.0, 50.0, 200.0):
            try:
                hd = hp.copy()
                hd[idx, idx] += damp * mult
                l_t = torch.linalg.cholesky(torch.from_numpy(hd))
                hinv_t = torch.cholesky_inverse(l_t)
                hinv_ch = torch.linalg.cholesky(hinv_t, upper=True).numpy().astype(np.float32)
                break
            except RuntimeError:
                continue
    if hinv_ch is None:
        hinv_ch = np.diag(1.0 / np.sqrt(np.maximum(np.diag(hp), 1e-6))).astype(np.float32)

    hinv_ch = np.ascontiguousarray(hinv_ch, dtype=np.float32)
    hd_full = (np.diag(h) + damp).astype(np.float32)
    hd_p = np.ascontiguousarray(hd_full[perm], dtype=np.float32)

    n = w_np.shape[0]
    wp = np.ascontiguousarray(w_np[:, perm], dtype=np.float32)
    s = np.empty(n, dtype=np.float32)
    _CPU_GPTQ.hdiag_ssz(wp.ctypes.data, hd_p.ctypes.data, n, k, 1, s.ctypes.data)

    block = 128
    qp = np.empty((n, k), dtype=np.int8)
    wp_t = torch.from_numpy(wp)
    hinv_t = torch.from_numpy(hinv_ch)
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        cnt = i2 - i1
        wb = np.ascontiguousarray(wp[:, i1:i2], dtype=np.float32)
        hb = np.ascontiguousarray(hinv_ch[i1:i2, i1:i2], dtype=np.float32)
        qb = np.empty((n, cnt), dtype=np.int8)
        eb = np.empty((n, cnt), dtype=np.float32)
        _CPU_GPTQ.diag_block(
            wb.ctypes.data,
            hb.ctypes.data,
            s.ctypes.data,
            n,
            cnt,
            1,
            qb.ctypes.data,
            eb.ctypes.data,
        )
        qp[:, i1:i2] = qb
        if i2 < k:
            wp_t[:, i2:].addmm_(torch.from_numpy(eb), hinv_t[i1:i2, i2:], alpha=-1.0)

    q = np.ascontiguousarray(qp[:, invperm], dtype=np.int8)
    return q, s, perm, float(damp)


def _cpu_gptq_expert_job(job):
    """Run UP + DOWN GPTQ for one expert on 1 dedicated CPU core via page-cache IPC files."""
    local_e, in_path, out_path, has_dn_h, damp_dn = job
    with open(in_path, "rb") as f:
        w_up = np.fromfile(f, dtype=np.float32, count=4608 * 5120).reshape(4608, 5120)
        h_up = np.fromfile(f, dtype=np.float32, count=5120 * 5120).reshape(5120, 5120)
        w_dn = np.fromfile(f, dtype=np.float32, count=5120 * 2304).reshape(5120, 2304)
        h_dn = (
            np.fromfile(f, dtype=np.float32, count=2304 * 2304).reshape(2304, 2304)
            if has_dn_h
            else None
        )

    q_u, s_u, perm_u, damp_u = _cpu_factor_and_gptq(w_up, h_up, damp_frac=0.05)

    if has_dn_h and h_dn is not None:
        q_d, s_d, perm_d, damp_d_val = _cpu_factor_and_gptq(w_dn, h_dn, damp_frac=damp_dn)
    else:
        q_d, s_d, _ = _CPU_SSZ(w_dn)
        perm_d = np.arange(2304, dtype=np.int32)
        damp_d_val = 0.0

    with open(out_path, "wb") as f:
        q_u.tofile(f)
        s_u.tofile(f)
        perm_u.tofile(f)
        q_d.tofile(f)
        s_d.tofile(f)
        perm_d.tofile(f)
    return local_e, in_path, out_path, has_dn_h, damp_u, damp_d_val


def fast_engram_rows(
    ckpt: Checkpoint, weight_name: str, indices_list: list[np.ndarray], n_threads: int = 24
) -> list[np.ndarray]:
    """Multi-threaded sorted pread for official F8_E4M3 + F8_E8M0 Engram tables (bit-exact)."""
    scale_name = weight_name.removesuffix("weight") + "scale"
    w_file = ckpt.directory / ckpt.index[weight_name]
    s_file = ckpt.directory / ckpt.index[scale_name]
    w_start, w_hdr = local_header(w_file)
    s_start, s_hdr = local_header(s_file)
    w_meta = w_hdr[weight_name]
    s_meta = s_hdr[scale_name]
    if w_meta["dtype"] != "F8_E4M3" or s_meta["dtype"] != "F8_E8M0":
        raise ValueError(f"Unexpected Engram dtypes: {w_meta['dtype']} / {s_meta['dtype']}")
    w_cols = int(w_meta["shape"][1])
    s_cols = int(s_meta["shape"][1])
    if w_cols != s_cols * 32:
        raise ValueError(f"Unexpected Engram scale width: {w_cols} vs {s_cols}")
    w_base = w_start + int(w_meta["data_offsets"][0])
    s_base = s_start + int(s_meta["data_offsets"][0])

    flat_all = np.concatenate([np.asarray(idx, dtype=np.int64).reshape(-1) for idx in indices_list])
    unique, inv = np.unique(flat_all, return_inverse=True)
    n_u = len(unique)
    w_buf = np.empty((n_u, w_cols), dtype=np.uint8)
    s_buf = np.empty((n_u, s_cols), dtype=np.uint8)

    w_fd = os.open(w_file, os.O_RDONLY)
    s_fd = w_fd if s_file == w_file else os.open(s_file, os.O_RDONLY)
    try:
        chunk = max(1, (n_u + n_threads * 4 - 1) // (n_threads * 4))

        def _read_chunk(bounds):
            i0, i1 = bounds
            for pos in range(i0, i1):
                r = int(unique[pos])
                w_buf[pos] = np.frombuffer(
                    os.pread(w_fd, w_cols, w_base + r * w_cols), dtype=np.uint8
                )
                s_buf[pos] = np.frombuffer(
                    os.pread(s_fd, s_cols, s_base + r * s_cols), dtype=np.uint8
                )

        tasks = [(i, min(i + chunk, n_u)) for i in range(0, n_u, chunk)]
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            list(pool.map(_read_chunk, tasks))
    finally:
        if s_fd != w_fd:
            os.close(s_fd)
        os.close(w_fd)

    vals = FP8_VALUES[w_buf] * np.repeat(decode_scale(s_buf), 32, axis=1)
    bits = np.ascontiguousarray(vals, dtype="<f4").view("<u4")
    vals_bf16 = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")

    out = []
    off = 0
    for idx in indices_list:
        arr = np.asarray(idx, dtype=np.int64)
        sz = arr.size
        out.append(vals_bf16[inv[off : off + sz]].reshape(*arr.shape, w_cols))
        off += sz
    return out


def quantize_exact_w8_device(w_dev, s_ue_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Device-accelerated exact-grid lossless W8 quantization matching quantize_exact_w8."""
    import torch

    w = w_dev.float()
    s_min = torch.from_numpy(np.asarray(s_ue_np, dtype=np.float32).min(axis=1)).to(w.device)
    amax = w.abs().amax(dim=1).clamp_min(1e-30)

    def _eval(raw_scale):
        bf16_s = raw_scale.clamp_min(1e-30).to(torch.bfloat16)
        dec = bf16_s.float()
        q = torch.round(w / dec[:, None]).clamp(-127.0, 127.0)
        err = (w - q * dec[:, None]).square().sum(dim=1)
        return q, bf16_s, err

    qa, sa, ea = _eval(amax / 127.0)
    qb, sb, eb = _eval(0.5 * s_min)
    qc, sc, ec = _eval(s_min)

    use_c = ec < ea
    best_q = torch.where(use_c[:, None], qc, qa)
    best_s = torch.where(use_c, sc, sa)
    best_e = torch.where(use_c, ec, ea)

    use_b = eb <= best_e
    best_q = torch.where(use_b[:, None], qb, best_q)
    best_s = torch.where(use_b, sb, best_s)
    best_e = torch.where(use_b, eb, best_e)

    energy = max(float(w.square().sum().item()), 1e-30)
    q_np = np.ascontiguousarray(best_q.to(torch.int8).T.cpu().numpy())
    enc_np = best_s.view(torch.uint16).cpu().numpy().astype("<u2")
    return q_np, enc_np, float(best_e.sum().item() / energy)


def npu_refine_h_scale(
    w_np: np.ndarray,
    h_np: np.ndarray,
    q_np: np.ndarray,
    perm_np: np.ndarray,
    damp_val: float,
    dev: str,
    cd_pass_fn,
    cd_iters: int = 0,
    cd_threads: int = 4,
):
    """Exact closed-form H-optimal scale (+ optional sparse Gauss-Seidel CD) on NPU."""
    import torch

    k = h_np.shape[0]
    h_damped = np.array(h_np, dtype=np.float32, copy=True)
    dead = np.diag(h_damped) <= 0.0
    h_damped[dead, dead] = 1.0
    h_damped[np.arange(k), np.arange(k)] += np.float32(damp_val)

    h_t = torch.from_numpy(h_damped).to(dev)
    w_t = torch.from_numpy(w_np).to(dev)
    wh_t = w_t @ h_t
    q_t = torch.from_numpy(q_np).to(dev).float()
    qh_t = q_t @ h_t
    num = (wh_t * q_t).sum(dim=1)
    den = (qh_t * q_t).sum(dim=1).clamp_min(1e-30)
    s = (num / den).clamp_min(1e-30).cpu().numpy().astype(np.float32)

    q_out = q_np
    if cd_iters > 0:
        q_out = np.ascontiguousarray(q_np, dtype=np.int8).copy()
        hd_full = np.ascontiguousarray(np.diag(h_damped), dtype=np.float32)
        n = w_np.shape[0]
        for _ in range(cd_iters):
            r_np = np.ascontiguousarray(
                (wh_t - torch.from_numpy(s).to(dev)[:, None] * qh_t).cpu().numpy(),
                dtype=np.float32,
            )
            cd_pass_fn(
                h_damped.ctypes.data,
                hd_full.ctypes.data,
                perm_np.ctypes.data,
                s.ctypes.data,
                n,
                k,
                cd_threads,
                q_out.ctypes.data,
                r_np.ctypes.data,
            )
            q_t = torch.from_numpy(q_out).to(dev).float()
            qh_t = q_t @ h_t
            num = (wh_t * q_t).sum(dim=1)
            den = (qh_t * q_t).sum(dim=1).clamp_min(1e-30)
            s = (num / den).clamp_min(1e-30).cpu().numpy().astype(np.float32)

    err_h = float(
        (
            (w_t - torch.from_numpy(s).to(dev)[:, None] * q_t)
            * (wh_t - torch.from_numpy(s).to(dev)[:, None] * qh_t)
        )
        .sum()
        .item()
    )
    en_h = max(float((w_t * wh_t).sum().item()), 1e-30)
    return q_out, s, max(err_h, 0.0) / en_h


def _round_fp8_e4m3_npu(x):
    """Bit-exact float8_e4m3fn round-to-nearest-even in FP32 on NPU via IEEE-754 bit ops."""
    import torch

    v = x.clamp(-448.0, 448.0)
    a = v.abs()
    min_norm = 2.0**-6
    is_sub = a < min_norm
    shifted = torch.where(is_sub, a + min_norm, a)
    bits = shifted.view(torch.int32)
    lsb = (bits >> 20) & 1
    rounded = ((bits + 0x7FFFF + lsb) & ~0xFFFFF).view(torch.float32)
    q = torch.where(is_sub, rounded - min_norm, rounded)
    return torch.where(v < 0, -q, q)


def quantize_activation_npu(x, group: int = 32, fp4: bool = False, e4m3_scale: bool = False):
    """Bit-exact NPU equivalent of reference.quantize_activation."""
    import torch

    f = x.detach().float().reshape(-1, group)
    maximum = f.abs().amax(-1, keepdim=True)
    if fp4 and e4m3_scale:
        scale = _round_fp8_e4m3_npu(maximum.clamp_min(6.0 * (2.0**-9)) / 6.0)
    else:
        maximum = maximum.clamp_min(6.0 * (2.0**-126) if fp4 else 1e-4)
        raw = maximum * (1.0 / 6.0 if fp4 else 1.0 / 448.0)
        bits = raw.view(torch.int32)
        exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
        scale = ((exponent.clamp(-126, 127) + 127) << 23).to(torch.int32).view(torch.float32)
    value = f / scale
    if fp4:
        mag = value.abs().clamp_max(6.0)
        q = torch.zeros_like(mag)
        q = torch.where(mag > 0.25, torch.full_like(q, 0.5), q)
        q = torch.where(mag >= 0.75, torch.full_like(q, 1.0), q)
        q = torch.where(mag > 1.25, torch.full_like(q, 1.5), q)
        q = torch.where(mag >= 1.75, torch.full_like(q, 2.0), q)
        q = torch.where(mag > 2.5, torch.full_like(q, 3.0), q)
        q = torch.where(mag >= 3.5, torch.full_like(q, 4.0), q)
        q = torch.where(mag > 5.0, torch.full_like(q, 6.0), q)
        value = torch.where(value < 0, -q, q)
    else:
        value = _round_fp8_e4m3_npu(value)
    return (value * scale).reshape(x.shape).to(dtype=x.dtype)


def sparse_attn_npu(q, kv, sink, indices, softmax_scale):
    """Exact 64-key online BF16-prob accumulation matching reference.sparse_attn."""
    import torch

    batches, tokens, heads, dimension = q.shape
    output = torch.empty_like(q)
    for batch in range(batches):
        for first in range(0, tokens, 64):
            end = min(first + 64, tokens)
            query = q[batch, first:end].float()
            maximum = torch.full(
                (end - first, heads, 1), -1e30, dtype=torch.float32, device=q.device
            )
            denominator = torch.zeros_like(maximum)
            accumulator = torch.zeros(
                end - first, heads, dimension, dtype=torch.float32, device=q.device
            )
            for start in range(0, indices.shape[-1], 64):
                selected = indices[batch, first:end, start : start + 64].to(
                    device=q.device, dtype=torch.long
                )
                values = kv[batch][selected.clamp_min(0)].float()
                scores = torch.einsum("shd,skd->shk", query, values) * softmax_scale
                scores.masked_fill_(selected[:, None, :] < 0, -torch.inf)
                updated = torch.maximum(maximum, scores.amax(-1, keepdim=True))
                alpha = (maximum - updated).exp()
                probabilities = (scores - updated).exp()
                denominator = denominator * alpha + probabilities.sum(-1, keepdim=True)
                accumulator = accumulator * alpha + torch.einsum(
                    "shk,skd->shd", probabilities.bfloat16().float(), values
                )
                maximum = updated
            denominator += (sink[None, :, None] - maximum).exp()
            output[batch, first:end] = (accumulator / denominator).to(q.dtype)
    return output


def load_official_module(official_dir: Path):
    """Load the official DSv4.1 inference/model.py wired to NPU reference helpers."""
    import torch

    from orinfer_build.models.deepseek_v41 import reference as ref_mod

    kernel = types.ModuleType("kernel")

    def _act_quant(x, block_size=32, scale_fmt=None, scale_dtype=None, inplace=False):
        res = quantize_activation_npu(x, block_size)
        if inplace:
            x.copy_(res)
            return x
        return res, None

    def _fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
        res = quantize_activation_npu(
            x, block_size, fp4=True, e4m3_scale=(scale_dtype == torch.float8_e4m3fn)
        )
        if inplace:
            x.copy_(res)
            return x
        return res, None

    kernel.act_quant = _act_quant
    kernel.fp4_act_quant = _fp4_act_quant
    kernel.hc_split_sinkhorn = ref_mod.hc_split_sinkhorn
    kernel.sparse_attn = sparse_attn_npu
    kernel.fp4_gemm = kernel.fp8_gemm = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("unexpected direct gemm")
    )
    sys.modules["kernel"] = kernel

    vision = types.ModuleType("vision")
    vision.Aligner = vision.ViT = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("vision unused")
    )
    sys.modules["vision"] = vision

    infer_dir = official_dir / "inference"
    if str(infer_dir) not in sys.path:
        sys.path.insert(0, str(infer_dir))
    spec = importlib.util.spec_from_file_location(
        "official_deepseek_v41_calib", infer_dir / "model.py"
    )
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    official.apply_rotary_emb = ref_mod.apply_rotary_emb
    return official


def linear_official(x, w, quantized: bool):
    import torch
    import torch.nn.functional as F

    dtype = x.dtype
    if quantized:
        x = quantize_activation_npu(x, 32, fp4=False)
    if w.dtype == torch.float32:
        return F.linear(x.float(), w).to(dtype)
    return F.linear(x, w).to(dtype)


def rms_norm(x, weight, eps: float):
    import torch

    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (weight.float() * xf).to(dtype)


def _quant_a8_per_token(x):
    import torch

    xf = x.float()
    s = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
    return (torch.round(xf / s).clamp(-127.0, 127.0) * s).to(x.dtype)


def _quant_a8_tp8_chunks(x, chunks: int = 8):
    import torch

    rows, dim = x.shape
    xc = x.float().view(rows, chunks, dim // chunks)
    s = xc.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
    return (torch.round(xc / s).clamp(-127.0, 127.0) * s).view(rows, dim).to(x.dtype)


def fit_smoothquant_scale(
    x_probe,
    ch_max_global,
    w_list: list,
    alphas=(0.0, 0.35, 0.50, 0.65, 0.75),
    clamp_min: float = 1e-3,
    clamp_max: float = 1e3,
    min_gain_pct: float = 2.0,
):
    """Find SmoothQuant scale minimizing actual per-token A8 + per-channel W8 output MSE."""
    import torch

    xv = x_probe.float()
    w_col_max = (
        torch.stack([w.float().abs().amax(dim=0) for w in w_list], dim=0)
        .amax(dim=0)
        .clamp_min(1e-8)
    )
    w_eval = [w.float() for w in w_list]
    y_refs = [xv @ w.T for w in w_eval]
    y_ens = [float(y.square().sum().item()) + 1e-30 for y in y_refs]

    def eval_s(scale):
        xs = _quant_a8_per_token(xv / scale[None, :])
        errs = []
        for w, y_ref, y_en in zip(w_eval, y_refs, y_ens, strict=True):
            ws = w * scale[None, :]
            wsc = ws.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) / 127.0
            wq = torch.round(ws / wsc).clamp(-127.0, 127.0) * wsc
            errs.append(float(((xs @ wq.T) - y_ref).square().sum().item() / y_en))
        return float(np.mean(errs))

    ones = torch.ones_like(ch_max_global)
    raw = eval_s(ones)
    best_a, best_e, best_s = 0.0, raw, ones
    for a in alphas:
        if a == 0.0:
            continue
        s = (ch_max_global.pow(a) / w_col_max.pow(1.0 - a)).clamp(clamp_min, clamp_max)
        e = eval_s(s)
        if e < best_e:
            best_e, best_a, best_s = e, a, s

    gain_pct = (1.0 - best_e / max(raw, 1e-30)) * 100.0
    if gain_pct < min_gain_pct:
        return ones, 0.0, raw, raw, 0.0
    return best_s, best_a, raw, best_e, gain_pct


def fit_shared_swiglu_sq(mid_probe, mid_ch_max, u_unclamped_max, sh_w2):
    """Exact Ascend TP8 (8x288 A8 + full-2304 W8) clamp-exact SmoothQuant for shared_experts."""
    import torch

    mf = mid_probe.float()
    w2f = sh_w2.float()
    w_col_max = w2f.abs().amax(dim=0).clamp_min(1e-8)
    safe_mask = u_unclamped_max <= SWIGLU_SAFE_U_MAX
    y_ref = mf @ w2f.T
    y_en = float(y_ref.square().sum().item()) + 1e-30

    def eval_s(scale):
        ms = _quant_a8_tp8_chunks(mf / scale[None, :], chunks=8)
        ws = w2f * scale[None, :]
        wsc = ws.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) / 127.0
        wq = torch.round(ws / wsc).clamp(-127.0, 127.0) * wsc
        return float(((ms @ wq.T) - y_ref).square().sum().item() / y_en)

    ones = torch.ones_like(mid_ch_max)
    raw = eval_s(ones)
    best_a, best_e, best_s = 0.0, raw, ones
    med = mid_ch_max.median().clamp_min(1e-6)
    rel = (mid_ch_max / med).clamp_min(1.0)
    for a in (0.35, 0.50, 0.65, 0.75):
        for cand in (
            (mid_ch_max.pow(a) / w_col_max.pow(1.0 - a)).clamp(1.0, 6.0),
            rel.pow(a).clamp(1.0, 6.0),
        ):
            s = torch.where(safe_mask, cand, ones)
            e = eval_s(s)
            if e < best_e:
                best_e, best_a, best_s = e, a, s

    gain_pct = (1.0 - best_e / max(raw, 1e-30)) * 100.0
    if gain_pct < 2.0:
        return ones, 0.0, raw, raw, 0.0
    return best_s, best_a, raw, best_e, gain_pct


def fit_ffn_w4a8_sq(x_probe, ch_max_global, w4_samples: list, w8_samples: list):
    """W4A8-aware ffn_norm SmoothQuant gated directly on routed W4 + A8 MSE."""
    import torch

    xv = x_probe.float()
    w_all = [w.float() for w in w4_samples] + [w.float() for w in w8_samples]
    w_col_max = torch.stack([w.abs().amax(dim=0) for w in w_all], dim=0).amax(dim=0).clamp_min(1e-8)
    w4f = [w.float() for w in w4_samples]
    y4_refs = [xv @ w.T for w in w4f]
    y4_ens = [float(y.square().sum().item()) + 1e-30 for y in y4_refs]

    def eval_w4a8(scale):
        xs = _quant_a8_per_token(xv / scale[None, :])
        errs = []
        for w, y_ref, y_en in zip(w4f, y4_refs, y4_ens, strict=True):
            ws = w * scale[None, :]
            wsc = (ws.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) * 0.85) / 7.5
            wq = torch.round(ws / wsc).clamp(-8.0, 7.0) * wsc
            errs.append(float(((xs @ wq.T) - y_ref).square().sum().item() / y_en))
        return float(np.mean(errs))

    ones = torch.ones_like(ch_max_global)
    raw = eval_w4a8(ones)
    best_a, best_e, best_s = 0.0, raw, ones
    med = ch_max_global.median().clamp_min(1e-6)
    rel = (ch_max_global / med).clamp_min(1.0)
    for a in (0.15, 0.25, 0.35):
        for cand in (
            (ch_max_global.pow(a) / w_col_max.pow(1.0 - a)).clamp(0.5, 2.5),
            rel.pow(a).clamp(1.0, 2.5),
        ):
            e = eval_w4a8(cand)
            if e < best_e:
                best_e, best_a, best_s = e, a, cand

    gain_pct = (1.0 - best_e / max(raw, 1e-30)) * 100.0
    if gain_pct < 3.0:
        return ones, 0.0, 0.0
    return best_s, best_a, gain_pct


def fit_expert_swiglu_sq(mid_probe, w2, ch_max_m, u_unclamped_max):
    """Per-expert clamp-exact w3/w2 SmoothQuant gated on combined A8(mid/s) + W4(w2*s) MSE."""
    import torch

    mf = mid_probe.float()
    w2f = w2.float()
    safe_mask = u_unclamped_max <= SWIGLU_SAFE_U_MAX
    med = ch_max_m.median().clamp_min(1e-6)
    rel = (ch_max_m / med).clamp_min(1.0)
    if float(torch.where(safe_mask, rel, torch.ones_like(rel)).max().item()) < 2.5:
        return torch.ones_like(ch_max_m), 0.0, 0.0

    y_ref = mf @ w2f.T
    y_en = float(y_ref.square().sum().item()) + 1e-30

    def eval_s(s):
        ms = _quant_a8_per_token(mf / s[None, :])
        ws = w2f * s[None, :]
        wsc = (ws.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) * 0.85) / 7.5
        wq = torch.round(ws / wsc).clamp(-8.0, 7.0) * wsc
        return float(((ms @ wq.T) - y_ref).square().sum().item() / y_en)

    ones = torch.ones_like(ch_max_m)
    raw = eval_s(ones)
    best_a, best_e, best_s = 0.0, raw, ones
    for a in (0.25, 0.40, 0.55):
        cand = rel.pow(a).clamp(1.0, 4.0)
        s = torch.where(safe_mask, cand, ones)
        e = eval_s(s)
        if e < best_e:
            best_e, best_a, best_s = e, a, s

    gain = (1.0 - best_e / max(raw, 1e-30)) * 100.0
    if gain < 2.0:
        return ones, 0.0, 0.0
    return best_s, best_a, gain


def _worker_fn(rank: int, world_size: int, args_dict: dict, shm_names: dict):
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torch.nn.functional as F

    t_start = time.monotonic()
    n_cpu_workers = args_dict["workers_per_rank"]
    official_dir = Path(args_dict["official_dir"])
    corpus_path = Path(args_dict["corpus"])
    out_dir = Path(args_dict["out_dir"])
    prep_dir = out_dir / "prepared-experts"
    sq_dir = out_dir / "smoothquant"
    cache_dir = out_dir / "native-cache"
    ipc_dir = out_dir / "ipc" / f"r{rank}"
    if ipc_dir.exists():
        shutil.rmtree(ipc_dir)
    ipc_dir.mkdir(parents=True, exist_ok=True)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # Spawn pure-CPU 1-thread GPTQ workers BEFORE importing torch_npu or initializing ACL.
    cpu_pool = ProcessPoolExecutor(
        max_workers=n_cpu_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_cpu_worker,
        initargs=(str(cache_dir),),
    )
    main_gptq = GPTQQuantizer(cache_dir, n_threads=4)

    import torch_npu  # noqa: F401
    from transformers import AutoTokenizer

    from orinfer_build.models.deepseek_v41 import reference as ref_mod

    dev = f"npu:{rank}"
    torch.npu.set_device(dev)
    torch.npu.matmul.allow_hf32 = False

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args_dict["port"])
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    w8_set = set(args_dict["w8_layers"])
    cd_layers = set(args_dict["cd_layers"])
    kv_src_set = set(KV_SOURCE_LAYERS)

    n_seqs_total = args_dict["num_seqs"]
    seq_len = args_dict["seq_len"]
    n_tot = n_seqs_total * seq_len
    n_loc_seqs = n_seqs_total // world_size
    n_loc_tok = n_loc_seqs * seq_len

    shm_x = shared_memory.SharedMemory(name=shm_names["x"])
    shm_xq = shared_memory.SharedMemory(name=shm_names["xq"])
    shm_idx = shared_memory.SharedMemory(name=shm_names["idx"])
    shm_wt = shared_memory.SharedMemory(name=shm_names["wt"])

    x_shm = np.ndarray((n_tot, 5120), dtype=np.uint16, buffer=shm_x.buf)
    xq_shm = np.ndarray((n_tot, 5120), dtype=np.uint16, buffer=shm_xq.buf)
    idx_shm = np.ndarray((n_tot, 6), dtype=np.int32, buffer=shm_idx.buf)
    wt_shm = np.ndarray((n_tot, 6), dtype=np.float32, buffer=shm_wt.buf)

    official = load_official_module(official_dir)
    cfg = official.ModelArgs(**json.loads((official_dir / "inference/config.json").read_text()))
    cfg.max_batch_size, cfg.max_seq_len = 1, seq_len + 1
    cfg.dspark_block_size, cfg.n_mtp_layers, cfg.vision_n_layers, cfg.temperature = 0, 0, 0, 0

    corpus = json.loads(corpus_path.read_text())["sequences"][:n_seqs_total]
    my_seqs = corpus[rank * n_loc_seqs : (rank + 1) * n_loc_seqs]

    decoder = Decoder(cache_dir)
    ckpt = Checkpoint(official_dir, decoder)

    tokenizer = AutoTokenizer.from_pretrained(official_dir, local_files_only=True)
    engram_layout = official.EngramLayout.from_args(cfg)
    eh = official.NgramHashState(cfg, engram_layout, tokenizer).to(dev)

    emb_full = ckpt.matrix("embed.weight")
    engram_hashes, h_list, pre_mix_list = [], [], []
    for s in my_seqs:
        toks = s["tokens"][:seq_len]
        ids = torch.tensor([toks], dtype=torch.long, device=dev)
        engram_hashes.append(eh(ids, 0, None).cpu())
        hb = (
            torch.from_numpy(emb_full[np.asarray(toks, dtype=np.int64)])
            .to(dtype=torch.bfloat16)
            .unsqueeze(0)
        )
        hb = hb.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        h_list.append(hb)
        pre_mix_list.append(official.make_identity_pre_mix(hb, cfg.hc_mult))
    del eh, emb_full

    freqs_plain = official.precompute_freqs_cis(
        cfg.rope_head_dim,
        seq_len,
        0,
        cfg.rope_theta,
        cfg.rope_factor,
        cfg.beta_fast,
        cfg.beta_slow,
    ).to(dev)
    freqs_comp = official.precompute_freqs_cis(
        cfg.rope_head_dim,
        seq_len,
        cfg.original_seq_len,
        cfg.compress_rope_theta,
        cfg.rope_factor,
        cfg.beta_fast,
        cfg.beta_slow,
    ).to(dev)
    topk_win = official.get_window_topk_idxs(cfg.window_size, 1, seq_len, 0).to(dev)
    shared_states = [
        dict(compress_kv=None, index_k=None, topk_idxs=None, candidates=None)
        for _ in range(n_loc_seqs)
    ]

    def read_t(name, dtype):
        arr = ckpt.matrix(name) if name.endswith(".weight") else ckpt.read(name)
        return torch.from_numpy(np.array(arr, copy=True)).to(device=dev, dtype=dtype)

    if rank == 0:
        print(
            f"LAUNCHED 8-NPU + {8 * n_cpu_workers}-PROC FULL-CORE CALIBRATION: "
            f"{n_seqs_total}x{seq_len}={n_tot:,} tokens, w8_layers={sorted(w8_set)}, "
            f"cd_layers={sorted(cd_layers)}",
            flush=True,
        )

    try:
        with torch.inference_mode():
            for layer_id in range(cfg.n_layers):
                tl0 = time.monotonic()
                prefix = f"layers.{layer_id}"

                t_eng = 0.0
                if layer_id in engram_layout.layer_ids:
                    te0 = time.monotonic()
                    l_idx = engram_layout.layer_ids.index(layer_id)
                    idx_list = [engram_hashes[b][0, :, l_idx, :].numpy() for b in range(n_loc_seqs)]
                    eng_rows = fast_engram_rows(
                        ckpt, f"{prefix}.engram.embed.weight", idx_list, n_threads=n_cpu_workers
                    )
                    ewkv = read_t(f"{prefix}.engram.wkv.weight", torch.bfloat16)
                    ew_prod = read_t(f"{prefix}.engram.q_weight", torch.float32) * read_t(
                        f"{prefix}.engram.k_weight", torch.float32
                    )
                    for b in range(n_loc_seqs):
                        flat = (
                            torch.from_numpy(eng_rows[b])
                            .view(torch.bfloat16)
                            .to(device=dev)
                            .reshape(1, seq_len, -1)
                        )
                        kv = linear_official(flat, ewkv, quantized=True)
                        key, val = kv.split([cfg.hc_mult * cfg.dim, cfg.dim], dim=-1)
                        key = key.float().unflatten(-1, (cfg.hc_mult, cfg.dim))
                        hb = h_list[b].to(dev).float()
                        rstd = torch.rsqrt(hb.square().mean(-1) + cfg.norm_eps) * torch.rsqrt(
                            key.square().mean(-1) + cfg.norm_eps
                        )
                        dot = (hb * ew_prod * key).sum(-1) * rstd * (cfg.dim**-0.5)
                        mag = dot.abs().clamp_min(1e-6).sqrt()
                        gate = torch.sigmoid(torch.where(dot < 0, -mag, mag))
                        h_list[b] = (
                            (hb + gate.unsqueeze(-1) * val.float().unsqueeze(-2))
                            .to(torch.bfloat16)
                            .cpu()
                        )
                    del ewkv, ew_prod, eng_rows
                    t_eng = time.monotonic() - te0

                ratio = cfg.compress_ratios[layer_id]
                is_kv_src = layer_id in cfg.kv_source_layers
                is_idx_src = layer_id in cfg.index_source_layers
                is_cand_src = layer_id == cfg.candidate_source_layer
                uses_cand = 0 <= cfg.candidate_source_layer < layer_id

                ld = {
                    "hc_attn_fn": read_t(f"{prefix}.hc_attn_fn", torch.float32),
                    "hc_attn_scale": read_t(f"{prefix}.hc_attn_scale", torch.float32),
                    "hc_attn_base": read_t(f"{prefix}.hc_attn_base", torch.float32),
                    "hc_ffn_fn": read_t(f"{prefix}.hc_ffn_fn", torch.float32),
                    "hc_ffn_scale": read_t(f"{prefix}.hc_ffn_scale", torch.float32),
                    "hc_ffn_base": read_t(f"{prefix}.hc_ffn_base", torch.float32),
                    "attn_norm": read_t(f"{prefix}.attn_norm.weight", torch.bfloat16),
                    "ffn_norm": read_t(f"{prefix}.ffn_norm.weight", torch.bfloat16),
                    "wq_a": read_t(f"{prefix}.attn.wq_a.weight", torch.bfloat16),
                    "q_norm": read_t(f"{prefix}.attn.q_norm.weight", torch.bfloat16),
                    "wq_b": read_t(f"{prefix}.attn.wq_b.weight", torch.bfloat16),
                    "wkv": read_t(f"{prefix}.attn.wkv.weight", torch.bfloat16),
                    "kv_norm": read_t(f"{prefix}.attn.kv_norm.weight", torch.bfloat16),
                    "attn_sink": read_t(f"{prefix}.attn.attn_sink", torch.float32),
                    "wo_a": read_t(f"{prefix}.attn.wo_a.weight", torch.bfloat16),
                    "wo_b": read_t(f"{prefix}.attn.wo_b.weight", torch.bfloat16),
                    "gate_w": read_t(f"{prefix}.ffn.gate.weight", torch.float32),
                    "gate_b": read_t(f"{prefix}.ffn.gate.bias", torch.float32),
                    "sh_w1": read_t(f"{prefix}.ffn.shared_experts.w1.weight", torch.bfloat16),
                    "sh_w2": read_t(f"{prefix}.ffn.shared_experts.w2.weight", torch.bfloat16),
                    "sh_w3": read_t(f"{prefix}.ffn.shared_experts.w3.weight", torch.bfloat16),
                }
                if is_kv_src:
                    ld["comp_norm"] = read_t(
                        f"{prefix}.attn.compressor.norm.weight", torch.bfloat16
                    )
                    ld["comp_wkv"] = read_t(
                        f"{prefix}.attn.compressor.wkv.weight",
                        torch.float32 if ratio > 1 else torch.bfloat16,
                    )
                    if ratio > 1:
                        ld["comp_wgate"] = read_t(
                            f"{prefix}.attn.compressor.wgate.weight", torch.float32
                        )
                    ld["idx_wk"] = read_t(f"{prefix}.attn.indexer.wk.weight", torch.bfloat16)
                    ld["idx_k_norm"] = read_t(
                        f"{prefix}.attn.indexer.k_norm.weight", torch.bfloat16
                    )
                if is_idx_src:
                    ld["idx_wqb"] = read_t(f"{prefix}.attn.indexer.wq_b.weight", torch.bfloat16)
                    ld["idx_wp"] = read_t(
                        f"{prefix}.attn.indexer.weights_proj.weight", torch.bfloat16
                    )

                freqs = freqs_comp if ratio else freqs_plain
                rd = cfg.rope_head_dim

                xa_max = torch.zeros(5120, dtype=torch.float32, device=dev)
                qr_max = torch.zeros(1280, dtype=torch.float32, device=dev)
                xf_max = torch.zeros(5120, dtype=torch.float32, device=dev)
                sh_m_max = torch.zeros(2304, dtype=torch.float32, device=dev)
                sh_u_max = torch.zeros(2304, dtype=torch.float32, device=dev)
                H_ffn_loc = torch.zeros((5120, 5120), dtype=torch.float32, device=dev)

                probe_xa, probe_qr, probe_xf, probe_shm = [], [], [], []
                ffn_res_loc, ffn_post_loc, ffn_comb_loc, shared_out_loc = [], [], [], []

                tok_base = rank * n_loc_tok
                for b in range(n_loc_seqs):
                    x = h_list[b].to(dev)
                    pm = pre_mix_list[b].to(dev)
                    res = x
                    xf = x.flatten(2).float()
                    rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + cfg.norm_eps)
                    mix = F.linear(xf, ld["hc_attn_fn"]) * rs
                    attn_pre, attn_post, attn_comb = ref_mod.hc_split_sinkhorn(
                        mix,
                        ld["hc_attn_scale"],
                        ld["hc_attn_base"],
                        cfg.hc_mult,
                        cfg.hc_sinkhorn_iters,
                        cfg.hc_eps,
                    )
                    xa = rms_norm(
                        torch.sum(pm.unsqueeze(-1) * x.float(), dim=2).to(x.dtype),
                        ld["attn_norm"],
                        cfg.norm_eps,
                    )
                    xa0 = xa[0]
                    xa_max = torch.maximum(xa_max, xa0.float().abs().amax(dim=0))

                    qr = rms_norm(linear_official(xa, ld["wq_a"], True), ld["q_norm"], cfg.norm_eps)
                    qr0 = qr[0]
                    qr_max = torch.maximum(qr_max, qr0.float().abs().amax(dim=0))

                    q = linear_official(qr, ld["wq_b"], True).unflatten(
                        -1, (cfg.n_heads, cfg.head_dim)
                    )
                    ref_mod.apply_rotary_emb(q[..., -rd:], freqs)
                    kv = rms_norm(linear_official(xa, ld["wkv"], True), ld["kv_norm"], cfg.norm_eps)
                    ref_mod.apply_rotary_emb(kv[..., -rd:], freqs)
                    kv = quantize_activation_npu(kv, 32, fp4=False)
                    topk_idxs = topk_win

                    st = shared_states[b]
                    if ratio:
                        compress_len = seq_len // ratio
                        latent = None
                        if is_kv_src:
                            if ratio == 1:
                                latent = rms_norm(
                                    linear_official(xa, ld["comp_wkv"], False),
                                    ld["comp_norm"],
                                    cfg.norm_eps,
                                )
                            else:
                                xaf = xa.float()
                                ckv = F.linear(xaf, ld["comp_wkv"]).unflatten(1, (-1, ratio))
                                cscore = F.linear(xaf, ld["comp_wgate"]).unflatten(1, (-1, ratio))
                                latent = rms_norm(
                                    (ckv * cscore.softmax(dim=2)).sum(dim=2).to(xa.dtype),
                                    ld["comp_norm"],
                                    cfg.norm_eps,
                                )
                        if is_idx_src:
                            if is_kv_src:
                                f_comp = freqs[: seq_len - seq_len % ratio : ratio]
                                ik = rms_norm(
                                    linear_official(latent, ld["idx_wk"], False),
                                    ld["idx_k_norm"],
                                    cfg.norm_eps,
                                )
                                ref_mod.apply_rotary_emb(ik[..., -rd:], f_comp)
                                st["index_k"] = quantize_activation_npu(
                                    ik, 32, fp4=True, e4m3_scale=False
                                ).cpu()
                            iq = linear_official(qr, ld["idx_wqb"], True).unflatten(
                                -1, (cfg.index_n_heads, cfg.index_head_dim)
                            )
                            ref_mod.apply_rotary_emb(iq[..., -rd:], freqs)
                            iq = quantize_activation_npu(iq, 32, fp4=True, e4m3_scale=False)
                            ik_all = st["index_k"][:, :compress_len].to(dev)
                            iweights = linear_official(xa, ld["idx_wp"], False) * (
                                (cfg.index_head_dim**-0.5) * (cfg.index_n_heads**-0.5)
                            )
                            iscore = torch.empty(
                                (1, seq_len, compress_len), dtype=torch.float32, device=dev
                            )
                            for s0 in range(0, seq_len, 256):
                                s1 = min(s0 + 256, seq_len)
                                sc = torch.einsum(
                                    "bshd,btd->bsht", iq[:, s0:s1].float(), ik_all.float()
                                ).relu_()
                                iscore[:, s0:s1] = (
                                    sc * iweights[:, s0:s1].float().unsqueeze(-1)
                                ).sum(dim=2)
                            clens = (torch.arange(1, seq_len + 1, device=dev) // ratio).unsqueeze(
                                -1
                            )
                            iscore.masked_fill_(
                                torch.arange(compress_len, device=dev) >= clens, -torch.inf
                            )
                            if is_cand_src:
                                st["candidates"] = official.select_candidate_blocks(
                                    iscore,
                                    clens,
                                    cfg.candidate_topk_blocks,
                                    cfg.candidate_block_size,
                                ).cpu()
                            elif uses_cand:
                                iscore = iscore.masked_fill(~st["candidates"].to(dev), -torch.inf)
                            tk = min(cfg.index_topk, compress_len)
                            cidxs = (
                                iscore.topk(tk, dim=-1, sorted=False).indices.sort(dim=-1).values
                            )
                            st["topk_idxs"] = (
                                torch.where(cidxs < clens, cidxs + seq_len, -1).int().cpu()
                            )
                        if latent is not None:
                            f_comp = freqs[: seq_len - seq_len % ratio : ratio]
                            ref_mod.apply_rotary_emb(latent[..., -rd:], f_comp)
                            st["compress_kv"] = quantize_activation_npu(
                                latent, 16, fp4=True, e4m3_scale=True
                            ).cpu()
                        kv = torch.cat([kv, st["compress_kv"][:, :compress_len].to(dev)], dim=1)
                        topk_idxs = torch.cat([topk_idxs, st["topk_idxs"].to(dev)], dim=-1)

                    o = sparse_attn_npu(q, kv, ld["attn_sink"], topk_idxs, cfg.head_dim**-0.5)
                    ref_mod.apply_rotary_emb(o[..., -rd:], freqs, True)
                    o = quantize_activation_npu(o.view(1, seq_len, cfg.o_groups, -1), 32, fp4=False)
                    ao = linear_official(
                        torch.einsum(
                            "bsgd,grd->bsgr",
                            o,
                            ld["wo_a"].view(cfg.o_groups, cfg.o_lora_rank, -1),
                        ).flatten(2),
                        ld["wo_b"],
                        True,
                    )
                    x = (
                        attn_post.unsqueeze(-1) * ao.unsqueeze(-2)
                        + torch.sum(attn_comb.unsqueeze(-1) * res.unsqueeze(-2), dim=2)
                    ).to(x.dtype)
                    ffn_res_loc.append(x.cpu())

                    xf2 = x.flatten(2).float()
                    rs2 = torch.rsqrt(xf2.square().mean(-1, keepdim=True) + cfg.norm_eps)
                    mix2 = F.linear(xf2, ld["hc_ffn_fn"]) * rs2
                    ffn_pre, ffn_post, ffn_comb = ref_mod.hc_split_sinkhorn(
                        mix2,
                        ld["hc_ffn_scale"],
                        ld["hc_ffn_base"],
                        cfg.hc_mult,
                        cfg.hc_sinkhorn_iters,
                        cfg.hc_eps,
                    )
                    pre_mix_list[b] = ffn_pre.cpu()
                    ffn_post_loc.append(ffn_post.cpu())
                    ffn_comb_loc.append(ffn_comb.cpu())

                    xffn = rms_norm(
                        torch.sum(attn_pre.unsqueeze(-1) * x.float(), dim=2).to(x.dtype),
                        ld["ffn_norm"],
                        cfg.norm_eps,
                    )[0]
                    xffn_f = xffn.float()
                    xf_max = torch.maximum(xf_max, xffn_f.abs().amax(dim=0))
                    H_ffn_loc.addmm_(xffn_f.T, xffn_f)

                    sg = linear_official(xffn, ld["sh_w1"], True).float()
                    su = linear_official(xffn, ld["sh_w3"], True).float()
                    sh_u_max = torch.maximum(sh_u_max, su.abs().amax(dim=0))
                    if cfg.swiglu_limit > 0:
                        su = torch.clamp(su, min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
                        sg = torch.clamp(sg, max=cfg.swiglu_limit)
                    sh_mid = (F.silu(sg) * su).to(torch.bfloat16)
                    sh_m_max = torch.maximum(sh_m_max, sh_mid.float().abs().amax(dim=0))
                    shared_out_loc.append(linear_official(sh_mid, ld["sh_w2"], True).float().cpu())

                    if rank == 0 and b < 4:
                        probe_xa.append(xa0.cpu())
                        probe_qr.append(qr0.cpu())
                        probe_xf.append(xffn.cpu())
                        probe_shm.append(sh_mid.cpu())

                    scores = F.linear(xffn_f, ld["gate_w"]) / cfg.gate_temp
                    scores = F.softplus(scores).sqrt()
                    idxs = (scores + ld["gate_b"]).topk(cfg.n_activated_experts, dim=-1)[1]
                    wts = scores.gather(1, idxs)
                    wts = (wts / (wts.sum(dim=-1, keepdim=True) + 1e-20)) * cfg.route_scale

                    t0_i = tok_base + b * seq_len
                    t1_i = t0_i + seq_len
                    x_shm[t0_i:t1_i] = xffn.cpu().view(torch.uint16).numpy()
                    xq_shm[t0_i:t1_i] = (
                        quantize_activation_npu(xffn, 32, fp4=False)
                        .cpu()
                        .view(torch.uint16)
                        .numpy()
                    )
                    idx_shm[t0_i:t1_i] = idxs.cpu().numpy().astype(np.int32)
                    wt_shm[t0_i:t1_i] = wts.cpu().numpy().astype(np.float32)

                t_attn = time.monotonic() - tl0 - t_eng

                xa_max_c = xa_max.cpu()
                qr_max_c = qr_max.cpu()
                xf_max_c = xf_max.cpu()
                shm_max_c = sh_m_max.cpu()
                shu_max_c = sh_u_max.cpu()
                H_ffn_c = H_ffn_loc.cpu()
                dist.all_reduce(xa_max_c, op=dist.ReduceOp.MAX)
                dist.all_reduce(qr_max_c, op=dist.ReduceOp.MAX)
                dist.all_reduce(xf_max_c, op=dist.ReduceOp.MAX)
                dist.all_reduce(shm_max_c, op=dist.ReduceOp.MAX)
                dist.all_reduce(shu_max_c, op=dist.ReduceOp.MAX)
                dist.all_reduce(H_ffn_c, op=dist.ReduceOp.SUM)

                s_ffn_vec = torch.ones(5120, dtype=torch.float32)
                sq_path = sq_dir / f"L{layer_id:02d}.npz"
                is_w8 = layer_id in w8_set
                dest = prep_dir / f"L{layer_id:02d}_r{rank}"
                marker = dest / "done.json"
                already_quantized = marker.exists()

                if rank == 0:
                    sq_dir.mkdir(parents=True, exist_ok=True)
                    if already_quantized and sq_path.exists():
                        s_ffn_vec = torch.from_numpy(np.load(sq_path)["s_ffn"].astype(np.float32))
                    else:
                        pxa = torch.cat(probe_xa, dim=0).to(dev)
                        pqr = torch.cat(probe_qr, dim=0).to(dev)
                        pxf = torch.cat(probe_xf, dim=0).to(dev)
                        pshm = torch.cat(probe_shm, dim=0).to(dev)

                        q_wlist = [ld["wq_b"]] + ([ld["idx_wqb"]] if is_idx_src else [])
                        s_q, a_q, r_q, e_q, g_q = fit_smoothquant_scale(
                            pqr, qr_max_c.to(dev), q_wlist
                        )

                        if layer_id in kv_src_set:
                            s_attn, a_attn, r_attn, e_attn, g_attn = (
                                torch.ones(5120, device=dev),
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                            )
                        else:
                            s_attn, a_attn, r_attn, e_attn, g_attn = fit_smoothquant_scale(
                                pxa, xa_max_c.to(dev), [ld["wkv"], ld["wq_a"]]
                            )

                        s_shw, a_shw, r_shw, e_shw, g_shw = fit_shared_swiglu_sq(
                            pshm, shm_max_c.to(dev), shu_max_c.to(dev), ld["sh_w2"]
                        )

                        if is_w8:
                            s_ffn, a_ffn, g_ffn = torch.ones(5120, device=dev), 0.0, 0.0
                        else:
                            w4_samps = []
                            for se in (0, 96, 192, 288):
                                w4_samps.append(
                                    read_t(f"{prefix}.ffn.experts.{se}.w1.weight", torch.bfloat16)
                                )
                                w4_samps.append(
                                    read_t(f"{prefix}.ffn.experts.{se}.w3.weight", torch.bfloat16)
                                )
                            s_ffn, a_ffn, g_ffn = fit_ffn_w4a8_sq(
                                pxf, xf_max_c.to(dev), w4_samps, [ld["sh_w1"], ld["sh_w3"]]
                            )
                            del w4_samps
                        s_ffn_vec = s_ffn.cpu()

                        np.savez(
                            sq_path,
                            s_q=s_q.cpu().numpy().astype(np.float32),
                            s_attn=s_attn.cpu().numpy().astype(np.float32),
                            s_shw=s_shw.cpu().numpy().astype(np.float32),
                            s_ffn=s_ffn_vec.numpy().astype(np.float32),
                            meta=json.dumps(
                                dict(
                                    layer=layer_id,
                                    q=dict(
                                        a=a_q,
                                        raw=round(r_q, 6),
                                        sq=round(e_q, 6),
                                        gain=round(g_q, 1),
                                    ),
                                    attn=dict(
                                        a=a_attn,
                                        raw=round(r_attn, 6),
                                        sq=round(e_attn, 6),
                                        gain=round(g_attn, 1),
                                    ),
                                    shw=dict(
                                        a=a_shw,
                                        raw=round(r_shw, 6),
                                        sq=round(e_shw, 6),
                                        gain=round(g_shw, 1),
                                        safe_ch=int((shu_max_c <= SWIGLU_SAFE_U_MAX).sum().item()),
                                    ),
                                    ffn=dict(a=a_ffn, gain=round(g_ffn, 1)),
                                )
                            ),
                        )
                        del pxa, pqr, pxf, pshm

                dist.broadcast(s_ffn_vec, src=0)
                s_ffn_np = s_ffn_vec.numpy().astype(np.float32)
                has_ffn_sq = bool(np.any(s_ffn_np != 1.0))

                del ld
                torch.npu.empty_cache()
                dist.barrier()

                H_ffn_np = H_ffn_c.numpy()
                if has_ffn_sq:
                    inv_sf = 1.0 / s_ffn_np
                    H_ffn_scaled = (H_ffn_np * inv_sf[:, None]) * inv_sf[None, :]
                else:
                    H_ffn_scaled = H_ffn_np

                e0, e1 = rank * 48, (rank + 1) * 48
                idx_flat = idx_shm.reshape(-1)
                tok_ids_all = np.repeat(np.arange(n_tot, dtype=np.int32), 6)
                wt_flat = wt_shm.reshape(-1)
                loc_mask = (idx_flat >= e0) & (idx_flat < e1)
                loc_exp = idx_flat[loc_mask] - e0
                loc_tok = tok_ids_all[loc_mask]
                loc_wt = wt_flat[loc_mask]
                order = np.argsort(loc_exp, kind="mergesort")
                loc_exp = loc_exp[order]
                loc_tok = loc_tok[order]
                loc_wt = loc_wt[order]
                bounds = np.searchsorted(loc_exp, np.arange(49))

                y_loc = torch.zeros((n_tot, cfg.dim), dtype=torch.float32, device=dev)
                up_qs = [None] * 48
                up_ss = [None] * 48
                dn_qs = [None] * 48
                dn_ss = [None] * 48
                nmse_up_list = [0.0] * 48
                nmse_dn_list = [0.0] * 48
                swsq_cnt = 0
                cd_iters_up = 1 if (layer_id in cd_layers and not is_w8) else 0
                s_ffn_dev = torch.from_numpy(s_ffn_np).to(dev) if has_ffn_sq else None

                futures = []
                for local_e in range(48):
                    e = e0 + local_e
                    ep = f"{prefix}.ffn.experts.{e}"
                    w1_np = ckpt.matrix(f"{ep}.w1.weight")
                    w3_np = ckpt.matrix(f"{ep}.w3.weight")
                    w2_np = ckpt.matrix(f"{ep}.w2.weight")

                    p0, p1 = int(bounds[local_e]), int(bounds[local_e + 1])
                    n_e = p1 - p0
                    w1_dev = torch.from_numpy(w1_np).to(device=dev, dtype=torch.bfloat16)
                    w3_dev = torch.from_numpy(w3_np).to(device=dev, dtype=torch.bfloat16)
                    w2_dev = torch.from_numpy(w2_np).to(device=dev, dtype=torch.bfloat16)

                    H_up_e_np = None
                    H_dn_e_np = None
                    mid_ch_max = None
                    u_ch_max = None
                    mid_probe_t = None
                    mean_wt2 = 1.0

                    if n_e > 0:
                        tids = loc_tok[p0:p1]
                        wts_np = loc_wt[p0:p1]
                        mean_wt2 = float(np.mean(wts_np * wts_np))
                        wts_e = torch.from_numpy(wts_np).to(dev).unsqueeze(-1)
                        need_h = (not already_quantized) and (not is_w8)
                        H_up_acc = (
                            torch.zeros((5120, 5120), dtype=torch.float32, device=dev)
                            if need_h
                            else None
                        )
                        H_dn_acc = (
                            torch.zeros((2304, 2304), dtype=torch.float32, device=dev)
                            if need_h
                            else None
                        )
                        if need_h:
                            mid_ch_max = torch.zeros(2304, dtype=torch.float32, device=dev)
                            u_ch_max = torch.zeros(2304, dtype=torch.float32, device=dev)

                        for c0 in range(0, n_e, 8192):
                            c1 = min(c0 + 8192, n_e)
                            ctids = tids[c0:c1]
                            ctids_dev = torch.from_numpy(ctids).to(dev, dtype=torch.long)
                            xe_q = torch.from_numpy(xq_shm[ctids]).view(torch.bfloat16).to(dev)
                            gate = F.linear(xe_q, w1_dev).float()
                            up = F.linear(xe_q, w3_dev).float()
                            if need_h:
                                u_ch_max = torch.maximum(u_ch_max, up.abs().amax(dim=0))
                            if cfg.swiglu_limit > 0:
                                up = torch.clamp(up, min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
                                gate = torch.clamp(gate, max=cfg.swiglu_limit)
                            mid_unw = (F.silu(gate) * up).to(torch.bfloat16)
                            wt_c = wts_e[c0:c1]
                            mid_w = (mid_unw.float() * wt_c).to(torch.bfloat16)
                            mid_q = quantize_activation_npu(mid_w, 32, fp4=False)
                            out = F.linear(mid_q, w2_dev).float()
                            y_loc.index_add_(0, ctids_dev, out)

                            if H_up_acc is not None:
                                xe_raw = (
                                    torch.from_numpy(x_shm[ctids])
                                    .view(torch.bfloat16)
                                    .to(dev)
                                    .float()
                                )
                                if has_ffn_sq:
                                    xe_raw = xe_raw / s_ffn_dev[None, :]
                                xe_w = xe_raw * wt_c
                                H_up_acc.addmm_(xe_w.T, xe_w)
                                mf = mid_unw.float()
                                mf_w = mf * wt_c
                                H_dn_acc.addmm_(mf_w.T, mf_w)
                                mid_ch_max = torch.maximum(mid_ch_max, mf.abs().amax(dim=0))
                                if mid_probe_t is None:
                                    mid_probe_t = mf[: min(mf.shape[0], 1024)].clone()

                        if H_up_acc is not None:
                            H_up_e_np = H_up_acc.cpu().numpy()
                            H_dn_e_np = H_dn_acc.cpu().numpy()
                            del H_up_acc, H_dn_acc

                    if not already_quantized:
                        if is_w8:
                            w_up_dev = torch.cat([w1_dev, w3_dev], dim=0)
                            s_up_ue = np.concatenate(
                                [ckpt.read(f"{ep}.w1.scale"), ckpt.read(f"{ep}.w3.scale")], axis=0
                            )
                            q_u, enc_u, nm_u = quantize_exact_w8_device(w_up_dev, s_up_ue)
                            del w_up_dev
                            up_qs[local_e], up_ss[local_e], nmse_up_list[local_e] = (
                                q_u,
                                enc_u,
                                nm_u,
                            )

                            q_d, enc_d, nm_d = quantize_exact_w8_device(
                                w2_dev, ckpt.read(f"{ep}.w2.scale")
                            )
                            dn_qs[local_e], dn_ss[local_e], nmse_dn_list[local_e] = (
                                q_d,
                                enc_d,
                                nm_d,
                            )
                        else:
                            s_sw_np = np.ones(2304, dtype=np.float32)
                            if mid_probe_t is not None and mid_probe_t.shape[0] >= 128:
                                s_sw_t, a_sw, _ = fit_expert_swiglu_sq(
                                    mid_probe_t, w2_dev, mid_ch_max, u_ch_max
                                )
                                if a_sw > 0.0:
                                    s_sw_np = s_sw_t.cpu().numpy().astype(np.float32)
                                    swsq_cnt += 1

                            w1_q_in = w1_np * s_ffn_np[None, :] if has_ffn_sq else w1_np
                            w3_q_in = w3_np * s_ffn_np[None, :] if has_ffn_sq else w3_np
                            if np.any(s_sw_np != 1.0):
                                w3_q_in = w3_q_in / s_sw_np[:, None]
                                w2_q_in = w2_np * s_sw_np[None, :]
                                inv_sw = 1.0 / s_sw_np
                                H_dn_use = (H_dn_e_np * inv_sw[:, None]) * inv_sw[None, :]
                            else:
                                w2_q_in = w2_np
                                H_dn_use = H_dn_e_np

                            w_up_in = np.ascontiguousarray(
                                np.concatenate([w1_q_in, w3_q_in], axis=0), dtype=np.float32
                            )
                            w2_q_in = np.ascontiguousarray(w2_q_in, dtype=np.float32)
                            prior_w = (max(512.0, 0.12 * float(n_e)) * max(mean_wt2, 1e-4)) / float(
                                n_tot
                            )
                            if H_up_e_np is not None and n_e >= 64:
                                H_up_use = np.ascontiguousarray(
                                    H_up_e_np + prior_w * H_ffn_scaled, dtype=np.float32
                                )
                            else:
                                H_up_use = np.ascontiguousarray(H_ffn_scaled, dtype=np.float32)

                            has_dn_h = (H_dn_use is not None) and (n_e >= 64)
                            damp_dn = 0.05 if n_e >= 2048 else (0.10 if n_e >= 512 else 0.20)
                            in_path = str(ipc_dir / f"L{layer_id:02d}_e{local_e:02d}_in.bin")
                            out_path = str(ipc_dir / f"L{layer_id:02d}_e{local_e:02d}_out.bin")
                            with open(in_path, "wb") as f:
                                w_up_in.tofile(f)
                                H_up_use.tofile(f)
                                w2_q_in.tofile(f)
                                if has_dn_h:
                                    np.ascontiguousarray(H_dn_use, dtype=np.float32).tofile(f)
                            futures.append(
                                cpu_pool.submit(
                                    _cpu_gptq_expert_job,
                                    (local_e, in_path, out_path, has_dn_h, damp_dn),
                                )
                            )

                    del w1_dev, w3_dev, w2_dev, mid_probe_t

                if futures:
                    for fut in as_completed(futures):
                        local_e, in_path, out_path, has_dn_h, damp_u, damp_d_val = fut.result()
                        with open(out_path, "rb") as f:
                            q_u = np.fromfile(f, dtype=np.int8, count=4608 * 5120).reshape(
                                4608, 5120
                            )
                            s_u = np.fromfile(f, dtype=np.float32, count=4608)
                            perm_u = np.fromfile(f, dtype=np.int32, count=5120)
                            q_d = np.fromfile(f, dtype=np.int8, count=5120 * 2304).reshape(
                                5120, 2304
                            )
                            s_d = np.fromfile(f, dtype=np.float32, count=5120)
                            perm_d = np.fromfile(f, dtype=np.int32, count=2304)
                        with open(in_path, "rb") as f:
                            w_up = np.fromfile(f, dtype=np.float32, count=4608 * 5120).reshape(
                                4608, 5120
                            )
                            h_up = np.fromfile(f, dtype=np.float32, count=5120 * 5120).reshape(
                                5120, 5120
                            )
                            w_dn = np.fromfile(f, dtype=np.float32, count=5120 * 2304).reshape(
                                5120, 2304
                            )
                            h_dn = (
                                np.fromfile(f, dtype=np.float32, count=2304 * 2304).reshape(
                                    2304, 2304
                                )
                                if has_dn_h
                                else None
                            )
                        os.unlink(in_path)
                        os.unlink(out_path)

                        q_u, s_u, nm_u = npu_refine_h_scale(
                            w_up,
                            h_up,
                            q_u,
                            perm_u,
                            damp_u,
                            dev,
                            main_gptq.cd_pass,
                            cd_iters=cd_iters_up,
                            cd_threads=4,
                        )
                        up_qs[local_e], up_ss[local_e], nmse_up_list[local_e] = q_u, s_u, nm_u

                        if has_dn_h and h_dn is not None:
                            q_d, s_d, nm_d = npu_refine_h_scale(
                                w_dn,
                                h_dn,
                                q_d,
                                perm_d,
                                damp_d_val,
                                dev,
                                main_gptq.cd_pass,
                                cd_iters=0,
                                cd_threads=4,
                            )
                        else:
                            err_d = float(
                                np.square(
                                    w_dn - q_d.astype(np.float32) * s_d[:, None], dtype=np.float64
                                ).sum()
                            )
                            nm_d = err_d / max(
                                float(np.square(w_dn, dtype=np.float64).sum()), 1e-30
                            )
                        dn_qs[local_e], dn_ss[local_e], nmse_dn_list[local_e] = q_d, s_d, nm_d

                if not already_quantized:
                    dest.mkdir(parents=True, exist_ok=True)
                    if is_w8:
                        np.save(dest / "up_w8.npy", np.stack(up_qs, axis=0))
                        np.save(dest / "up_scale_bf16.npy", np.stack(up_ss, axis=0))
                        np.save(dest / "down_w8.npy", np.stack(dn_qs, axis=0))
                        np.save(dest / "down_scale_bf16.npy", np.stack(dn_ss, axis=0))
                    else:
                        nz, sc, ast = pack_w4_nz_bank(up_qs, up_ss)
                        np.save(dest / "up_nz.npy", nz)
                        np.save(dest / "up_scale.npy", sc)
                        np.save(dest / "up_assist.npy", ast)
                        nz, sc, ast = pack_w4_nz_bank(dn_qs, dn_ss)
                        np.save(dest / "down_nz.npy", nz)
                        np.save(dest / "down_scale.npy", sc)
                        np.save(dest / "down_assist.npy", ast)
                    rec = dict(
                        layer=layer_id,
                        rank=rank,
                        is_w8=is_w8,
                        swsq_experts=swsq_cnt,
                        mean_nmse_up=float(np.mean(nmse_up_list)),
                        mean_nmse_down=float(np.mean(nmse_dn_list)),
                    )
                    atomic_json(marker, rec)
                else:
                    rec = json.loads(marker.read_text())

                y_loc_cpu = y_loc.cpu()
                del y_loc
                y_routed_mine_t = None
                for dst in range(world_size):
                    sl = y_loc_cpu[dst * n_loc_tok : (dst + 1) * n_loc_tok].contiguous()
                    dist.reduce(sl, dst=dst, op=dist.ReduceOp.SUM)
                    if dst == rank:
                        y_routed_mine_t = sl
                del y_loc_cpu

                y_shared_mine = torch.cat(shared_out_loc, dim=0)
                y_tot_mine = (
                    (y_routed_mine_t + y_shared_mine)
                    .to(torch.bfloat16)
                    .reshape(n_loc_seqs, seq_len, cfg.dim)
                )
                del y_routed_mine_t, y_shared_mine

                for b in range(n_loc_seqs):
                    yb = y_tot_mine[b : b + 1].to(dev)
                    res = ffn_res_loc[b].to(dev)
                    h_list[b] = (
                        (
                            ffn_post_loc[b].to(dev).unsqueeze(-1) * yb.unsqueeze(-2)
                            + torch.sum(
                                ffn_comb_loc[b].to(dev).unsqueeze(-1) * res.unsqueeze(-2), dim=2
                            )
                        )
                        .to(torch.bfloat16)
                        .cpu()
                    )

                dist.barrier()
                if rank == 0:
                    sq_meta = json.loads(str(np.load(sq_path)["meta"]))
                    mode = (
                        "W8" if is_w8 else f"GPTQ(cd={cd_iters_up},swSQ={rec['swsq_experts']}/48)"
                    )
                    eng_s = f"eng {t_eng:.1f}s, " if t_eng > 0 else ""
                    print(
                        f"L{layer_id:02d}/39 [{mode:20s}] {time.monotonic() - tl0:5.1f}s "
                        f"({eng_s}attn {t_attn:4.1f}s, tot {time.monotonic() - t_start:6.1f}s) | "
                        f"H-NMSE up={rec['mean_nmse_up']:.5f} dn={rec['mean_nmse_down']:.5f} | "
                        f"SQ gain: q={sq_meta['q']['gain']:+.0f}% attn={sq_meta['attn']['gain']:+.0f}% "
                        f"shw={sq_meta['shw']['gain']:+.0f}% ffn={sq_meta['ffn']['gain']:+.0f}%",
                        flush=True,
                    )
    finally:
        cpu_pool.shutdown(wait=True)
        shutil.rmtree(ipc_dir, ignore_errors=True)
        ckpt.close()
        for s in (shm_x, shm_xq, shm_idx, shm_wt):
            s.close()
        dist.destroy_process_group()


def main():
    import torch.multiprocessing as mp

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--official-dir", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--w8-layers", type=int, nargs="*", default=list(W8_DEFAULT))
    ap.add_argument("--cd-layers", type=int, nargs="*", default=list(CD_DEFAULT))
    ap.add_argument("--num-seqs", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--workers-per-rank", type=int, default=24)
    ap.add_argument("--port", type=int, default=29517)
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "native-cache"
    Decoder(cache_dir)
    SSZQuantizer(cache_dir)
    GPTQQuantizer(cache_dir, n_threads=1)

    n_tot = args.num_seqs * args.seq_len
    world_size = 8
    shms = {
        "x": shared_memory.SharedMemory(create=True, size=n_tot * 5120 * 2),
        "xq": shared_memory.SharedMemory(create=True, size=n_tot * 5120 * 2),
        "idx": shared_memory.SharedMemory(create=True, size=n_tot * 6 * 4),
        "wt": shared_memory.SharedMemory(create=True, size=n_tot * 6 * 4),
    }
    shm_names = {k: v.name for k, v in shms.items()}
    args_dict = dict(
        official_dir=str(args.official_dir.resolve()),
        corpus=str(args.corpus.resolve()),
        out_dir=str(out_dir),
        w8_layers=args.w8_layers,
        cd_layers=args.cd_layers,
        num_seqs=args.num_seqs,
        seq_len=args.seq_len,
        workers_per_rank=args.workers_per_rank,
        port=args.port,
    )
    try:
        mp.spawn(_worker_fn, args=(world_size, args_dict, shm_names), nprocs=world_size, join=True)
    finally:
        for s in shms.values():
            try:
                s.close()
                s.unlink()
            except Exception:
                pass
        shutil.rmtree(out_dir / "ipc", ignore_errors=True)


if __name__ == "__main__":
    main()
