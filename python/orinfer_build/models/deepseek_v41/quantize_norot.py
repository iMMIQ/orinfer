"""Reproducible unrotated native-grid quantization & release builder for DSv4.1 on Ascend 910B.

Quantizes DeepSeek-V4.1-Flash-1017 directly from the official MXFP4+FP8 checkpoint in its
original unrotated basis (no global QuaRot32, no ln_fuse, no AWQ column scaling):
  - L00..L37 routed experts: per-channel SSZ signed INT4 [-8, 7] in Ascend Transposed-NZ layout.
  - L38..L39 routed experts: exact-grid lossless INT8 ND with BF16 scales.
  - Shared KV-source layers (L02, L08, L14, L20): BF16 attn.wq_a & attn.wkv with tuned
    small-M row_selections; remaining 36 layers keep unrotated W8A8 + fused norm_quant.
  - Engram basis restoration: 20-AIC persistent BF16->FP32 cast (engram.cast_fp32).
  - DSpark draft (mtp.0/1/2): exact-grid lossless full-W8 routed experts (or ssz_up_w8_down / ssz_w4).
  - Unrotated full-precision / W8A8 anchors: embed, 3-plane BF16 head, norms, mHC, wo_b,
    compressor, indexer, shared_experts, engram.wkv, mtp.0.main_proj, vision aligner.w2 & image_*.
  - Host Engram tables (weights/host/, 206 GiB) and weights/dspark/embedding.safetensors are
    basis-invariant and hardlinked byte-for-byte from the base model.

Subcommands:
  compile-kernels  Compile required TileLang-Ascend AOT kernels (engram.cast_fp32, tuned BF16
                   wq_a/wkv + row_selections, draft A8 gather/swiglu) and sync source_sha256
                   on touched ops. Run inside tools/environments/ascend/run.sh --without-devices.
  export           Prepare target & draft expert banks in parallel and export physical safetensors
                   plus the matching (optionally compact-RoPE) A2 execution package. Run on CPU.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import ctypes
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np

from orinfer_build.common.files import atomic_json
from orinfer_build.common.native_c import load_shared_c
from orinfer_build.common.shared_json import decode
from orinfer_build.models.deepseek_v41.export_release import TensorRelease
from orinfer_build.models.deepseek_v41.floating import bf16_parts
from orinfer_build.models.deepseek_v41.official import Checkpoint, Decoder
from orinfer_build.models.deepseek_v41.weight_layout import transposed_nz_blocks
from orinfer_build.weights.safetensors_source import local_header

KV_SOURCE_LAYERS = (2, 8, 14, 20)
SYNC_OPS = {
    "memory.gather_quantized_rows",
    "model_ops.swiglu_packed_quant",
    "engram.gate",
    "dense.projection",
}


class SSZQuantizer:
    """Hash-cached OpenMP C11 per-channel SSZ signed INT4 [-8, 7] quantizer."""

    def __init__(self, cache_dir: Path):
        self.library = load_shared_c(
            Path(__file__).with_name("ssz_w4.c"),
            Path(cache_dir),
            prefix="ssz_w4",
            openmp=True,
            math_lib=True,
        )
        self.call = self.library.orinfer_ssz_w4_rows
        self.call.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.call.restype = None

    def __call__(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        w = np.ascontiguousarray(weights, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError("SSZ quantization expects a 2D weight matrix")
        n, k = w.shape
        q = np.empty((n, k), dtype=np.int8)
        s = np.empty(n, dtype=np.float32)
        err = ctypes.c_double(0.0)
        energy = ctypes.c_double(0.0)
        self.call(
            w.ctypes.data,
            n,
            k,
            q.ctypes.data,
            s.ctypes.data,
            ctypes.byref(err),
            ctypes.byref(energy),
        )
        return q, s, float(err.value / max(energy.value, 1e-30))


class GPTQQuantizer:
    """Hash-cached OpenMP C11 Act-Order Blocked GPTQ + H-optimal scale INT4 [-8, 7] quantizer."""

    def __init__(self, cache_dir: Path, n_threads: int | None = None):
        self.n_threads = max(1, int(os.cpu_count() or 1) if n_threads is None else int(n_threads))
        self.library = load_shared_c(
            Path(__file__).with_name("gptq_w4.c"),
            Path(cache_dir),
            prefix="gptq_w4",
            openmp=True,
            math_lib=True,
        )
        self.hdiag_ssz = self.library.orinfer_hdiag_ssz_scale
        self.hdiag_ssz.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.hdiag_ssz.restype = None

        self.diag_block = self.library.orinfer_gptq_diag_block
        self.diag_block.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.diag_block.restype = None

        self.cd_pass = self.library.orinfer_gptq_cd_pass
        self.cd_pass.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.cd_pass.restype = None

    @staticmethod
    def factor_hessian(
        hessian: np.ndarray, damp_frac: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute act-order permutation and upper Cholesky of damped H^{-1}."""
        h = np.array(hessian, dtype=np.float64, copy=True)
        k = h.shape[0]
        if h.shape != (k, k):
            raise ValueError("Hessian must be square")
        dead = np.diag(h) <= 0.0
        h[dead, dead] = 1.0
        perm = np.argsort(-np.diag(h)).astype(np.int32)
        invperm = np.argsort(perm).astype(np.int32)
        hp = h[np.ix_(perm, perm)]
        damp = max(damp_frac * float(np.mean(np.diag(hp))), 1e-6)
        idx = np.arange(k)
        hinv_ch = None
        for mult in (1.0, 2.0, 5.0, 10.0, 50.0, 200.0):
            try:
                hd = hp.copy()
                hd[idx, idx] += damp * mult
                chol = np.linalg.cholesky(hd)
                inv_chol = np.linalg.solve(chol, np.eye(k, dtype=np.float64))
                hinv = inv_chol.T @ inv_chol
                hinv_ch = np.linalg.cholesky(hinv).T
                break
            except np.linalg.LinAlgError:
                continue
        if hinv_ch is None:
            hinv_ch = np.diag(1.0 / np.sqrt(np.maximum(np.diag(hp), 1e-6)))
        h_damped = h.copy()
        h_damped[idx, idx] += damp
        return (
            perm,
            invperm,
            np.ascontiguousarray(hinv_ch, dtype=np.float32),
            np.ascontiguousarray(h_damped, dtype=np.float32),
        )

    def blocked_gptq(
        self,
        weights: np.ndarray,
        hinv_ch: np.ndarray,
        perm: np.ndarray,
        invperm: np.ndarray,
        hd_p: np.ndarray,
        *,
        block_size: int = 128,
        update_fn=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run C11 Hessian-diagonal SSZ init + blocked Act-Order GPTQ elimination."""
        w = np.ascontiguousarray(weights, dtype=np.float32)
        n, k = w.shape
        wp = np.ascontiguousarray(w[:, perm], dtype=np.float32)
        hd_p = np.ascontiguousarray(hd_p, dtype=np.float32)
        hinv_ch = np.ascontiguousarray(hinv_ch, dtype=np.float32)
        s = np.empty(n, dtype=np.float32)
        self.hdiag_ssz(wp.ctypes.data, hd_p.ctypes.data, n, k, self.n_threads, s.ctypes.data)

        qp = np.empty((n, k), dtype=np.int8)
        for i1 in range(0, k, block_size):
            i2 = min(i1 + block_size, k)
            cnt = i2 - i1
            wb = np.ascontiguousarray(wp[:, i1:i2], dtype=np.float32)
            hb = np.ascontiguousarray(hinv_ch[i1:i2, i1:i2], dtype=np.float32)
            qb = np.empty((n, cnt), dtype=np.int8)
            eb = np.empty((n, cnt), dtype=np.float32)
            self.diag_block(
                wb.ctypes.data,
                hb.ctypes.data,
                s.ctypes.data,
                n,
                cnt,
                self.n_threads,
                qb.ctypes.data,
                eb.ctypes.data,
            )
            qp[:, i1:i2] = qb
            if i2 < k:
                if update_fn is not None:
                    update_fn(wp, eb, i1, i2)
                else:
                    wp[:, i2:] -= eb @ hinv_ch[i1:i2, i2:]

        return np.ascontiguousarray(qp[:, invperm], dtype=np.int8), s

    def quantize_cpu(
        self,
        weights: np.ndarray,
        hessian: np.ndarray,
        *,
        damp_frac: float = 0.05,
        block_size: int = 128,
        cd_iters: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Pure-CPU Act-Order Blocked GPTQ + exact H-optimal scale (+ optional CD refinement)."""
        w = np.ascontiguousarray(weights, dtype=np.float32)
        n, k = w.shape
        perm, invperm, hinv_ch, h_damped = self.factor_hessian(hessian, damp_frac=damp_frac)
        hd_p = np.ascontiguousarray(np.diag(h_damped)[perm], dtype=np.float32)
        q, s = self.blocked_gptq(w, hinv_ch, perm, invperm, hd_p, block_size=block_size)

        wh = w @ h_damped
        qf = q.astype(np.float32)
        qh = qf @ h_damped
        num = (wh * qf).sum(axis=1)
        den = np.maximum((qh * qf).sum(axis=1), 1e-30)
        s = np.maximum(num / den, 1e-30).astype(np.float32)

        if cd_iters > 0:
            hd_full = np.ascontiguousarray(np.diag(h_damped), dtype=np.float32)
            for _ in range(cd_iters):
                r_mat = np.ascontiguousarray(wh - s[:, None] * qh, dtype=np.float32)
                self.cd_pass(
                    h_damped.ctypes.data,
                    hd_full.ctypes.data,
                    perm.ctypes.data,
                    s.ctypes.data,
                    n,
                    k,
                    self.n_threads,
                    q.ctypes.data,
                    r_mat.ctypes.data,
                )
                qf = q.astype(np.float32)
                qh = qf @ h_damped
                num = (wh * qf).sum(axis=1)
                den = np.maximum((qh * qf).sum(axis=1), 1e-30)
                s = np.maximum(num / den, 1e-30).astype(np.float32)

        err_h = float(((w - s[:, None] * qf) * (wh - s[:, None] * qh)).sum(dtype=np.float64))
        en_h = max(float((w * wh).sum(dtype=np.float64)), 1e-30)
        return q, s, max(err_h, 0.0) / en_h


def to_bf16_u16(values: np.ndarray) -> np.ndarray:
    """IEEE round-to-nearest-even FP32 -> BF16 stored as little-endian uint16."""
    v = np.ascontiguousarray(values, dtype="<f4")
    bits = v.view("<u4")
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")


def bf16_round_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype="<f4")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Scales must be finite and positive")
    enc = to_bf16_u16(values)
    dec = (enc.astype("<u4") << 16).view("<f4")
    if np.any(dec <= 0):
        raise ValueError("BF16 scale underflow")
    return enc, dec


def quantize_per_channel_i8_f32(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-channel symmetric INT8 [-127, 127] with FP32 scale [out, 1]."""
    w = np.asarray(weights, dtype=np.float32)
    scale = np.maximum(np.abs(w).max(axis=1) / 127.0, 1e-30).astype("<f4")
    q = np.rint(w / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale[:, None]


def quantize_exact_w8(
    weights: np.ndarray, ue8m0_scales: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-output-channel exact-grid lossless INT8 with BF16 scales.

    When max|w| <= 127 * (0.5 * s_min), step = 0.5 * s_min is an exact power of two in BF16 and
    recovers every official MXFP4 E2M1 weight with 0.0 error. On wider dynamic-range rows, picks
    the minimum-MSE BF16 scale among {0.5 * s_min, s_min, amax / 127}.
    """
    w = np.asarray(weights, dtype=np.float32)
    s_min = np.asarray(ue8m0_scales, dtype=np.float32).min(axis=1)
    amax = np.abs(w).max(axis=1).clip(1e-30)

    enc_a, dec_a = bf16_round_scale(amax / 127.0)
    qa = np.rint(w / dec_a[:, None]).clip(-127, 127).astype(np.int8)
    err_a = np.square(w - qa.astype(np.float32) * dec_a[:, None], dtype=np.float64).sum(axis=1)

    enc_b, dec_b = bf16_round_scale(np.maximum(0.5 * s_min, 1e-30))
    qb = np.rint(w / dec_b[:, None]).clip(-127, 127).astype(np.int8)
    err_b = np.square(w - qb.astype(np.float32) * dec_b[:, None], dtype=np.float64).sum(axis=1)

    enc_c, dec_c = bf16_round_scale(np.maximum(s_min, 1e-30))
    qc = np.rint(w / dec_c[:, None]).clip(-127, 127).astype(np.int8)
    err_c = np.square(w - qc.astype(np.float32) * dec_c[:, None], dtype=np.float64).sum(axis=1)

    best_q, best_enc, best_err = qa.copy(), enc_a.copy(), err_a.copy()
    use_c = err_c < best_err
    best_q[use_c], best_enc[use_c], best_err[use_c] = qc[use_c], enc_c[use_c], err_c[use_c]
    use_b = err_b <= best_err
    best_q[use_b], best_enc[use_b], best_err[use_b] = qb[use_b], enc_b[use_b], err_b[use_b]

    energy = max(float(np.square(w, dtype=np.float64).sum()), 1e-30)
    return best_q, best_enc, float(best_err.sum() / energy)


def pack_w4_nz_bank(
    q_list: list[np.ndarray], s_list: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack per-expert signed INT4 [-8, 7] weights into Ascend Transposed-NZ banks.

    Returns:
      packed_nz: [E, K//64, N//16, 16, 32] int8 (two's-complement nibbles viewed as I8)
      scale_u64: [E, 1, N] uint64 (low 32 bits = FP32 scale)
      assist:    [E, N] float32 (= 8.0 * scale * sum_k(q))
    """
    e_count = len(q_list)
    n, k = q_list[0].shape
    if k % 64 or n % 16:
        raise ValueError("Transposed-NZ W4 banks require K%64=0 and N%16=0")
    packed_nd = np.empty((e_count, k, n // 2), dtype=np.uint8)
    scale_u64 = np.zeros((e_count, 1, n), dtype="<u8")
    assist = np.empty((e_count, n), dtype="<f4")
    for e, (q, s) in enumerate(zip(q_list, s_list, strict=True)):
        if q.shape != (n, k) or q.dtype != np.int8:
            raise ValueError("Expected int8 [N, K] expert matrix")
        s32 = np.asarray(s, dtype="<f4")
        scale_u64[e, 0, :] = s32.view("<u4").astype("<u8")
        assist[e, :] = (np.float32(8.0) * s32) * q.sum(axis=1, dtype=np.int32).astype("<f4")
        q_t = np.ascontiguousarray(q.T)
        lo = q_t[:, 0::2].view(np.uint8) & np.uint8(15)
        hi = q_t[:, 1::2].view(np.uint8) & np.uint8(15)
        packed_nd[e] = lo | (hi << np.uint8(4))
    nz_blocks = list(transposed_nz_blocks(packed_nd))
    packed_nz = np.stack(nz_blocks, axis=0).reshape(e_count, k // 64, n // 16, 16, 32)
    return packed_nz.view(np.int8), scale_u64, assist


def save_expert_bank(
    dest: Path,
    up_qs: list[np.ndarray],
    up_ss: list[np.ndarray],
    dn_qs: list[np.ndarray],
    dn_ss: list[np.ndarray],
    *,
    up_w8: bool,
    dn_w8: bool,
) -> None:
    """Persist quantized routed-expert up/down banks in W8 ND or W4 Transposed-NZ layout."""
    dest.mkdir(parents=True, exist_ok=True)
    for stem, qs, ss, is_w8 in (("up", up_qs, up_ss, up_w8), ("down", dn_qs, dn_ss, dn_w8)):
        if is_w8:
            np.save(dest / f"{stem}_w8.npy", np.stack(qs, axis=0))
            np.save(dest / f"{stem}_scale_bf16.npy", np.stack(ss, axis=0))
        else:
            nz, sc, ast = pack_w4_nz_bank(qs, ss)
            np.save(dest / f"{stem}_nz.npy", nz)
            np.save(dest / f"{stem}_scale.npy", sc)
            np.save(dest / f"{stem}_assist.npy", ast)


def _prepare_expert_group(
    ckpt: Checkpoint,
    ssz: SSZQuantizer | None,
    prefixes: list[str],
    *,
    up_w8: bool,
    dn_w8: bool,
):
    up_qs, up_ss, dn_qs, dn_ss = [], [], [], []
    nmse_up, nmse_dn = [], []
    for pref in prefixes:
        w_up = np.concatenate(
            [ckpt.matrix(f"{pref}.w1.weight"), ckpt.matrix(f"{pref}.w3.weight")], axis=0
        )
        if up_w8:
            s_up = np.concatenate(
                [ckpt.read(f"{pref}.w1.scale"), ckpt.read(f"{pref}.w3.scale")], axis=0
            )
            q, enc, nmse = quantize_exact_w8(w_up, s_up)
            up_qs.append(np.ascontiguousarray(q.T))
            up_ss.append(enc)
        else:
            q, sc, nmse = ssz(w_up)
            up_qs.append(q)
            up_ss.append(sc)
        nmse_up.append(nmse)

        w2 = ckpt.matrix(f"{pref}.w2.weight")
        if dn_w8:
            q, enc, nmse = quantize_exact_w8(w2, ckpt.read(f"{pref}.w2.scale"))
            dn_qs.append(np.ascontiguousarray(q.T))
            dn_ss.append(enc)
        else:
            q, sc, nmse = ssz(w2)
            dn_qs.append(q)
            dn_ss.append(sc)
        nmse_dn.append(nmse)
    return up_qs, up_ss, dn_qs, dn_ss, float(np.mean(nmse_up)), float(np.mean(nmse_dn))


def _prepare_target_bank(task):
    official_dir, layer, rank, is_w8, prep_dir, cache_dir = task
    dest = Path(prep_dir) / f"L{layer:02d}_r{rank}"
    marker = dest / "done.json"
    if marker.exists():
        rec = json.loads(marker.read_text())
        if rec.get("is_w8") == is_w8:
            return rec

    ckpt = Checkpoint(Path(official_dir), Decoder(Path(cache_dir)))
    ssz = None if is_w8 else SSZQuantizer(Path(cache_dir))
    try:
        prefixes = [f"layers.{layer}.ffn.experts.{rank * 48 + local}" for local in range(48)]
        up_qs, up_ss, dn_qs, dn_ss, mean_up, mean_dn = _prepare_expert_group(
            ckpt, ssz, prefixes, up_w8=is_w8, dn_w8=is_w8
        )
        save_expert_bank(dest, up_qs, up_ss, dn_qs, dn_ss, up_w8=is_w8, dn_w8=is_w8)
        rec = dict(
            layer=layer,
            rank=rank,
            is_w8=is_w8,
            mean_nmse_up=mean_up,
            mean_nmse_down=mean_dn,
        )
        atomic_json(marker, rec)
        return rec
    finally:
        ckpt.close()


def _prepare_draft_bank(task):
    official_dir, draft_idx, rank, up_w8, dn_w8, prep_dir, cache_dir = task
    dl = 40 + draft_idx
    dest = Path(prep_dir) / f"L{dl}_r{rank}"
    marker = dest / "done.json"
    if marker.exists():
        rec = json.loads(marker.read_text())
        if rec.get("up_w8") == up_w8 and rec.get("dn_w8") == dn_w8:
            return rec

    ckpt = Checkpoint(Path(official_dir), Decoder(Path(cache_dir)))
    ssz = None if (up_w8 and dn_w8) else SSZQuantizer(Path(cache_dir))
    try:
        prefixes = [f"mtp.{draft_idx}.ffn.experts.{rank * 16 + local}" for local in range(16)]
        up_qs, up_ss, dn_qs, dn_ss, mean_up, mean_dn = _prepare_expert_group(
            ckpt, ssz, prefixes, up_w8=up_w8, dn_w8=dn_w8
        )
        save_expert_bank(dest, up_qs, up_ss, dn_qs, dn_ss, up_w8=up_w8, dn_w8=dn_w8)
        rec = dict(
            layer=dl,
            rank=rank,
            up_w8=up_w8,
            dn_w8=dn_w8,
            mean_nmse_up=mean_up,
            mean_nmse_down=mean_dn,
        )
        atomic_json(marker, rec)
        return rec
    finally:
        ckpt.close()


class NoRotRelease(TensorRelease):
    """Full unrotated native-grid release with disk-backed expert banks."""

    def __init__(
        self,
        *args,
        official_dir: Path,
        prep_dir: Path,
        kernels: dict,
        remap: dict,
        w8_layers: list[int],
        bf16_qkv_layers: list[int],
        draft_prep_dir: Path | None,
        draft_mode: str | None,
        smoothquant_dir: Path | None = None,
        **kwargs,
    ):
        super().__init__(*args, overlay=False, **kwargs)
        self.official_dir = Path(official_dir)
        self.prep_dir = Path(prep_dir)
        self.kernels = kernels
        self.remap = remap or {}
        self.w8_layers = sorted(set(w8_layers))
        self.bf16_qkv_layers = sorted(set(bf16_qkv_layers))
        self.draft_prep_dir = Path(draft_prep_dir) if draft_prep_dir else None
        self.draft_mode = draft_mode
        self.smoothquant_dir = Path(smoothquant_dir) if smoothquant_dir else None
        self.file_replacements = {}

    def _load_layer_smoothquant(
        self, layer_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load per-layer calibrated SmoothQuant vectors (s_q, s_attn, s_shw, s_ffn)."""
        ones_q = np.ones(1280, dtype=np.float32)
        ones_d = np.ones(5120, dtype=np.float32)
        ones_sh = np.ones(2304, dtype=np.float32)
        if not self.smoothquant_dir:
            return ones_q, ones_d, ones_sh, ones_d
        path = self.smoothquant_dir / f"L{layer_id:02d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing SmoothQuant calibration file: {path}")
        with np.load(path) as data:
            s_q = np.asarray(data["s_q"], dtype=np.float32)
            s_attn = np.asarray(data["s_attn"], dtype=np.float32)
            s_shw = np.asarray(data["s_shw"], dtype=np.float32)
            s_ffn = np.asarray(data["s_ffn"], dtype=np.float32)
        if (
            s_q.shape != (1280,)
            or s_attn.shape != (5120,)
            or s_shw.shape != (2304,)
            or s_ffn.shape != (5120,)
        ):
            raise ValueError(f"Invalid SmoothQuant vector shape in {path}")
        for name, vec in (("s_q", s_q), ("s_attn", s_attn), ("s_shw", s_shw), ("s_ffn", s_ffn)):
            if not np.isfinite(vec).all() or np.any(vec <= 0.0):
                raise ValueError(f"Non-positive or non-finite {name} in {path}")
        if np.any(s_shw < 1.0 - 1e-6):
            raise ValueError(f"Shared SwiGLU SmoothQuant s_shw must be >= 1.0 in {path}")
        if (layer_id in self.bf16_qkv_layers or layer_id in KV_SOURCE_LAYERS) and np.any(
            s_attn != 1.0
        ):
            raise ValueError(f"s_attn must be unity on BF16/KV-source layer {layer_id}")
        if layer_id in self.w8_layers and np.any(s_ffn != 1.0):
            raise ValueError(f"s_ffn must be unity on exact-grid W8 layer {layer_id}")
        return s_q, s_attn, s_shw, s_ffn

    @staticmethod
    def _npy_info(path: Path):
        with path.open("rb") as f:
            ver = np.lib.format.read_magic(f)
            shape, fortran, dtype = (
                np.lib.format.read_array_header_1_0(f)
                if ver == (1, 0)
                else np.lib.format.read_array_header_2_0(f)
            )
            if fortran:
                raise ValueError(f"Fortran-order array unsupported: {path}")
            offset = f.tell()
        nbytes = int(np.prod(shape)) * dtype.itemsize
        return str(path), offset, nbytes, list(shape)

    def replace_from_npy(self, definition, npy_path: Path, *, suffix: str, dtype: str, group: str):
        original = definition["prepared_tensor"]["name"]
        replacement = original + suffix
        fpath, offset, nbytes, shape = self._npy_info(npy_path)
        if original in self.transforms:
            if self.transforms[original] != replacement:
                raise ValueError(f"Conflicting replacement for {original}")
        else:
            if replacement in self.index["weight_map"]:
                raise ValueError(f"Replacement already exists: {replacement}")
            self.transforms[original] = replacement
            self.file_replacements[replacement] = dict(
                file=fpath, offset=offset, bytes=nbytes, dtype=dtype, shape=shape, group=group
            )
        definition.update(
            bytes=nbytes,
            prepared_tensor=dict(name=replacement, dtype=dtype, shape=shape),
        )

    def _replace_shared(self, rank_defs, buf_name, values, dtype, suffix=".norot"):
        common_grp = "weights/shared/model-common.safetensors"
        for ri in range(8):
            self.replace_tensor(
                rank_defs[ri][buf_name],
                values,
                suffix=suffix,
                dtype=dtype,
                group=common_grp,
            )

    def _replace_expert_bank(
        self,
        rdefs,
        folder: Path,
        prefix: str,
        stem: str,
        *,
        is_w8: bool,
        group: str,
        w8_packed_suffix: str,
        w8_scale_suffix: str,
        w4_suffix: str,
    ):
        if is_w8:
            self.replace_from_npy(
                rdefs[prefix + ".packed"],
                folder / f"{stem}_w8.npy",
                suffix=w8_packed_suffix,
                dtype="I8",
                group=group,
            )
            self.replace_from_npy(
                rdefs[prefix + ".scale"],
                folder / f"{stem}_scale_bf16.npy",
                suffix=w8_scale_suffix,
                dtype="BF16",
                group=group,
            )
        else:
            self.replace_from_npy(
                rdefs[prefix + ".packed"],
                folder / f"{stem}_nz.npy",
                suffix=w4_suffix,
                dtype="I8",
                group=group,
            )
            self.replace_from_npy(
                rdefs[prefix + ".scale"],
                folder / f"{stem}_scale.npy",
                suffix=w4_suffix,
                dtype="U64",
                group=group,
            )
            self.replace_from_npy(
                rdefs[prefix + ".assist"],
                folder / f"{stem}_assist.npy",
                suffix=w4_suffix,
                dtype="F32",
                group=group,
            )

    def _plan_shared_backbone(self, ckpt, ranks, rank_defs):
        common_grp = "weights/shared/model-common.safetensors"
        for count, bname in ((256, "norot.ones256"), (1280, "norot.ones1280")):
            tname = "native." + bname
            self.add_tensor(tname, np.ones(count, dtype="<f4"), dtype="F32", group=common_grp)
            for ri in range(8):
                d = dict(
                    name=bname,
                    bytes=count * 4,
                    prepared_tensor=dict(name=tname, dtype="F32", shape=[count]),
                )
                ranks[ri]["buffers"].append(d)
                rank_defs[ri][bname] = d

        if "rotation" in rank_defs[0]:
            self._replace_shared(rank_defs, "rotation", np.eye(32, dtype="<f4"), "F32")
        self._replace_shared(
            rank_defs,
            "w.norm.weight.0-5120.0-1.f32",
            ckpt.read("norm.weight").reshape(5120, 1).astype("<f4"),
            "F32",
        )
        self._replace_shared(
            rank_defs,
            "dspark.warm_decode.w.mtp.0.main_proj.weight.0-5120.0-15360.bf16",
            to_bf16_u16(ckpt.matrix("mtp.0.main_proj.weight")),
            "BF16",
        )

        emb_off = to_bf16_u16(ckpt.matrix("embed.weight"))
        self.transforms["embed.weight"] = "embed.weight"
        self.payloads["embed.weight"] = emb_off
        self.payload_dtypes["embed.weight"] = "BF16"
        self.override_groups["embed.weight"] = "weights/shared/embedding.safetensors"

        head_full = ckpt.matrix("head.weight")
        for ri in range(8):
            h_parts = bf16_parts(head_full[ri * 16160 : (ri + 1) * 16160]).astype("<u2")
            bname = f"w.head.weight.{ri * 16160}-16160.0-5120.f32"
            grp = f"weights/gpu/rank-{ri:02}/layers-36-39.safetensors"
            self.replace_tensor(
                rank_defs[ri][bname], h_parts, suffix=".norot", dtype="BF16", group=grp
            )
        del head_full

        for el in (1, 14):
            ew = ckpt.matrix(f"layers.{el}.engram.wkv.weight")
            first = el // 4 * 4
            for ri in range(8):
                sl = to_bf16_u16(ew[ri * 3200 : (ri + 1) * 3200])
                bname = f"w.layers.{el}.engram.wkv.weight.{ri * 3200}-3200.0-6144.bf16"
                grp = f"weights/gpu/rank-{ri:02}/layers-{first:02}-{first + 3:02}.safetensors"
                self.replace_tensor(
                    rank_defs[ri][bname], sl, suffix=".norot", dtype="BF16", group=grp
                )
            del ew

    def _plan_layer_weights(self, ckpt, layer: int, rank_defs, sq_active: dict):
        pref = f"layers.{layer}"
        s_q, s_attn, s_shw, s_ffn = self._load_layer_smoothquant(layer)
        has_q = bool(np.any(s_q != 1.0))
        has_attn = bool(np.any(s_attn != 1.0))
        has_shw = bool(np.any(s_shw != 1.0))
        has_ffn = bool(np.any(s_ffn != 1.0))
        sq_active["q"] += int(has_q)
        sq_active["attn"] += int(has_attn)
        sq_active["shw"] += int(has_shw)
        sq_active["ffn"] += int(has_ffn)

        attn_norm_w = ckpt.read(f"{pref}.attn_norm.weight").astype("<f4")
        if has_attn:
            attn_norm_w = attn_norm_w / s_attn
        ffn_norm_w = ckpt.read(f"{pref}.ffn_norm.weight").astype("<f4")
        if has_ffn:
            ffn_norm_w = ffn_norm_w / s_ffn
        q_norm_w = ckpt.read(f"{pref}.attn.q_norm.weight").astype("<f4")
        if has_q:
            q_norm_w = q_norm_w / s_q

        self._replace_shared(
            rank_defs,
            f"w.{pref}.attn_norm.weight.0-5120.0-1.f32",
            attn_norm_w.reshape(5120, 1).astype("<f4"),
            "F32",
        )
        self._replace_shared(
            rank_defs,
            f"w.{pref}.ffn_norm.weight.0-5120.0-1.f32",
            ffn_norm_w.reshape(5120, 1).astype("<f4"),
            "F32",
        )
        self._replace_shared(
            rank_defs,
            f"w.{pref}.attn.q_norm.weight.0-1280.0-1.f32",
            q_norm_w.reshape(1280, 1).astype("<f4"),
            "F32",
        )
        self._replace_shared(
            rank_defs,
            f"w.{pref}.attn.kv_norm.weight.0-512.0-1.f32",
            ckpt.read(f"{pref}.attn.kv_norm.weight").reshape(512, 1).astype("<f4"),
            "F32",
        )

        for hc_name in ("hc_attn_fn", "hc_ffn_fn"):
            padded = np.zeros((32, 20480), dtype="<f4")
            padded[:24] = ckpt.read(f"{pref}.{hc_name}").astype("<f4")
            self._replace_shared(rank_defs, f"w.{pref}.{hc_name}", padded.reshape(-1), "F32")
            self._replace_shared(
                rank_defs,
                f"parts.w.{pref}.{hc_name}",
                bf16_parts(padded).astype("<u2"),
                "BF16",
            )

        gw = ckpt.matrix(f"{pref}.ffn.gate.weight").astype("<f4")
        if has_ffn:
            gw = (gw * s_ffn[None, :]).astype("<f4")
        self._replace_shared(rank_defs, f"w.{pref}.ffn.gate.weight.0-384.0-5120.f32", gw, "F32")
        self._replace_shared(
            rank_defs, f"w.{pref}.ffn.gate.weight.0-384.0-5120.bf16", to_bf16_u16(gw), "BF16"
        )

        wqa = ckpt.matrix(f"{pref}.attn.wq_a.weight")
        wkv = ckpt.matrix(f"{pref}.attn.wkv.weight")
        if layer in self.bf16_qkv_layers:
            self._replace_shared(
                rank_defs,
                f"w.{pref}.attn.wq_a.weight.0-1280.0-5120.i8",
                to_bf16_u16(wqa),
                "BF16",
                suffix=".norot_bf16",
            )
            self._replace_shared(
                rank_defs,
                f"w.{pref}.attn.wkv.weight.0-512.0-5120.i8",
                to_bf16_u16(wkv),
                "BF16",
                suffix=".norot_bf16",
            )
        else:
            if has_attn:
                wqa = wqa * s_attn[None, :]
                wkv = wkv * s_attn[None, :]
            q_wqa, s_wqa = quantize_per_channel_i8_f32(wqa)
            q_wkv, s_wkv = quantize_per_channel_i8_f32(wkv)
            self._replace_shared(
                rank_defs, f"w.{pref}.attn.wq_a.weight.0-1280.0-5120.i8", q_wqa, "I8"
            )
            self._replace_shared(
                rank_defs, f"w.{pref}.attn.wq_a.weight_scale.0-1280.0-1.f32", s_wqa, "F32"
            )
            self._replace_shared(
                rank_defs, f"w.{pref}.attn.wkv.weight.0-512.0-5120.i8", q_wkv, "I8"
            )
            self._replace_shared(
                rank_defs, f"w.{pref}.attn.wkv.weight_scale.0-512.0-1.f32", s_wkv, "F32"
            )

        if layer in (2, 8, 14):
            cwkv = ckpt.matrix(f"{pref}.attn.compressor.wkv.weight").astype("<f4")
            cwgate = ckpt.matrix(f"{pref}.attn.compressor.wgate.weight").astype("<f4")
            self._replace_shared(
                rank_defs, f"w.{pref}.attn.compressor.wkv.weight.0-512.0-5120.f32", cwkv, "F32"
            )
            self._replace_shared(
                rank_defs,
                f"parts.w.{pref}.attn.compressor.wkv.weight.0-512.0-5120.f32",
                bf16_parts(cwkv).astype("<u2"),
                "BF16",
            )
            self._replace_shared(
                rank_defs, f"w.{pref}.attn.compressor.wgate.weight.0-512.0-5120.f32", cwgate, "F32"
            )
            self._replace_shared(
                rank_defs,
                f"parts.w.{pref}.attn.compressor.wgate.weight.0-512.0-5120.f32",
                bf16_parts(cwgate).astype("<u2"),
                "BF16",
            )
        elif layer == 20:
            cwkv = ckpt.matrix(f"{pref}.attn.compressor.wkv.weight")
            self._replace_shared(
                rank_defs,
                f"w.{pref}.attn.compressor.wkv.weight.0-512.0-5120.bf16",
                to_bf16_u16(cwkv),
                "BF16",
            )

        if layer in (2, 8, 14, 20, 24, 28, 32, 36):
            iwp = ckpt.matrix(f"{pref}.attn.indexer.weights_proj.weight")
            if has_attn:
                iwp = iwp * s_attn[None, :]
            self._replace_shared(
                rank_defs,
                f"w.{pref}.attn.indexer.weights_proj.weight.0-32.0-5120.bf16",
                to_bf16_u16(iwp),
                "BF16",
            )
            idx_wqb = ckpt.matrix(f"{pref}.attn.indexer.wq_b.weight")
            if has_q:
                idx_wqb = idx_wqb * s_q[None, :]
            qi, si = quantize_per_channel_i8_f32(idx_wqb)
            self._replace_shared(
                rank_defs, f"w.{pref}.attn.indexer.wq_b.weight.0-4096.0-1280.i8", qi, "I8"
            )
            self._replace_shared(
                rank_defs, f"w.{pref}.attn.indexer.wq_b.weight_scale.0-4096.0-1.f32", si, "F32"
            )

        wqb_full = ckpt.matrix(f"{pref}.attn.wq_b.weight")
        if has_q:
            wqb_full = wqb_full * s_q[None, :]
        wob_full = ckpt.matrix(f"{pref}.attn.wo_b.weight")
        sh_w1_m = ckpt.matrix(f"{pref}.ffn.shared_experts.w1.weight")
        sh_w3_m = ckpt.matrix(f"{pref}.ffn.shared_experts.w3.weight")
        sh_w2_m = ckpt.matrix(f"{pref}.ffn.shared_experts.w2.weight")
        if has_ffn:
            sh_w1_m = sh_w1_m * s_ffn[None, :]
            sh_w3_m = sh_w3_m * s_ffn[None, :]
        if has_shw:
            sh_w3_m = sh_w3_m / s_shw[:, None]
            sh_w2_m = sh_w2_m * s_shw[None, :]
        sh_w1_q, sh_w1_s = quantize_per_channel_i8_f32(sh_w1_m)
        sh_w3_q, sh_w3_s = quantize_per_channel_i8_f32(sh_w3_m)
        sh_w2_q, sh_w2_s = quantize_per_channel_i8_f32(sh_w2_m)
        self._replace_shared(
            rank_defs,
            f"w.{pref}.ffn.shared_experts.w2.weight_scale.0-5120.0-1.f32",
            sh_w2_s,
            "F32",
        )

        first = layer // 4 * 4
        is_w8 = layer in self.w8_layers
        for ri in range(8):
            grp = f"weights/gpu/rank-{ri:02}/layers-{first:02}-{first + 3:02}.safetensors"
            exp_grp = f"weights/gpu/rank-{ri:02}/layer-{layer:02}-w8.safetensors" if is_w8 else grp

            q_wqb, s_wqb = quantize_per_channel_i8_f32(wqb_full[ri * 4096 : (ri + 1) * 4096])
            self.replace_tensor(
                rank_defs[ri][f"w.{pref}.attn.wq_b.weight.{ri * 4096}-4096.0-1280.i8"],
                q_wqb,
                suffix=".norot",
                dtype="I8",
                group=grp,
            )
            self.replace_tensor(
                rank_defs[ri][f"w.{pref}.attn.wq_b.weight_scale.{ri * 4096}-4096.0-1.f32"],
                s_wqb,
                suffix=".norot",
                dtype="F32",
                group=grp,
            )
            self.replace_tensor(
                rank_defs[ri][f"w.{pref}.attn.wo_b.weight.0-5120.{ri * 1024}-1024.bf16"],
                to_bf16_u16(wob_full[:, ri * 1024 : (ri + 1) * 1024]),
                suffix=".norot",
                dtype="BF16",
                group=grp,
            )
            sh_up = np.concatenate(
                [sh_w1_q[ri * 288 : (ri + 1) * 288], sh_w3_q[ri * 288 : (ri + 1) * 288]],
                axis=0,
            )
            sh_sc = np.concatenate(
                [
                    sh_w1_s[ri * 288 : (ri + 1) * 288, 0],
                    sh_w3_s[ri * 288 : (ri + 1) * 288, 0],
                ],
                axis=0,
            )
            sh_dn = np.ascontiguousarray(sh_w2_q[:, ri * 288 : (ri + 1) * 288])
            self.replace_tensor(
                rank_defs[ri][f"w.shared.{layer}.{ri}.up"],
                sh_up,
                suffix=".norot",
                dtype="I8",
                group=grp,
            )
            self.replace_tensor(
                rank_defs[ri][f"w.shared.{layer}.{ri}.scale"],
                sh_sc,
                suffix=".norot",
                dtype="F32",
                group=grp,
            )
            self.replace_tensor(
                rank_defs[ri][f"w.shared.{layer}.{ri}.down"],
                sh_dn,
                suffix=".norot",
                dtype="I8",
                group=grp,
            )

            edir = self.prep_dir / f"L{layer:02d}_r{ri}"
            self._replace_expert_bank(
                rank_defs[ri],
                edir,
                f"w.L{layer}.['w1', 'w3']",
                "up",
                is_w8=is_w8,
                group=exp_grp,
                w8_packed_suffix=".norot_w8",
                w8_scale_suffix=".norot_w8",
                w4_suffix=".norot_ssz",
            )
            self._replace_expert_bank(
                rank_defs[ri],
                edir,
                f"w.L{layer}.['w2']",
                "down",
                is_w8=is_w8,
                group=exp_grp,
                w8_packed_suffix=".norot_w8",
                w8_scale_suffix=".norot_w8",
                w4_suffix=".norot_ssz",
            )

    def _plan_vision_and_draft_weights(self, ckpt, ranks, rank_defs):
        vis_grp = "vision.safetensors"
        for vname in (
            "aligner.w2.weight",
            "aligner.w2.bias",
            "image_start",
            "image_end",
            "image_newline",
        ):
            val = ckpt.matrix(vname) if vname.endswith(".weight") else ckpt.read(vname)
            self.transforms[vname] = vname
            self.payloads[vname] = to_bf16_u16(val)
            self.payload_dtypes[vname] = "BF16"
            self.override_groups[vname] = vis_grp

        if not (self.draft_prep_dir and self.draft_mode):
            return
        up_w8 = self.draft_mode == "full_w8"
        dn_w8 = self.draft_mode in ("ssz_up_w8_down", "full_w8")
        for ri in range(8):
            dgrp = f"weights/dspark/rank-{ri:02}-layers-40-42.safetensors"
            if up_w8:
                up_out_name = "dspark.draft.batch.prefill.m512.expert.up.output"
                if up_out_name not in rank_defs[ri]:
                    bdef = dict(name=up_out_name, bytes=1536 * 4608 * 2)
                    ranks[ri]["buffers"].append(bdef)
                    rank_defs[ri][up_out_name] = bdef
            for dl in (40, 41, 42):
                folder = self.draft_prep_dir / f"L{dl}_r{ri}"
                self._replace_expert_bank(
                    rank_defs[ri],
                    folder,
                    f"dspark.draft.w.L{dl}.['w1', 'w3']",
                    "up",
                    is_w8=up_w8,
                    group=dgrp,
                    w8_packed_suffix=".w8_nd",
                    w8_scale_suffix=".w8_bf16",
                    w4_suffix=".ssz_w4",
                )
                self._replace_expert_bank(
                    rank_defs[ri],
                    folder,
                    f"dspark.draft.w.L{dl}.['w2']",
                    "down",
                    is_w8=dn_w8,
                    group=dgrp,
                    w8_packed_suffix=".w8_nd",
                    w8_scale_suffix=".w8_bf16",
                    w4_suffix=".ssz_w4",
                )

    def _rewrite_packet_graphs(self, ranks):
        specs = {}

        def get_spec(manifest):
            if manifest not in specs:
                specs[manifest] = json.loads((self.package / manifest).read_text())
            return specs[manifest]

        bf16_set = set(self.bf16_qkv_layers)
        w8_set = set(self.w8_layers)

        for rank in ranks:
            p1 = rank["packets"][1]
            p1_a8_by_layer = {}
            for i, s in enumerate(p1["body"]):
                m_up = re.search(r"\.L(\d+)\.up$", s.get("name", ""))
                if m_up and int(m_up[1]) in w8_set:
                    layer = int(m_up[1])
                    p1_a8_by_layer[layer] = deepcopy(p1["body"][i - 1 : i + 5])

            for pi, packet in enumerate(rank["packets"]):
                body = packet["body"]
                role = packet.get("role", "target")

                if role == "draft":
                    if self.draft_mode:
                        self._rewrite_draft_packet(body)
                    continue

                self._rewrite_target_packet(packet, pi, get_spec, bf16_set, w8_set, p1_a8_by_layer)

        if self.remap:
            self._apply_manifest_remap(ranks)

        if bf16_set:
            rs = self.model.setdefault("row_selections", {})
            for tag in ("wqa", "wkv"):
                rs[self.kernels[f"bf16_{tag}_2048"]] = [
                    dict(
                        min_rows=lo,
                        max_rows=hi,
                        multiple_of=1,
                        kernels=[self.kernels[f"bf16_{tag}_2048_m{hi}"]],
                    )
                    for lo, hi in ((1, 16), (17, 32), (33, 64), (65, 127))
                ]
                rs[self.kernels[f"bf16_{tag}_256"]] = [
                    dict(
                        min_rows=1,
                        max_rows=64,
                        multiple_of=1,
                        kernels=[self.kernels[f"bf16_{tag}_256_m64"]],
                    ),
                ]

        used_manifests = {
            step["manifest"]
            for rank in ranks
            for packet in rank["packets"]
            for step in packet["body"] + packet["head"]
            if step["op"] == "kernel"
        }
        if "row_selections" in self.model:
            self.model["row_selections"] = {
                k: v for k, v in self.model["row_selections"].items() if k in used_manifests
            }

    def _rewrite_draft_packet(self, body):
        up_w8 = self.draft_mode == "full_w8"
        dn_w8 = self.draft_mode in ("ssz_up_w8_down", "full_w8")
        for i, step in enumerate(body):
            sname = step.get("name", "")
            if dn_w8 and step["op"] == "w4" and re.search(r"\.L4[0-2]\.down$", sname):
                step["op"] = "w8"
                step["weight_layout"] = "nd"
                step.pop("expected_expert_rows", None)
            elif up_w8 and step["op"] == "w4_a4" and re.search(r"\.L4[0-2]\.up$", sname):
                dl = int(re.search(r"\.L(4[0-2])\.up$", sname)[1])
                g_s, f_s = body[i - 1], body[i + 1]
                body[i - 1] = dict(
                    op="kernel",
                    manifest=self.kernels["draft_gather"],
                    buffers=[
                        g_s["buffers"][0],
                        g_s["buffers"][1],
                        g_s["buffers"][2],
                        g_s["buffers"][4],
                        g_s["buffers"][5],
                        g_s["buffers"][6],
                    ],
                )
                body[i] = dict(
                    op="w8",
                    name=sname,
                    rows=1536,
                    input=5120,
                    output=4608,
                    experts=16,
                    workspace=step["workspace"],
                    groups=f"dspark.draft.batch.prefill.m512.L{dl}.counts",
                    weight_layout="nd",
                    buffers=[
                        "dspark.draft.batch.prefill.m512.expert.input",
                        step["buffers"][1],
                        step["buffers"][2],
                        step["buffers"][3],
                        "dspark.draft.batch.prefill.m512.expert.input.scale",
                        "dspark.draft.batch.prefill.m512.expert.up.output",
                    ],
                )
                body[i + 1] = dict(
                    op="kernel",
                    manifest=self.kernels["draft_swiglu"],
                    buffers=[
                        "dspark.draft.batch.prefill.m512.expert.up.output",
                        f_s["buffers"][4],
                        f_s["buffers"][5],
                    ],
                )

    def _rewrite_target_packet(self, packet, pi, get_spec, bf16_set, w8_set, p1_a8_by_layer):
        body = packet["body"]
        for step in body:
            if step["op"] == "kernel":
                sp = get_spec(step["manifest"])
                if sp["op"] in ("engram.hadamard32", "engram.rotate_blocks", "engram.cast_fp32"):
                    cap = sp["parameters"]["_row_capacity"]
                    step["manifest"] = self.kernels[f"engram_cast_{cap}"]
                    if len(step["buffers"]) == 3:
                        step["buffers"] = [step["buffers"][0], step["buffers"][2]]

        norm5120_template = None
        for st in body:
            if st["op"] == "kernel":
                sp = get_spec(st["manifest"])
                if sp["op"] == "model_ops.norm" and any(
                    "layers.2.attn_norm" in b for b in st["buffers"]
                ):
                    norm5120_template = (st["manifest"], st["buffers"][2], st["buffers"][3])
                    break

        quant_producers = {}
        qkv_rewritten_layers = set()
        obsolete_quant_indices = set()
        for idx, step in enumerate(body):
            if step["op"] != "kernel":
                continue
            sp = get_spec(step["manifest"])
            if sp["op"] == "vector.quantize":
                quant_producers[step["buffers"][1]] = ("quantize", step["buffers"][0], idx)
            elif sp["op"] == "model_ops.norm_quant" and any(
                ".attn_norm." in b for b in step["buffers"]
            ):
                quant_producers[step["buffers"][2]] = ("norm_quant", norm5120_template[1], idx)
            elif sp["op"] == "dense.projection":
                m_qkv = re.fullmatch(
                    r"w\.layers\.(\d+)\.attn\.(wq_a|wkv)\.weight\.0-(\d+)\.0-5120\.i8",
                    step["buffers"][1],
                )
                if not m_qkv:
                    continue
                layer, part, out_n = int(m_qkv[1]), m_qkv[2], int(m_qkv[3])
                if layer not in bf16_set:
                    continue
                kind, norm_buf, q_idx = quant_producers[step["buffers"][0]]
                cap = sp["parameters"]["_row_capacity"]
                k_key = f"bf16_{'wqa' if part == 'wq_a' else 'wkv'}_{cap}"
                tok_ones = {32: "batch.decode.ones1", 2048: "ones2048", 256: "norot.ones256"}[cap]
                ch_ones = "norot.ones1280" if out_n == 1280 else "ones512"
                step["manifest"] = self.kernels[k_key]
                step["buffers"] = [
                    norm_buf,
                    step["buffers"][1],
                    tok_ones,
                    ch_ones,
                    step["buffers"][4],
                    step["buffers"][5],
                ]
                qkv_rewritten_layers.add((layer, part))
                if (layer, "wq_a") in qkv_rewritten_layers and (
                    layer,
                    "wkv",
                ) in qkv_rewritten_layers:
                    if kind == "quantize":
                        obsolete_quant_indices.add(q_idx)
                    else:
                        qstep = body[q_idx]
                        qstep["manifest"] = norm5120_template[0]
                        qstep["buffers"] = [
                            qstep["buffers"][0],
                            qstep["buffers"][1],
                            norm5120_template[1],
                            norm5120_template[2],
                        ]

        if obsolete_quant_indices:
            body = [s for idx, s in enumerate(body) if idx not in obsolete_quant_indices]
            packet["body"] = body

        if pi in (0, 1):
            for step in body:
                m_ex = re.search(r"\.L(\d+)\.(up|down)$", step.get("name", ""))
                if step["op"] == "w4" and m_ex and int(m_ex[1]) in w8_set:
                    step["op"] = "w8"
                    step["weight_layout"] = "nd"
                    step.pop("expected_expert_rows", None)
        elif pi in (2, 3):
            i = 0
            while i < len(body):
                step = body[i]
                m_up = re.search(r"\.L(\d+)\.up$", step.get("name", ""))
                if step["op"] == "w4_a4" and m_up and int(m_up[1]) in w8_set:
                    layer = int(m_up[1])
                    rep = deepcopy(p1_a8_by_layer[layer])
                    for rs in rep:
                        if rs["op"] == "w4":
                            rs["op"] = "w8"
                            rs["weight_layout"] = "nd"
                            rs.pop("expected_expert_rows", None)
                    body[i - 1 : i + 5] = rep
                    i += 5
                else:
                    i += 1

    def _apply_manifest_remap(self, ranks):
        for rank in ranks:
            for packet in rank["packets"]:
                for step in packet["body"] + packet["head"]:
                    if step["op"] == "kernel":
                        if step["manifest"] in self.remap:
                            step["manifest"] = self.remap[step["manifest"]]
                        if step.get("uncompacted_manifest") in self.remap:
                            step["uncompacted_manifest"] = self.remap[step["uncompacted_manifest"]]
        if "row_selections" in self.model:
            new_rs = {}
            for k, choices in self.model["row_selections"].items():
                nk = self.remap.get(k, k)
                new_rs[nk] = [
                    {**ch, "kernels": [self.remap.get(x, x) for x in ch["kernels"]]}
                    for ch in choices
                ]
            self.model["row_selections"] = new_rs

    def _assemble_shard_groups(self):
        extra_shards_entries = {}
        for group, record in self.old_records.items():
            path = self.source / group
            start, header = local_header(path)
            for name in sorted(record["tensors"], key=lambda n: header[n]["data_offsets"][0]):
                spec = header[name]
                entry = dict(
                    dtype=spec["dtype"],
                    shape=spec["shape"],
                    bytes=spec["data_offsets"][1] - spec["data_offsets"][0],
                    source=dict(file=str(path), offset=start + spec["data_offsets"][0]),
                )
                if name in self.transforms:
                    replacement = self.transforms[name]
                    if replacement in self.file_replacements:
                        fr = self.file_replacements[replacement]
                        target_grp = fr["group"]
                        if replacement != name:
                            self.index["weight_map"].pop(name, None)
                        self.index["weight_map"][replacement] = target_grp
                        new_entry = dict(
                            dtype=fr["dtype"],
                            shape=fr["shape"],
                            bytes=fr["bytes"],
                            source=dict(file=fr["file"], offset=fr["offset"]),
                        )
                        self.rewrite_groups.add(group)
                        if target_grp == group:
                            self.groups[group].append((replacement, new_entry))
                        else:
                            extra_shards_entries.setdefault(target_grp, []).append(
                                (replacement, new_entry)
                            )
                            self.rewrite_groups.add(target_grp)
                        continue
                    values = self.payloads[replacement]
                    if replacement != name:
                        self.index["weight_map"].pop(name, None)
                    self.index["weight_map"][replacement] = group
                    entry.update(
                        dtype=self.payload_dtypes[replacement],
                        shape=list(values.shape),
                        bytes=values.nbytes,
                        source=dict(replacement=replacement),
                    )
                    name = replacement
                    self.rewrite_groups.add(group)
                self.groups[group].append((name, entry))

        for tgt_grp, items in extra_shards_entries.items():
            self.groups[tgt_grp].extend(items)
            self.rewrite_groups.add(tgt_grp)

        for name, values in self.payloads.items():
            if name in self.index["weight_map"]:
                continue
            grp = self.override_groups[name]
            self.index["weight_map"][name] = grp
            self.groups[grp].append(
                (
                    name,
                    dict(
                        dtype=self.payload_dtypes[name],
                        shape=list(values.shape),
                        bytes=values.nbytes,
                        source=dict(replacement=name),
                    ),
                )
            )
            self.rewrite_groups.add(grp)

        for grp, items in self.groups.items():
            sz = sum(e["bytes"] for _, e in items)
            if sz >= 4 << 30:
                raise ValueError(f"Shard {grp} exceeds 4 GiB: {sz / (1 << 30):.2f} GiB")

    def plan(self):
        ckpt = Checkpoint(self.official_dir)
        t0 = time.monotonic()
        try:
            ranks = self.model["ranks"]
            rank_defs = [{b["name"]: b for b in r["buffers"]} for r in ranks]
            self._plan_shared_backbone(ckpt, ranks, rank_defs)

            sq_active = dict(q=0, attn=0, shw=0, ffn=0)
            for layer in range(40):
                self._plan_layer_weights(ckpt, layer, rank_defs, sq_active)

            self._plan_vision_and_draft_weights(ckpt, ranks, rank_defs)
            self._rewrite_packet_graphs(ranks)
            self._assemble_shard_groups()

            self.manifest["norot_upgrade"] = dict(
                basis="unrotated-official-native-grid",
                w8_expert_layers=self.w8_layers,
                bf16_qkv_layers=self.bf16_qkv_layers,
                draft_mode=self.draft_mode,
                smoothquant_calibrated=self.smoothquant_dir is not None,
                smoothquant_active_layers=sq_active,
                rewritten_shards=len(self.rewrite_groups),
                hardlinked_shards=len(self.groups) - len(self.rewrite_groups),
            )
            print(
                f"Plan complete in {time.monotonic() - t0:.1f}s: "
                f"rewriting {len(self.rewrite_groups)} shards, "
                f"hardlinking {len(self.groups) - len(self.rewrite_groups)} shards",
                flush=True,
            )
            return self
        finally:
            ckpt.close()


def _compile_new_kernel(job):
    from kernels.ascend import dense, engram, memory, model_ops
    from kernels.shared.rows import row_count
    from orinfer_build.compiler.ascend.build import compile_kernel
    from orinfer_build.contracts.row_parameters import dynamic_parameters

    key, mod_name, fn_name, arguments, specs, symbolic, capacity, grid_limit, cache_dir = job
    mod = {"dense": dense, "engram": engram, "memory": memory, "model_ops": model_ops}[mod_name]
    fn = getattr(mod, fn_name)
    call_args = dict(arguments)
    m = row_count(capacity, True)
    for ax, factor in symbolic.items():
        call_args[ax] = m * factor
    func = fn(**call_args)
    params = dynamic_parameters(arguments, symbolic, capacity)
    manifest = compile_kernel(
        func,
        f"{mod_name}.{fn_name}",
        specs,
        params,
        Path(cache_dir),
        row_capacity=capacity,
        grid_limit=grid_limit,
    )
    return key, str(Path(manifest).relative_to(Path(cache_dir).parent))


def _sync_existing_kernel(job):
    from kernels.ascend import dense, engram, memory, model_ops
    from kernels.shared.rows import row_count
    from orinfer_build.compiler.ascend.build import compile_kernel
    from orinfer_build.contracts.row_parameters import dynamic_parameters

    old_rel, spec, cache_dir = job
    op = spec["op"]
    mod_name, fn_name = op.split(".", 1)
    mod = {"dense": dense, "engram": engram, "memory": memory, "model_ops": model_ops}[mod_name]
    fn = getattr(mod, fn_name)
    params = dict(spec["parameters"])
    cap = params.pop("_row_capacity", None)
    grid = params.pop("_grid_limit", 64)
    dyn = params.pop("_dynamic_arguments", None)
    call_args = dict(params)
    if cap is not None:
        m = row_count(cap, True)
        if dyn:
            for ax, factor in dyn.items():
                call_args[ax] = m * factor
        elif "rows" in call_args and call_args["rows"] == cap:
            call_args["rows"] = m
    func = fn(**call_args)
    out_params = dynamic_parameters(params, dyn, cap) if dyn else dict(params)
    manifest = compile_kernel(
        func,
        op,
        spec["operands"],
        out_params,
        Path(cache_dir),
        row_capacity=cap,
        grid_limit=grid,
    )
    return old_rel, str(Path(manifest).relative_to(Path(cache_dir).parent))


def compile_kernels_command(args):
    """Compile extra AOT kernels and sync source_sha256 for modified operator modules."""
    from orinfer_build.compiler.ascend.build import operand

    base_pkg = args.base_package.resolve(strict=True)
    out_dir = args.output_dir.resolve()
    out_pkg = out_dir / base_pkg.name
    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_pkg, out_pkg, copy_function=os.link)
    cache_dir = out_pkg / "cache"

    def op_specs(pairs):
        return [operand(n, dt, sh, acc) for n, dt, sh, acc in pairs]

    def bf16_proj_job(key, rows, out_dim, tile_m, tile_n, block_k, stages):
        return (
            key,
            "dense",
            "projection",
            dict(
                rows=rows,
                output=out_dim,
                reduction=5120,
                integer=False,
                block_k=block_k,
                tile_n=tile_n,
                tile_m=tile_m,
                pipeline_stages=stages,
                unit_scale=True,
            ),
            op_specs(
                [
                    ("activation", "bf16", (rows, 5120), "read"),
                    ("weight", "bf16", (out_dim, 5120), "read"),
                    ("token_scale", "f32", (rows,), "read"),
                    ("channel_scale", "f32", (out_dim,), "read"),
                    ("out", "bf16", (rows, out_dim), "write"),
                    ("workspace", "f32", (rows, out_dim), "scratch"),
                ]
            ),
            {"rows": 1},
            rows,
            20,
            str(cache_dir),
        )

    def engram_cast_job(key, rows):
        return (
            key,
            "engram",
            "cast_fp32",
            dict(rows=rows, width=5120, tile=2048),
            op_specs(
                [
                    ("hidden", "bf16", (rows * 4 * 5120,), "read"),
                    ("out", "f32", (rows * 4 * 5120,), "write"),
                ]
            ),
            {"rows": 1},
            rows,
            20,
            str(cache_dir),
        )

    jobs = [
        (
            "draft_gather",
            "memory",
            "gather_quantized_rows",
            dict(rows=1536, table_rows=512, width=5120),
            op_specs(
                [
                    ("table", "i8", (512, 5120), "read"),
                    ("scales", "f32", (512,), "read"),
                    ("indices", "i32", (1536,), "read"),
                    ("out", "i8", (1536, 5120), "write"),
                    ("out_scales", "f32", (1536,), "write"),
                    ("errors", "i32", (1536,), "write"),
                ]
            ),
            {"rows": 3, "table_rows": 1},
            512,
            64,
            str(cache_dir),
        ),
        (
            "draft_swiglu",
            "model_ops",
            "swiglu_packed_quant",
            dict(rows=1536, width=2304),
            op_specs(
                [
                    ("packed", "bf16", (1536, 4608), "read"),
                    ("out", "i8", (1536, 2304), "write"),
                    ("scales", "f32", (1536,), "write"),
                ]
            ),
            {"rows": 3},
            512,
            64,
            str(cache_dir),
        ),
        engram_cast_job("engram_cast_32", 32),
        engram_cast_job("engram_cast_256", 256),
        engram_cast_job("engram_cast_2048", 2048),
        bf16_proj_job("bf16_wqa_32", 32, 1280, 16, 64, 512, 0),
        bf16_proj_job("bf16_wkv_32", 32, 512, 16, 64, 512, 0),
        bf16_proj_job("bf16_wqa_2048", 2048, 1280, 128, 128, 512, 0),
        bf16_proj_job("bf16_wqa_2048_m16", 2048, 1280, 16, 64, 512, 3),
        bf16_proj_job("bf16_wqa_2048_m32", 2048, 1280, 32, 64, 512, 3),
        bf16_proj_job("bf16_wqa_2048_m64", 2048, 1280, 32, 128, 512, 2),
        bf16_proj_job("bf16_wqa_2048_m127", 2048, 1280, 64, 128, 512, 2),
        bf16_proj_job("bf16_wkv_2048", 2048, 512, 128, 128, 512, 0),
        bf16_proj_job("bf16_wkv_2048_m16", 2048, 512, 16, 64, 512, 3),
        bf16_proj_job("bf16_wkv_2048_m32", 2048, 512, 32, 64, 512, 3),
        bf16_proj_job("bf16_wkv_2048_m64", 2048, 512, 32, 128, 512, 2),
        bf16_proj_job("bf16_wkv_2048_m127", 2048, 512, 64, 128, 512, 2),
        bf16_proj_job("bf16_wqa_256", 256, 1280, 128, 128, 512, 0),
        bf16_proj_job("bf16_wqa_256_m64", 256, 1280, 32, 128, 512, 2),
        bf16_proj_job("bf16_wkv_256", 256, 512, 128, 128, 512, 0),
        bf16_proj_job("bf16_wkv_256_m64", 256, 512, 32, 128, 512, 2),
    ]

    t0 = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        manifests = dict(pool.map(_compile_new_kernel, jobs))

    model = decode(json.loads((base_pkg / "model.json").read_text()))
    existing = set()
    for p in model["ranks"][0]["packets"]:
        for s in p["body"] + p["head"]:
            if s["op"] == "kernel":
                existing.add(s["manifest"])
                if s.get("uncompacted_manifest"):
                    existing.add(s["uncompacted_manifest"])
    for key, choices in model.get("row_selections", {}).items():
        existing.add(key)
        for ch in choices:
            existing.update(ch["kernels"])

    sync_jobs = []
    for rel in sorted(existing):
        sp = json.loads((base_pkg / rel).read_text())
        if sp["op"] in SYNC_OPS:
            sync_jobs.append((rel, sp, str(cache_dir)))

    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        remap = dict(pool.map(_sync_existing_kernel, sync_jobs))

    summary = dict(
        package_root=str(out_pkg),
        manifests=manifests,
        remap=remap,
        elapsed_s=round(time.monotonic() - t0, 1),
    )
    atomic_json(out_dir / "extra-kernels.json", summary)
    print(
        json.dumps(dict(kernels=len(manifests), synced=len(remap), elapsed_s=summary["elapsed_s"]))
    )


def export_command(args):
    """Prepare target & draft expert banks and export the unrotated physical release."""
    t0 = time.monotonic()
    work_dir = args.work_dir.resolve()
    prep_dir = args.prep_dir.resolve() if args.prep_dir else (work_dir / "prepared-experts")
    draft_prep_dir = (
        args.draft_prep_dir.resolve() if args.draft_prep_dir else (work_dir / "prepared-draft")
    )
    cache_dir = work_dir / "native-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    Decoder(cache_dir)
    SSZQuantizer(cache_dir)

    w8_set = set(args.w8_layers)
    bf16_layers = (
        list(range(40))
        if args.bf16_qkv == "all"
        else (list(KV_SOURCE_LAYERS) if args.bf16_qkv == "kv-sources" else [])
    )

    target_tasks = [
        (str(args.official_dir), L, r, L in w8_set, str(prep_dir), str(cache_dir))
        for L in range(40)
        for r in range(8)
    ]
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        target_recs = list(pool.map(_prepare_target_bank, target_tasks))

    draft_mode = None if args.draft_mode == "none" else args.draft_mode
    draft_recs = []
    if draft_mode:
        up_w8 = draft_mode == "full_w8"
        dn_w8 = draft_mode in ("ssz_up_w8_down", "full_w8")
        draft_tasks = [
            (str(args.official_dir), di, r, up_w8, dn_w8, str(draft_prep_dir), str(cache_dir))
            for di in range(3)
            for r in range(8)
        ]
        with ProcessPoolExecutor(
            max_workers=min(args.workers, 24), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            draft_recs = list(pool.map(_prepare_draft_bank, draft_tasks))

    kinfo = json.loads(args.kernels_json.read_text())
    source_exec = Path(kinfo["package_root"]).parent
    model_out = work_dir / "model"
    exec_out = work_dir / "exec-cache"
    for d in (model_out, exec_out):
        if d.exists():
            shutil.rmtree(d)

    NoRotRelease(
        args.base_model,
        model_out,
        execution_output=exec_out,
        source_execution=source_exec,
        official_dir=args.official_dir,
        prep_dir=prep_dir,
        kernels=kinfo["manifests"],
        remap=kinfo.get("remap", {}),
        w8_layers=args.w8_layers,
        bf16_qkv_layers=bf16_layers,
        draft_prep_dir=draft_prep_dir if draft_mode else None,
        draft_mode=draft_mode,
        smoothquant_dir=args.smoothquant_dir.resolve() if args.smoothquant_dir else None,
    ).plan().export(workers=args.export_workers)

    desc = json.loads((model_out / "cache/model.json").read_text())
    digest = desc["execution_package"]
    if args.compact_engine:
        from orinfer_build.models.deepseek_v41.publication import publish_execution

        compact_cache = work_dir / "compact-cache"
        if compact_cache.exists():
            shutil.rmtree(compact_cache)
        compact_dir = publish_execution(
            exec_out / digest,
            compact_cache,
            compact_rope=True,
            engine=args.compact_engine,
        )
        digest = compact_dir.name
        desc["execution_package"] = digest
        atomic_json(model_out / "cache/model.json", desc)

    w4_recs = [r for r in target_recs if not r["is_w8"]]
    w8_recs = [r for r in target_recs if r["is_w8"]]
    print(
        json.dumps(
            dict(
                model=str(model_out),
                execution_package=digest,
                w4_mean_nmse_up=float(np.mean([r["mean_nmse_up"] for r in w4_recs]))
                if w4_recs
                else 0.0,
                w4_mean_nmse_dn=float(np.mean([r["mean_nmse_down"] for r in w4_recs]))
                if w4_recs
                else 0.0,
                w8_mean_nmse_up=float(np.mean([r["mean_nmse_up"] for r in w8_recs]))
                if w8_recs
                else 0.0,
                w8_mean_nmse_dn=float(np.mean([r["mean_nmse_down"] for r in w8_recs]))
                if w8_recs
                else 0.0,
                draft_mean_nmse_up=float(np.mean([r["mean_nmse_up"] for r in draft_recs]))
                if draft_recs
                else 0.0,
                draft_mean_nmse_dn=float(np.mean([r["mean_nmse_down"] for r in draft_recs]))
                if draft_recs
                else 0.0,
                elapsed_s=round(time.monotonic() - t0, 1),
            ),
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compile-kernels", help="Compile AOT kernels on CPU (--without-devices)")
    c.add_argument("--base-package", type=Path, required=True)
    c.add_argument("--output-dir", type=Path, required=True)
    c.add_argument("--workers", type=int, default=16)
    c.set_defaults(func=compile_kernels_command)

    e = sub.add_parser("export", help="Quantize official weights & export physical release")
    e.add_argument("--official-dir", type=Path, required=True)
    e.add_argument("--base-model", type=Path, required=True)
    e.add_argument("--kernels-json", type=Path, required=True)
    e.add_argument("--work-dir", type=Path, required=True)
    e.add_argument("--prep-dir", type=Path, default=None)
    e.add_argument("--draft-prep-dir", type=Path, default=None)
    e.add_argument("--smoothquant-dir", type=Path, default=None)
    e.add_argument("--w8-layers", type=int, nargs="*", default=[15, 39])
    e.add_argument("--bf16-qkv", choices=["all", "kv-sources", "none"], default="kv-sources")
    e.add_argument(
        "--draft-mode", choices=["none", "ssz_w4", "ssz_up_w8_down", "full_w8"], default="full_w8"
    )
    e.add_argument("--compact-engine", type=Path, default=None)
    e.add_argument("--workers", type=int, default=32)
    e.add_argument("--export-workers", type=int, default=16)
    e.set_defaults(func=export_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
