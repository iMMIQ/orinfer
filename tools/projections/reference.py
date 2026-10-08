"""Captured-activation and checkpoint references for projection validation."""

import hashlib
import json
import time

import torch
from safetensors import safe_open

from tools.reference import CHECKPOINT

MODEL = str(CHECKPOINT / "model.safetensors")
CASES = {
    "gate_up": ["mlp.gate_proj", "mlp.up_proj"],
    "down": ["mlp.down_proj"],
    "gdn_qkvz": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"],
    "gdn_out": ["linear_attn.out_proj"],
}
MATCH = {
    "gate_up": "gate_up_proj",
    "down": "down_proj",
    "gdn_qkvz": "in_proj_qkvz",
    "gdn_out": "out_proj",
    "head": "lm_head",
}


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_weights(case, layer):
    prefix = f"model.language_model.layers.{layer}."
    fields = {name: [] for name in ("weight_packed", "weight_scale", "weight_zero_point")}
    identity = []
    started = time.monotonic()
    costs = dict(source_read_hash_cpu_s=0.0, unpack_repack_cpu_s=0.0)
    with safe_open(MODEL, framework="pt", device="cpu") as f:
        for part in CASES[case]:
            read_started = time.monotonic()
            part_t = {field: f.get_tensor(prefix + part + "." + field) for field in fields}
            n, k8 = part_t["weight_packed"].shape
            for field, t in part_t.items():
                identity.append(
                    dict(
                        name=prefix + part + "." + field,
                        shape=list(t.shape),
                        dtype=str(t.dtype),
                        sha256=sha(t),
                    )
                )
            costs["source_read_hash_cpu_s"] += time.monotonic() - read_started
            pack_started = time.monotonic()
            raw = part_t["weight_packed"]
            zeros = part_t["weight_zero_point"]
            scale = part_t["weight_scale"].half()
            shifts = torch.arange(8, dtype=torch.int32) * 4
            codes = ((raw[:, :, None] >> shifts) & 15).to(torch.uint8).reshape(n, k8 * 8)
            z = (
                ((zeros[:, None, :] >> shifts[None, :, None]) & 15)
                .to(torch.int8)
                .reshape(n, k8 * 8 // 128)
            )
            # Lossless adjacent nibble representation. BF16 scales -> FP16 is exact
            # only if verified (all selected values are checked below).
            assert torch.equal(scale.bfloat16(), part_t["weight_scale"]), (
                "Scale conversion is lossy"
            )
            fields["weight_packed"].append((codes[:, ::2] | (codes[:, 1::2] << 4)).contiguous())
            fields["weight_zero_point"].append(z)
            fields["weight_scale"].append(scale)
            costs["unpack_repack_cpu_s"] += time.monotonic() - pack_started
    copy_started = time.monotonic()
    p, s, z = [torch.cat(fields[field]).cuda() for field in fields]
    torch.cuda.synchronize()
    costs["concat_H2D_s"] = time.monotonic() - copy_started
    reference_started = time.monotonic()
    n, k2 = p.shape
    k = k2 * 2
    b = torch.empty((n, k), device="cuda", dtype=torch.float16)
    for start in range(0, n, 1024):
        code = torch.stack(
            (p[start : start + 1024] & 15, p[start : start + 1024] >> 4), dim=-1
        ).reshape(-1, k)
        b[start : start + 1024] = (
            (
                (code.reshape(-1, k // 128, 128).float() - z[start : start + 1024, :, None].float())
                * s[start : start + 1024, :, None].float()
            )
            .reshape(-1, k)
            .half()
        )
    torch.cuda.synchronize()
    costs["benchmark_only_reference_dequant_s"] = time.monotonic() - reference_started
    costs["total_s"] = time.monotonic() - started
    return p, s, z, b, identity, costs


def activations(folder, case, layer):
    rows = []
    for path in sorted(folder.glob("*.json")):
        meta = json.loads(path.read_text())
        kind = meta["kind"]
        if (case == "head" and kind == "lm_head") or (
            case != "head" and f"layers.{layer}." in kind and kind.endswith(MATCH[case])
        ):
            t = torch.load(folder / meta["file"], map_location="cpu", weights_only=True).half()
            if meta["mode"] == "prefill":
                rows.append(("prefill", t, meta))
            else:
                rows.append(("decode", t, meta))
    pref = next(
        (
            (t, meta)
            for mode, t, meta in rows
            if mode == "prefill" and (len(t) == 512 or case == "head")
        ),
        None,
    )
    dec = [(t, meta) for mode, t, meta in rows if mode == "decode"]
    dec.sort(key=lambda row: row[1]["computed_tokens_before"])
    if not pref or len(dec) < 8:
        raise RuntimeError(
            f"Missing real activations for {case}: prefill={bool(pref)} decode={len(dec)}"
        )
    joined = torch.cat([t for t, _ in dec])
    result = []
    for m in (1, 2, 3, 4, 5, 7, 8):
        result.append(
            (
                m,
                joined[:m].contiguous(),
                dict(
                    origin="concatenated consecutive actual M1 decode steps; not simultaneous batch trace",
                    sources=[meta for _, meta in dec[:m]],
                ),
            )
        )
    if case != "head":
        result.append((512, pref[0], dict(origin="actual 512-row prefill", sources=[pref[1]])))
    return result
