"""Independent bounded-memory Torch QSA oracle, never a native GPU fallback."""

import torch
from tools.model.flash_next.reference.math import norm


def rope(x, positions):
    positions = torch.as_tensor(positions, device=x.device, dtype=torch.float32)
    angle = positions[:, None, None] * 1e7 ** (
        -torch.arange(0, 64, 2, device=x.device).float()[None, None] / 64
    )
    first, second = x[..., :64].float().chunk(2, -1)
    out = x.clone()
    out[..., :64] = torch.cat(
        (first * angle.cos() - second * angle.sin(), second * angle.cos() + first * angle.sin()), -1
    ).to(x.dtype)
    return out


def index(qk, qweight, kweight):
    """Source mean -> round -> zero-centered norm -> first-member RoPE."""
    q = rope(norm(qk[:, :4], qweight), torch.arange(len(qk), device=qk.device))
    groups = len(qk) // 4
    pooled = qk[: groups * 4, 4].float().reshape(groups, 4, 128).mean(1).to(qk.dtype)
    k = rope(norm(pooled[:, None], kweight), torch.arange(groups, device=qk.device) * 4)[:, 0]
    return q, k


def select(query, cache, positions):
    """Exact score order; stable lower-index ties, all causal incomplete tails."""
    results = []
    for q, position in zip(query, positions):
        n = (int(position) + 1) // 4
        if n <= 512:
            ids = torch.arange(n, device=q.device)
        else:
            scores = (q.float() @ cache[:n].float().T).relu().sum(0) * 128**-0.5
            ids = scores.argsort(descending=True, stable=True)[:512]
        chosen = (ids[:, None] * 4 + torch.arange(4, device=q.device)).flatten()
        tail = torch.arange(n * 4, int(position) + 1, device=q.device)
        chosen = torch.cat((chosen, tail)).int()
        results.append(
            torch.cat(
                (chosen, torch.full((2051 - len(chosen),), -1, device=q.device, dtype=torch.int32))
            )
        )
    return torch.stack(results)


def attention(query, key, value, gate, selected, *, rounded_gate=False):
    out = []
    for q, g, ids in zip(query, gate, selected):
        ids = ids[ids >= 0].long()
        k = key[ids].repeat_interleave(12, 1).float()
        v = value[ids].repeat_interleave(12, 1).float()
        scores = torch.einsum("hd,shd->hs", q.float(), k) * 0.0625
        y = torch.einsum("hs,shd->hd", scores.softmax(-1), v)
        if rounded_gate:
            out.append(y.to(query.dtype) * torch.sigmoid(g))
        else:
            out.append((y * torch.sigmoid(g.float())).to(query.dtype))
    return torch.stack(out)


def quantize_kv(x):
    grouped = x.float().reshape(*x.shape[:-1], 4, 64)
    scale = (grouped.abs().amax(-1) / 127).clamp_min(2**-24).half()
    codes = (grouped / scale.float()[..., None]).round().clamp(-127, 127).to(torch.int8)
    return codes.reshape_as(x), scale


def dequantize_kv(codes, scale):
    return (codes.float().reshape(*codes.shape[:-1], 4, 64) * scale.float()[..., None]).flatten(-2)
