"""Lossless BF16 mantissa/sign bytes, five-bit exponents and sparse raw escapes."""

import numpy as np


def pack(raw):
    if raw.dtype != np.uint16 or raw.ndim < 1 or raw.shape[-1] != 64:
        raise ValueError("BF16 cache needs native uint16 bits in 64-element rows")
    exp = (raw >> 7) & 255
    base = max(0, int(exp.max()) - 30)
    escaped = (exp < base) | (exp > base + 30)
    codes = (exp.astype(np.int32) - base).astype(np.uint32)
    codes[escaped] = 31
    low = ((raw & 127) | ((raw >> 8) & 128)).astype(np.uint8)
    high = np.zeros((*raw.shape[:-1], 10), dtype=np.uint32)
    for j in range(64):
        pos = j * 5
        high[..., pos // 32] |= codes[..., j] << (pos % 32)
        if pos % 32 > 27:
            high[..., pos // 32 + 1] |= codes[..., j] >> (32 - pos % 32)
    where = np.flatnonzero(escaped).astype(np.uint64)
    patches = (where << np.uint64(16)) | raw.reshape(-1)[where.astype(np.int64)].astype(np.uint64)
    count = len(patches)
    if count == 0:
        patches = np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
    result = dict(low=low, high=high, patches=patches, patch_count=count, base=base)
    if not np.array_equal(unpack(result), raw):
        raise ValueError("Lossless BF16 cache round trip failed")
    return result


def unpack(encoded):
    lo = encoded["low"]
    hi = encoded["high"]
    out = np.empty_like(lo, dtype=np.uint16)
    for j in range(64):
        pos = j * 5
        q = hi[..., pos // 32] >> (pos % 32)
        if pos % 32 > 27:
            q |= hi[..., pos // 32 + 1] << (32 - pos % 32)
        m = lo[..., j].astype(np.uint16)
        out[..., j] = (
            (m & 127) | ((m & 128) << 8) | ((((q & 31) + encoded["base"]).astype(np.uint16)) << 7)
        )
    for patch in encoded["patches"][: encoded["patch_count"]]:
        out.reshape(-1)[int(patch) >> 16] = int(patch) & 65535
    return out
