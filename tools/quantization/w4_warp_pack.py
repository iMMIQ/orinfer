"""Offline lossless adjacent-U4 to N64/K128 warp-MMA storage permutation."""

import numpy as np
import torch

LAYOUT = "u4_warp_n64_k128_mma_f16"


def pack_array(packed, *, verify=True):
    assert packed.dtype == np.uint8 and packed.ndim == 2 and np.little_endian
    n, halfk = packed.shape
    k = halfk * 2
    assert n > 0 and n % 64 == 0 and k % 128 == 0
    q = np.empty((n, k), dtype=np.uint8)
    q[:, 0::2] = packed & 15
    q[:, 1::2] = packed >> 4
    ordered = (
        q.reshape(n // 64, 4, 2, 8, k // 128, 8, 2, 4, 2)
        .transpose(0, 4, 1, 3, 7, 5, 2, 6, 8)
        .copy()
    )
    codes = ordered.reshape(n // 64, k // 128, 128, 8, 8)
    result = np.zeros(codes.shape[:-1], dtype=np.uint32)
    for i in range(8):
        result |= codes[..., i].astype(np.uint32) << (i * 4)
    if verify:
        decoded = np.stack([(result >> (i * 4)) & 15 for i in range(8)], -1).astype(np.uint8)
        back = (
            decoded.reshape(ordered.shape).transpose(0, 2, 6, 3, 1, 5, 7, 4, 8).copy().reshape(n, k)
        )
        assert np.array_equal(back, q)
    assert result.nbytes == packed.nbytes
    return result


def pack(packed, *, verify=True):
    assert packed.device.type == "cpu" and packed.dtype == torch.uint8
    return torch.from_numpy(pack_array(packed.numpy(), verify=verify).view(np.int32))
