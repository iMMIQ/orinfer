"""Bounded GPU fitting for the shared signed-block128 integer Q4 codec."""

import numpy as np
import torch

from orinfer_build.quantization.row_q4 import MAGNITUDES, Weights


def fit(raw, signs):
    raw = np.asarray(raw)
    signs = np.asarray(signs)
    if raw.ndim != 2 or raw.shape[1] % 128 or signs.dtype != np.int8:
        raise ValueError("Expected a block128 matrix and INT8 rotation signs")
    n, k = raw.shape
    if signs.shape != (k,) or not np.isin(signs, [-1, 1]).all() or not np.isfinite(raw).all():
        raise ValueError("Invalid finite Q4 inputs")
    w = torch.from_numpy(raw.astype(np.float32)).cuda()
    w *= torch.from_numpy(signs).cuda()
    for stage in range(7):
        width = 1 << stage
        view = w.reshape(n, -1, 2 * width)
        left, right = view[..., :width].clone(), view[..., width:].clone()
        view[..., :width], view[..., width:] = left + right, left - right
    w *= 128**-0.5
    magnitude = w.abs()
    levels = torch.tensor(MAGNITUDES, device="cuda", dtype=torch.float32)
    edges = (levels[:-1] + levels[1:]) / 2
    initial = w.square().mean(1).sqrt() * (2.7 / 127)
    best = torch.full((n,), float("inf"), device="cuda")
    codes = torch.empty((n, k), dtype=torch.uint8, device="cuda")
    scale_out = torch.empty(n, dtype=torch.float16, device="cuda")
    for multiplier in (0.9, 1.0, 1.15):
        scale = (initial * multiplier).clamp_min(2**-24)
        for _ in range(4):
            indices = torch.bucketize(magnitude / scale[:, None], edges)
            decoded = levels[indices]
            scale = (
                (magnitude * decoded).sum(1) / decoded.square().sum(1).clamp_min(1e-30)
            ).clamp_min(2**-24)
        scale = scale.half()
        if not bool(torch.isfinite(scale).all()):
            raise ValueError("Q4 row scale cannot be represented as FP16")
        indices = torch.bucketize(magnitude / scale.float()[:, None], edges)
        loss = (magnitude - levels[indices] * scale.float()[:, None]).square().sum(1)
        better = loss < best
        candidate = indices.to(torch.uint8) | ((w < 0).to(torch.uint8) * 8)
        codes = torch.where(better[:, None], candidate, codes)
        scale_out = torch.where(better, scale, scale_out)
        best = torch.minimum(best, loss)
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    result = Weights(packed.cpu().numpy(), scale_out.cpu().numpy(), signs.copy())
    result.validate()
    return result
