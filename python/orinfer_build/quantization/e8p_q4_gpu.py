"""Bounded GPU fitting for the shared two-stage integer E8P-Q4 codec."""

import numpy as np
import torch

from orinfer_build.quantization.e8p_gpu import Encoder as E8PEncoder
from orinfer_build.quantization.e8p_q4 import Weights, validate_coefficients
from orinfer_build.quantization.vq import rotate


class Encoder:
    def __init__(self):
        self.e8p = E8PEncoder()

    def fit(self, raw, signs, *, coefficients=(4, 1), iterations=2):
        a, b = validate_coefficients(coefficients)
        raw, signs = np.asarray(raw, dtype=np.float32), np.asarray(signs)
        if (
            raw.ndim != 2
            or min(raw.shape) <= 0
            or raw.shape[1] % 128
            or not np.isfinite(raw).all()
            or signs.dtype != np.int8
            or signs.shape != (raw.shape[1],)
            or not np.isin(signs, [-1, 1]).all()
            or type(iterations) is not int
            or not 0 <= iterations <= 8
        ):
            raise ValueError("Invalid E8P-Q4 fitting inputs")
        w = torch.from_numpy(rotate(raw, signs)).cuda()
        n, k = w.shape
        initial = w.square().mean(1).sqrt() / (4 * a)
        best = torch.full((n,), float("inf"), device="cuda")
        codes = torch.empty((n, k // 8), device="cuda", dtype=torch.int64)
        scales = torch.empty(n, device="cuda", dtype=torch.float16)

        def quantize(x):
            ids = self.e8p.nearest(x.reshape(-1, 8)).long()
            return ids.reshape(n, k // 8), self.e8p.grid[ids].reshape(n, k)

        for multiplier in (0.9, 1.0, 1.1):
            scale = (initial * multiplier).clamp_min(2**-24).half()
            if not bool(torch.isfinite(scale).all()):
                raise ValueError("E8P-Q4 scale cannot be represented as FP16")
            first, q0 = quantize(w / (scale.float()[:, None] * a))
            second, q1 = quantize((w / scale.float()[:, None] - a * q0) / b)
            for _ in range(iterations):
                q = a * q0 + b * q1
                scale = (
                    ((w * q).sum(1) / q.square().sum(1).clamp_min(1e-30)).clamp_min(2**-24).half()
                )
                if not bool(torch.isfinite(scale).all()):
                    raise ValueError("E8P-Q4 scale cannot be represented as FP16")
                first, q0 = quantize((w / scale.float()[:, None] - b * q1) / a)
                second, q1 = quantize((w / scale.float()[:, None] - a * q0) / b)
            loss = (w - scale.float()[:, None] * (a * q0 + b * q1)).square().sum(1)
            better = loss < best
            codes = torch.where(better[:, None], first | (second << 16), codes)
            scales = torch.where(better, scale, scales)
            best = torch.minimum(best, loss)
        result = Weights(
            codes.cpu().numpy().astype(np.uint32), scales.cpu().numpy(), signs.copy(), coefficients
        )
        result.validate()
        return result

    def refine(self, raw, original, *, iterations=8):
        """Improve existing [4,1] codes with two exact coarse top-8 searches.

        Retain the incumbent at each row/scale boundary. This is the beam8
        weight-MSE search, not a claim of globally optimal additive coding.
        Callers bound memory by supplying output-row chunks (e.g. 128 rows).
        """
        n, k = original.validate()
        raw = np.asarray(raw, dtype=np.float32)
        if (
            raw.shape != (n, k)
            or not np.isfinite(raw).all()
            or original.coefficients != (4, 1)
            or type(iterations) is not int
            or not 0 <= iterations <= 8
        ):
            raise ValueError("Invalid E8P-Q4 refinement inputs")
        w = torch.from_numpy(rotate(raw, original.signs)).cuda()
        target = w.reshape(n, k // 8, 8)
        packed = torch.from_numpy(original.packed.astype(np.int64)).cuda()
        first, second = packed & 65535, packed >> 16
        scale = torch.from_numpy(original.scales).cuda()
        grid = self.e8p.grid
        q0, q1 = grid[first], grid[second]
        best_first, best_second, best_scale = first.clone(), second.clone(), scale.clone()
        best = (w - scale.float()[:, None] * (4 * q0 + q1).reshape(n, k)).square().sum(1)

        def keep():
            nonlocal best, best_first, best_second, best_scale
            loss = (w - scale.float()[:, None] * (4 * q0 + q1).reshape(n, k)).square().sum(1)
            better = loss < best
            best_first = torch.where(better[:, None], first, best_first)
            best_second = torch.where(better[:, None], second, best_second)
            best_scale = torch.where(better, scale, best_scale)
            best = torch.minimum(best, loss)

        def update_scale():
            q = (4 * q0 + q1).reshape(n, k)
            return ((w * q).sum(1) / q.square().sum(1).clamp_min(1e-30)).clamp_min(2**-24).half()

        def quantize(values):
            ids = self.e8p.nearest(values.reshape(-1, 8)).long().reshape(n, k // 8)
            return ids, grid[ids]

        for _ in range(iterations):
            scale = update_scale()
            keep()
            normalized = target / scale.float()[:, None, None]
            first, q0 = quantize((normalized - q1) / 4)
            second, q1 = quantize(normalized - 4 * q0)
            keep()
        first, second, scale = best_first, best_second, best_scale
        q0, q1 = grid[first], grid[second]
        for _ in range(2):
            normalized = target / scale.float()[:, None, None]
            candidates = self.e8p.nearest_top8((normalized / 4).reshape(-1, 8))
            candidates = candidates.long().reshape(n, k // 8, 8)
            residual = normalized[:, :, None, :] - 4 * grid[candidates]
            # All eight coarse candidates are independent at the current scale.
            # Search their residuals together rather than launching eight passes.
            residual_codes = self.e8p.nearest(residual.reshape(-1, 8)).long()
            residual_codes = residual_codes.reshape(n, k // 8, 8)
            errors = (residual - grid[residual_codes]).square().sum(-1)
            selected = errors.argmin(-1, keepdim=True)
            c0 = candidates.gather(-1, selected).squeeze(-1)
            c1 = residual_codes.gather(-1, selected).squeeze(-1)
            loss = errors.gather(-1, selected).squeeze(-1)
            incumbent = (normalized - 4 * q0 - q1).square().sum(-1)
            better = loss < incumbent
            first, second = torch.where(better, c0, first), torch.where(better, c1, second)
            q0, q1 = grid[first], grid[second]
            keep()
            scale = update_scale()
            keep()
        result = Weights(
            (best_first | (best_second << 16)).cpu().numpy().astype(np.uint32),
            best_scale.cpu().numpy(),
            original.signs.copy(),
        )
        result.validate()
        return result
