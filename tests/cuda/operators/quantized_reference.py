"""Exact-integer GPU validation helpers independent of weight codecs."""

import numpy as np
import torch


def upload(bank):
    storage = [w.gpu_layout() for w in bank]
    return tuple(torch.from_numpy(np.stack([s[i] for s in storage])).cuda() for i in range(3))


def reference(a, integer, ws, sa):
    # FP64 dot is an independent exact integer oracle for these bounded shapes.
    dot = torch.bmm(a.double(), integer.double().transpose(1, 2)).float()
    return ((dot * ws.float()[:, None, :]) * sa.float()[..., None]).half()
