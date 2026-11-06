"""CPU activation-weighted precision screening, independent of device or model.

Candidates use symmetric integers and FP32 row scales. Native scale formats,
rounding boundaries and full-model quality require their separate acceptance.
"""

import numpy as np


def matrix(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite activation/weight matrix")
    return values


def relative_mse(candidate, reference):
    candidate, reference = matrix(candidate), matrix(reference)
    if candidate.shape != reference.shape:
        raise ValueError("Output shapes differ")
    numerator = np.square(candidate - reference, dtype=np.float64).sum()
    denominator = np.square(reference, dtype=np.float64).sum()
    return float(numerator / max(denominator, 1e-30))


def quantize_rows(values, bits=8, clip=1.0):
    values = matrix(values)
    if bits not in (4, 8) or not 0 < clip <= 1:
        raise ValueError("Expected INT4/INT8 and clipping in (0, 1]")
    positive = (1 << (bits - 1)) - 1
    scale = np.maximum(np.abs(values).max(1, keepdims=True) * clip / positive, 1e-30)
    codes = np.clip(np.rint(values / scale), -positive - 1, positive)
    return codes * scale


def statistics(activation):
    activation = matrix(activation)
    energy = np.square(activation, dtype=np.float64).sum(0)
    return dict(
        rows=len(activation),
        channels=activation.shape[1],
        mean_square=float(energy.sum() / activation.size),
        max_abs=float(np.abs(activation).max()),
        max_channel_energy_fraction=float(energy.max() / max(energy.sum(), 1e-30)),
    )


def screen_linear(
    weight, train, holdout, *, bits, quantize_activation=False, clips=(0.8, 0.9, 0.95, 1.0)
):
    """Fit clipping on training outputs; score disjoint held-out activations.

    Measuring X @ delta_W uses the sample covariance, including correlations.
    Joint A8 error is explicit so a BF16-to-W8A8 candidate cannot conceal it.
    """
    weight, train, holdout = map(matrix, (weight, train, holdout))
    if train.shape[1] != weight.shape[1] or holdout.shape[1] != weight.shape[1] or not clips:
        raise ValueError("Activation dimensions or clipping candidates differ")
    train_input = quantize_rows(train) if quantize_activation else train
    holdout_input = quantize_rows(holdout) if quantize_activation else holdout
    reference = train @ weight.T
    candidates = [
        (relative_mse(train_input @ quantize_rows(weight, bits, clip).T, reference), clip)
        for clip in clips
    ]
    loss, clip = min(candidates)
    selected = quantize_rows(weight, bits, clip)
    reference = holdout @ weight.T
    return dict(
        bits=bits,
        activation_bits=8 if quantize_activation else None,
        clip=clip,
        train_nmse=loss,
        holdout_nmse=relative_mse(holdout_input @ selected.T, reference),
        activation_only_nmse=relative_mse(holdout_input @ weight.T, reference),
        weight_only_nmse=relative_mse(holdout @ selected.T, reference),
    )
