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


def fit_output_scales(products, reference, scales, weights=None, *, shrink=0.5, limit=0.05):
    """Bounded weighted least squares for fixed integer projection codes."""
    products, reference = matrix(products), matrix(reference)
    scales = np.asarray(scales, dtype=np.float32)
    if products.shape != reference.shape or scales.shape != (products.shape[1],):
        raise ValueError("Projection products, reference and scales differ")
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Projection scales must be positive and finite")
    if not 0 <= shrink <= 1 or not 0 <= limit < 1:
        raise ValueError("Scale shrink and relative limit are outside their bounds")
    weights = np.ones_like(products) if weights is None else np.asarray(weights, dtype=np.float32)
    if weights.shape != products.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Projection weights must be finite, nonnegative and shape matched")
    denominator = np.sum(weights * products * products, axis=0, dtype=np.float64)
    numerator = np.sum(weights * products * reference, axis=0, dtype=np.float64)
    fitted = np.divide(
        numerator, denominator, out=scales.astype(np.float64), where=denominator > 1e-20
    )
    fitted = np.clip(fitted, scales * (1 - limit), scales * (1 + limit))
    return (scales + shrink * (fitted - scales)).astype(np.float32)


def refine_integer_codes(activation, reference, codes, scales, weights, *, steps=4):
    """Greedy weighted coordinate descent with one legal INT4 update per output/step.

    The sample covariance enters the residual update. Channels and physical order
    stay fixed; this changes integer rounding without adding runtime metadata.
    """
    activation, reference, weights = map(matrix, (activation, reference, weights))
    codes, scales = np.asarray(codes), np.asarray(scales, dtype="f4")
    if (
        codes.ndim != 2
        or not np.issubdtype(codes.dtype, np.integer)
        or codes.shape != (reference.shape[1], activation.shape[1])
        or len(activation) != len(reference)
        or weights.shape != reference.shape
        or scales.shape != (len(codes),)
        or np.any(codes < -8)
        or np.any(codes > 7)
        or np.any(weights < 0)
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
        or not 1 <= steps <= 16
    ):
        raise ValueError(
            "Require matched activations, nonnegative importance and signed INT4 codes"
        )
    codes = codes.astype("i1", copy=True)
    residual = reference - (activation @ codes.astype("f4").T) * scales
    diagonal = weights.T @ np.square(activation)
    curvature = np.square(scales[:, None]) * diagonal
    outputs = np.arange(len(codes))
    for _ in range(steps):
        gradient = (weights * residual).T @ activation
        plus = np.where(codes < 7, curvature - 2 * scales[:, None] * gradient, np.inf)
        minus = np.where(codes > -8, curvature + 2 * scales[:, None] * gradient, np.inf)
        gain = np.minimum(plus, minus)
        columns = gain.argmin(1)
        active = outputs[gain[outputs, columns] < 0]
        if not len(active):
            break
        columns = columns[active]
        direction = np.where(plus[active, columns] < minus[active, columns], 1, -1).astype("i1")
        codes[active, columns] += direction
        residual[:, active] -= activation[:, columns] * (direction * scales[active])
    return codes


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
