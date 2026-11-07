"""Offline prefix-survival calibration shared by speculative decoder backends."""

import numpy as np


def calibrated_survival(confidence, temperatures):
    confidence = np.asarray(confidence, dtype="f8")
    temperatures = np.asarray(temperatures, dtype="f8")
    if (
        confidence.ndim != 2
        or temperatures.shape != (confidence.shape[1],)
        or not np.isfinite(confidence).all()
        or np.any((confidence < 0) | (confidence > 1))
        or not np.isfinite(temperatures).all()
        or np.any(temperatures <= 0)
    ):
        raise ValueError("Require finite confidence rows and positive position temperatures")
    values = np.clip(confidence, 1e-7, 1 - 1e-7)
    logits = (np.log(values) - np.log1p(-values)) / temperatures
    conditional = 1 / (1 + np.exp(np.clip(-logits, -80, 80)))
    return np.cumprod(conditional, axis=1)


def survival_labels(accepted, width):
    accepted = np.asarray(accepted)
    if (
        accepted.ndim != 1
        or not np.issubdtype(accepted.dtype, np.integer)
        or np.any((accepted < 0) | (accepted > width))
    ):
        raise ValueError("Accepted lengths must lie in the verified window")
    return accepted[:, None] > np.arange(width)[None, :]


def fit_temperatures(confidence, accepted, grid=None):
    """Sequential temperature scaling with cumulative Brier loss.

    After a rejection, cumulative survival labels are zero; these are not
    interpreted as independent conditional rejections of the remaining tokens.
    """
    confidence = np.asarray(confidence, dtype="f8")
    if confidence.ndim != 2 or not len(confidence) or not confidence.shape[1]:
        raise ValueError("Require nonempty calibration windows")
    labels = survival_labels(accepted, confidence.shape[1])
    if len(labels) != len(confidence):
        raise ValueError("Confidence and accepted lengths differ")
    grid = np.unique(np.append(np.geomspace(0.25, 8, 65) if grid is None else grid, 1.0))
    temperatures = np.ones(confidence.shape[1])
    calibrated_survival(confidence, temperatures)
    for position in range(confidence.shape[1]):
        best = None
        for value in grid:
            candidate = temperatures.copy()
            candidate[position] = value
            survival = calibrated_survival(confidence, candidate)[:, position]
            score = float(np.mean((survival - labels[:, position]) ** 2))
            identity = (score, abs(np.log(value)))
            if best is None or identity < best[0]:
                best = identity, value
        temperatures[position] = best[1]
    return temperatures


def calibration_metrics(confidence, accepted, temperatures):
    survival = calibrated_survival(confidence, temperatures)
    labels = survival_labels(accepted, survival.shape[1])
    if len(labels) != len(survival) or not len(labels):
        raise ValueError("Require matched nonempty evaluation windows")
    errors = []
    for position in range(survival.shape[1]):
        error = 0.0
        bins = np.minimum((survival[:, position] * 10).astype(int), 9)
        for index in range(10):
            rows = bins == index
            if np.any(rows):
                error += rows.mean() * abs(
                    survival[rows, position].mean() - labels[rows, position].mean()
                )
        errors.append(float(error))
    return dict(
        windows=len(survival),
        cumulative_brier=float(np.mean((survival - labels) ** 2)),
        prefix_ece=errors,
        predicted_committed_per_round=float(1 + survival.sum(1).mean()),
        actual_committed_per_round=float(1 + np.asarray(accepted).mean()),
    )
