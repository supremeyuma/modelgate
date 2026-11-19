"""Forecast metrics with explicit definitions.

All metrics are computed from arrays; the definitions are part of the contract and are locked by
golden-value tests:

* WAPE = sum |y - yhat| / sum y. Scale free, but zero-inflated targets need care.
* MASE = mean |y - yhat| / mean |y[t] - y[t-s]|, with the denominator measured in-sample.
  A seasonal-naive forecast evaluated on its own scale therefore scores exactly 1.
* Bias at horizon h = (sum yhat_h - sum y_h) / sum y_h, in percent. Positive means over-forecast.
* Pinball loss = mean of the asymmetric absolute loss at quantile q.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def _arrays(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(actual, dtype=float).ravel()
    yhat = np.asarray(predicted, dtype=float).ravel()
    if y.shape != yhat.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {yhat.shape}")
    if y.size == 0:
        raise ValueError("empty input")
    return y, yhat


def wape(actual: ArrayLike, predicted: ArrayLike) -> float:
    y, yhat = _arrays(actual, predicted)
    denominator = float(np.abs(y).sum())
    if denominator == 0:
        raise ValueError("WAPE is undefined when actual demand sums to zero")
    return float(np.abs(y - yhat).sum() / denominator)


def smape(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Symmetric MAPE; reported alongside WAPE for readers who expect it."""
    y, yhat = _arrays(actual, predicted)
    denominator = np.abs(y) + np.abs(yhat)
    ratio = np.divide(
        2.0 * np.abs(y - yhat),
        denominator,
        out=np.zeros_like(y),
        where=denominator != 0,
    )
    return float(np.mean(ratio))


def mase(
    actual: ArrayLike,
    predicted: ArrayLike,
    in_sample: ArrayLike,
    seasonality: int = 7,
) -> float:
    y, yhat = _arrays(actual, predicted)
    history = np.asarray(in_sample, dtype=float).ravel()
    if history.size <= seasonality:
        raise ValueError("in-sample series shorter than the seasonal period")
    scale = float(np.mean(np.abs(history[seasonality:] - history[:-seasonality])))
    if scale == 0:
        raise ValueError("MASE scale is zero (perfectly periodic history)")
    return float(np.mean(np.abs(y - yhat)) / scale)


def pinball_loss(actual: ArrayLike, predicted: ArrayLike, quantile: float) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between 0 and 1")
    y, yhat = _arrays(actual, predicted)
    residual = y - yhat
    loss = np.maximum(quantile * residual, (quantile - 1.0) * residual)
    return float(np.mean(loss))


def bias_by_horizon(
    actual: ArrayLike, predicted: ArrayLike, horizon: Sequence[int] | np.ndarray
) -> dict[int, float]:
    y, yhat = _arrays(actual, predicted)
    steps = np.asarray(horizon, dtype=int).ravel()
    if steps.shape != y.shape:
        raise ValueError("horizon must align with observations")
    result: dict[int, float] = {}
    for step in sorted(set(int(value) for value in steps)):
        mask = steps == step
        total = float(y[mask].sum())
        if total == 0:
            result[step] = 0.0
        else:
            result[step] = float((yhat[mask].sum() - total) / total * 100.0)
    return result


def interval_coverage(actual: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    y, low = _arrays(actual, lower)
    high = np.asarray(upper, dtype=float).ravel()
    if high.shape != y.shape:
        raise ValueError("interval bounds must align")
    return float(np.mean((y >= low) & (y <= high)))


def summarize(
    actual: ArrayLike,
    point: ArrayLike,
    in_sample: ArrayLike,
    horizon: Sequence[int],
    p10: ArrayLike,
    p90: ArrayLike,
    seasonality: int = 7,
) -> dict[str, object]:
    """The per-fold metric bundle written into the backtest report."""
    return {
        "wape": wape(actual, point),
        "mase": mase(actual, point, in_sample, seasonality=seasonality),
        "bias_pct": bias_by_horizon(actual, point, horizon),
        "pinball": {
            "p10": pinball_loss(actual, p10, 0.10),
            "p50": pinball_loss(actual, point, 0.50),
            "p90": pinball_loss(actual, p90, 0.90),
        },
        "interval_coverage_80": interval_coverage(actual, p10, p90),
    }
