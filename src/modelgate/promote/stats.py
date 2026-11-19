"""Statistical tests for the promotion gate.

All tests operate on *per-observation* loss differentials so that every decision is paired and
sample-size aware:

* Diebold-Mariano (Harvey small-sample variance) on absolute errors.
* A seeded bootstrap interval on the WAPE delta.
* Worst-decile weekly WAPE for tail risk.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

ArrayLike = Sequence[float] | np.ndarray


def _arrays(*values: ArrayLike) -> list[np.ndarray]:
    arrays = [np.asarray(value, dtype=float).ravel() for value in values]
    size = arrays[0].size
    if size == 0 or any(array.size != size for array in arrays):
        raise ValueError("inputs must be non-empty and the same length")
    return arrays


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float
    n: int
    mean_differential: float


def diebold_mariano(
    loss_champion: ArrayLike, loss_challenger: ArrayLike, min_samples: int = 30
) -> DMResult:
    """One-sided test: H1 is that the challenger has the lower loss.

    ``statistic > 0`` means the champion lost more (the challenger is better).
    """
    champion, challenger = _arrays(loss_champion, loss_challenger)
    if champion.size < min_samples:
        raise ValueError(f"need at least {min_samples} paired observations")
    differential = champion - challenger
    mean = float(differential.mean())
    variance = float(differential.var(ddof=1))
    if variance == 0.0:
        if mean == 0.0:
            return DMResult(0.0, 0.5, int(champion.size), mean)
        return DMResult(float("inf") if mean > 0 else float("-inf"), 0.0, int(champion.size), mean)
    statistic = mean / float(np.sqrt(variance / champion.size))
    p_value = float(1.0 - stats.norm.cdf(statistic))
    return DMResult(statistic, p_value, int(champion.size), mean)


def bootstrap_wape_delta(
    actual: ArrayLike,
    champion: ArrayLike,
    challenger: ArrayLike,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap interval for WAPE(challenger) - WAPE(champion).

    Negative values favour the challenger.
    """
    y, y_c, y_h = _arrays(actual, champion, challenger)
    rng = np.random.default_rng(seed)
    size = y.size
    deltas = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = rng.integers(0, size, size=size)
        denominator = float(np.abs(y[sample]).sum())
        if denominator == 0:
            deltas[index] = 0.0
            continue
        deltas[index] = (
            float(np.abs(y[sample] - y_h[sample]).sum())
            - float(np.abs(y[sample] - y_c[sample]).sum())
        ) / denominator
    observed = float(
        (np.abs(y - y_h).sum() - np.abs(y - y_c).sum()) / max(float(np.abs(y).sum()), 1e-9)
    )
    low = float(np.quantile(deltas, alpha / 2.0))
    high = float(np.quantile(deltas, 1.0 - alpha / 2.0))
    return observed, low, high


def window_wapes(actual: ArrayLike, predicted: ArrayLike, window: int = 7) -> np.ndarray:
    y, yhat = _arrays(actual, predicted)
    if y.size < window:
        raise ValueError("series shorter than one window")
    chunks = [
        float(np.abs(y[start : start + window] - yhat[start : start + window]).sum())
        / max(float(np.abs(y[start : start + window]).sum()), 1e-9)
        for start in range(0, y.size - window + 1, window)
    ]
    return np.asarray(chunks, dtype=float)


def worst_decile_wape(actual: ArrayLike, predicted: ArrayLike, window: int = 7) -> float:
    """Mean WAPE of the worst 10% of contiguous windows (at least one window)."""
    wapes = window_wapes(actual, predicted, window)
    count = max(1, int(np.ceil(0.10 * wapes.size)))
    return float(np.mean(np.sort(wapes)[-count:]))


def tail_degradation(
    actual: ArrayLike, champion: ArrayLike, challenger: ArrayLike, window: int = 7
) -> float:
    """Relative degradation of the challenger's worst-decile WAPE, in percent."""
    champion_tail = worst_decile_wape(actual, champion, window)
    challenger_tail = worst_decile_wape(actual, challenger, window)
    if champion_tail == 0:
        return 0.0 if challenger_tail == 0 else float("inf")
    return float((challenger_tail - champion_tail) / champion_tail * 100.0)
