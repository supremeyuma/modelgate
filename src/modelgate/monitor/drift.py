"""Drift detection: population stability index, residual KS and performance decay.

Thresholds are explicit and configurable. The tests enforce both directions: an injected shift
must alert, and stationary draws must stay under a 5% false-positive rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

ArrayLike = Sequence[float] | np.ndarray | pd.Series


@dataclass(frozen=True)
class DriftThresholds:
    psi: float = 0.20
    ks_p_value: float = 0.01
    wape_degradation_pct: float = 25.0


@dataclass
class DriftReport:
    psi: dict[str, float]
    alerts: list[str] = field(default_factory=list)
    ks_statistic: float = 0.0
    ks_p_value: float = 1.0
    rolling_wape: float | None = None

    @property
    def drifted(self) -> bool:
        return bool(self.alerts)


def population_stability_index(
    reference: ArrayLike, current: ArrayLike, bins: int = 10, epsilon: float = 1e-6
) -> float:
    """PSI with quantile bins from the reference distribution.

    Values below 0.1 are conventionally stable, 0.1-0.2 monitor, above 0.2 shift.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        raise ValueError("PSI needs non-empty samples")
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return 0.0 if float(ref.mean()) == float(cur.mean()) else float("inf")
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_share = (ref_counts + epsilon) / (ref.size + epsilon * ref_counts.size)
    cur_share = (cur_counts + epsilon) / (cur.size + epsilon * cur_counts.size)
    return float(np.sum((ref_share - cur_share) * np.log(ref_share / cur_share)))


def ks_residuals(reference: ArrayLike, current: ArrayLike) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov on forecast residuals."""
    result = stats.ks_2samp(
        np.asarray(reference, dtype=float).ravel(),
        np.asarray(current, dtype=float).ravel(),
    )
    return float(result.statistic), float(result.pvalue)


def rolling_wape(
    actual: ArrayLike, predicted: ArrayLike, window: int = 7
) -> np.ndarray:
    y = np.asarray(actual, dtype=float).ravel()
    yhat = np.asarray(predicted, dtype=float).ravel()
    if y.size != yhat.size or y.size < window:
        raise ValueError("series must align and be at least one window long")
    values: list[float] = []
    for end in range(window, y.size + 1):
        denominator = float(np.abs(y[end - window : end]).sum())
        values.append(
            float(np.abs(y[end - window : end] - yhat[end - window : end]).sum())
            / max(denominator, 1e-9)
        )
    return np.asarray(values, dtype=float)


def detect_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Sequence[str],
    thresholds: DriftThresholds | None = None,
) -> DriftReport:
    limits = thresholds or DriftThresholds()
    scores: dict[str, float] = {}
    alerts: list[str] = []
    for column in columns:
        if column not in reference.columns or column not in current.columns:
            raise KeyError(column)
        score = population_stability_index(reference[column], current[column])
        scores[column] = score
        if score > limits.psi:
            alerts.append(f"PSI {score:.3f} > {limits.psi} on feature {column}")
    return DriftReport(psi=scores, alerts=alerts)


def detect_performance_decay(
    expected_wape: float,
    current: pd.Series | ArrayLike,
    thresholds: DriftThresholds | None = None,
) -> DriftReport:
    limits = thresholds or DriftThresholds()
    series = (
        current.to_numpy(dtype=float)
        if isinstance(current, pd.Series)
        else np.asarray(current, dtype=float)
    )
    if series.size == 0:
        raise ValueError("empty WAPE series")
    latest = float(series[-1])
    alerts: list[str] = []
    if expected_wape <= 0:
        raise ValueError("expected WAPE must be positive")
    degradation = (latest - expected_wape) / expected_wape * 100.0
    if degradation > limits.wape_degradation_pct:
        alerts.append(
            f"rolling WAPE {latest:.4f} is {degradation:.1f}% above expectation "
            f"{expected_wape:.4f}"
        )
    return DriftReport(psi={}, alerts=alerts, rolling_wape=latest)
