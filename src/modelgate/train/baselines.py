"""Naive baselines that any model must beat before a gate conversation starts.

Forecasts are produced from the observed history only (the same information a model gets) and are
constant across the horizon for naive/moving-average, or repeat the last observed week for the
seasonal naive.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def naive(history: NDArray[np.float64], horizon: int) -> NDArray[np.float64]:
    if history.size == 0:
        raise ValueError("history is empty")
    return np.full(horizon, float(history[-1]))


def seasonal_naive(
    history: NDArray[np.float64], horizon: int, seasonality: int = 7
) -> NDArray[np.float64]:
    if history.size < seasonality:
        raise ValueError("history shorter than the seasonal period")
    last_season = history[-seasonality:]
    return np.array([last_season[step % seasonality] for step in range(horizon)], dtype=float)


def moving_average(
    history: NDArray[np.float64], horizon: int, window: int = 7
) -> NDArray[np.float64]:
    if history.size < window:
        raise ValueError("history shorter than the moving-average window")
    return np.full(horizon, float(np.mean(history[-window:])))


BASELINES = ("naive", "seasonal_naive", "moving_average")


def forecast_baseline(name: str, history: NDArray[np.float64], horizon: int) -> NDArray[np.float64]:
    if name == "naive":
        return naive(history, horizon)
    if name == "seasonal_naive":
        return seasonal_naive(history, horizon)
    if name == "moving_average":
        return moving_average(history, horizon)
    raise KeyError(name)
