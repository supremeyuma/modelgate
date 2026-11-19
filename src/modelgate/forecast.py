"""Recursive multi-step forecasting with a fitted model.

Kept separate from the backtest so serving uses exactly the same procedure. Each step appends the
mean prediction to the history and rebuilds the feature row, which is why the point-in-time
guarantee of the feature builder is the safety net here too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modelgate.features.build import compute_feature_frame
from modelgate.features.contract import FeatureContract, default_contract
from modelgate.train.pipeline import TrainedModel


def recursive_forecast(
    model: TrainedModel,
    history: pd.DataFrame,
    calendar: pd.DataFrame,
    contract: FeatureContract | None = None,
) -> pd.DataFrame:
    """Forecast every SKU for the calendar days, one day at a time."""
    active = contract or default_contract()
    buffer = history.loc[:, ["sku", "date", "family", "demand", "promo", "holiday"]].copy()
    buffer["demand"] = buffer["demand"].astype(float)
    days = pd.Series(pd.to_datetime(calendar["date"]).unique()).sort_values().reset_index(drop=True)
    records: list[pd.DataFrame] = []

    for step, day in enumerate(days, start=1):
        current = calendar.loc[pd.to_datetime(calendar["date"]) == day, :]
        future = current.loc[:, ["sku", "date", "family", "promo", "holiday"]].copy()
        future["demand"] = np.nan
        combined = pd.concat([buffer, future], ignore_index=True)
        computed = compute_feature_frame(combined, active)
        target_rows = computed.loc[computed["date"] == day].copy()
        predictions = model.predict_frame(target_rows, [step] * len(target_rows))
        record = target_rows.loc[:, ["sku", "date"]].copy()
        record["horizon"] = step
        record["mean"] = predictions["mean"]
        record["p10"] = predictions["p10"]
        record["p50"] = predictions["p50"]
        record["p90"] = predictions["p90"]
        records.append(record)
        future["demand"] = predictions["mean"]
        buffer = pd.concat([buffer, future], ignore_index=True)

    return pd.concat(records, ignore_index=True)
