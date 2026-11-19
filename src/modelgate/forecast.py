"""Recursive multi-step forecasting with a fitted model.

Kept separate from the backtest so serving uses exactly the same procedure. Each step appends the
mean prediction to the history and computes the next feature row directly from the demand tail.
The incremental calculation is checked against the full feature matrix in the unit tests, and the
point-in-time guarantee of the feature builder is the safety net here too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modelgate.features.build import tail_feature_vector
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
    latest = history.sort_values(["sku", "date"]).groupby("sku", sort=True).tail(1)
    series: dict[str, np.ndarray] = {}
    family: dict[str, str] = {}
    for _, row in latest.iterrows():
        sku = str(row["sku"])
        values = (
            history.loc[history["sku"] == sku]
            .sort_values("date")["demand"]
            .to_numpy(dtype=float)
        )
        series[sku] = values
        family[sku] = str(row["family"])

    days = pd.Series(pd.to_datetime(calendar["date"]).unique()).sort_values().reset_index(drop=True)
    flag_map = {
        (str(sku), pd.Timestamp(day)): (bool(promo), bool(holiday))
        for sku, day, promo, holiday in zip(
            calendar["sku"],
            pd.to_datetime(calendar["date"]),
            calendar["promo"],
            calendar["holiday"],
            strict=True,
        )
    }
    names = list(model.feature_names)
    records: list[dict[str, object]] = []

    for step, day in enumerate(days, start=1):
        rows: list[list[float]] = []
        keys: list[str] = []
        for sku in sorted(series):
            promo, holiday = flag_map.get((sku, pd.Timestamp(day)), (False, False))
            features = tail_feature_vector(
                active, series[sku], pd.Timestamp(day), promo, holiday
            )
            rows.append([float(features[name]) for name in names])
            keys.append(sku)
        matrix = np.asarray(rows, dtype=float)
        predictions = model.predict_matrix(matrix, [step] * len(rows))
        for index, sku in enumerate(keys):
            records.append(
                {
                    "sku": sku,
                    "date": day,
                    "horizon": step,
                    "mean": float(predictions["mean"][index]),
                    "p10": float(predictions["p10"][index]),
                    "p50": float(predictions["p50"][index]),
                    "p90": float(predictions["p90"][index]),
                }
            )
            series[sku] = np.append(series[sku], float(predictions["mean"][index]))

    return pd.DataFrame.from_records(records)
