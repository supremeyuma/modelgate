from __future__ import annotations

import numpy as np
import pandas as pd

from modelgate.data.generator import generate_demand


def _lag_autocorrelation(frame: pd.DataFrame, lag: int = 7) -> float:
    correlations: list[float] = []
    for _, group in frame.groupby("sku"):
        values = group.sort_values("date")["demand"].to_numpy(dtype=float)
        left = values[:-lag]
        right = values[lag:]
        correlation = float(np.corrcoef(left, right)[0, 1])
        if np.isfinite(correlation):
            correlations.append(correlation)
    return float(np.mean(correlations))


def test_generator_is_deterministic_for_seed() -> None:
    first = generate_demand(n_skus=4, n_days=120, seed=7)
    second = generate_demand(n_skus=4, n_days=120, seed=7)
    different = generate_demand(n_skus=4, n_days=120, seed=8)
    pd.testing.assert_frame_equal(first, second)
    assert not first["demand"].equals(different["demand"])


def test_generator_produces_expected_seasonality() -> None:
    frame = generate_demand(n_skus=6, n_days=400, seed=11)
    assert _lag_autocorrelation(frame, lag=7) > 0.30


def test_generator_has_promos_holidays_and_regime_shift() -> None:
    frame = generate_demand(n_skus=8, n_days=700, seed=3)
    assert frame["promo"].any()
    assert frame["holiday"].any()
    assert frame["demand"].min() >= 0
    assert frame["demand"].dtype == np.int64
    # SKU-001 is deliberately shifted down at day 600; the mean must drop.
    sku = frame.loc[frame["sku"] == "SKU-001"].sort_values("date")
    before = sku.iloc[500:590]["demand"].mean()
    after = sku.iloc[620:700]["demand"].mean()
    assert after < before * 0.95


def test_generator_rejects_bad_shape() -> None:
    import pytest

    with pytest.raises(ValueError):
        generate_demand(n_skus=0)
