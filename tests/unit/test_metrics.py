from __future__ import annotations

import numpy as np
import pytest

from modelgate.eval.metrics import (
    bias_by_horizon,
    interval_coverage,
    mase,
    pinball_loss,
    smape,
    summarize,
    wape,
)


def test_wape_matches_hand_computed_example() -> None:
    actual = [10.0, 20.0, 30.0]
    predicted = [12.0, 18.0, 33.0]
    assert wape(actual, predicted) == pytest.approx(7.0 / 60.0)


def test_seasonal_naive_scores_mase_one_by_definition() -> None:
    # Seasonal-naive applied to the series it was scaled on: numerator and denominator are the
    # same set of seasonal differences, so MASE is exactly 1.
    history = np.array(
        [5.0, 7.0, 9.0, 11.0, 8.0, 6.0, 4.0, 6.0, 9.0, 12.0, 10.0, 7.0, 5.0, 3.0,
         4.0, 8.0, 10.0, 13.0, 11.0, 9.0, 7.0, 6.0, 8.0, 11.0, 14.0, 12.0, 9.0, 7.0]
    )
    assert mase(history[7:], history[:-7], history) == pytest.approx(1.0)


def test_mase_hand_example() -> None:
    history = np.array([1.0, 2.0, 4.0, 8.0])
    # seasonal scale at lag 2 = mean(|4-1|, |8-2|) = 4.5
    assert mase([10.0], [13.0], history, seasonality=2) == pytest.approx(3.0 / 4.5)


def test_pinball_loss_hand_example() -> None:
    # q=0.9, actual 10, prediction 8 -> 0.9 * 2 = 1.8; q=0.1 -> 0.1 * 2 = 0.2
    assert pinball_loss([10.0], [8.0], 0.9) == pytest.approx(1.8)
    assert pinball_loss([10.0], [8.0], 0.1) == pytest.approx(0.2)


def test_bias_by_horizon_is_signed_and_grouped() -> None:
    bias = bias_by_horizon(
        actual=[10.0, 10.0, 20.0],
        predicted=[11.0, 9.0, 18.0],
        horizon=[1, 1, 2],
    )
    assert bias[1] == pytest.approx(0.0)
    assert bias[2] == pytest.approx(-10.0)


def test_interval_coverage_and_smape() -> None:
    assert interval_coverage([5.0, 6.0], [4.0, 8.0], [7.0, 9.0]) == pytest.approx(0.5)
    assert smape([10.0], [12.0]) == pytest.approx(2 * 2 / 22)


def test_summarize_bundle_is_complete() -> None:
    bundle = summarize(
        actual=[10.0, 20.0],
        point=[11.0, 19.0],
        in_sample=np.arange(1.0, 30.0),
        horizon=[1, 2],
        p10=[9.0, 17.0],
        p90=[13.0, 21.0],
    )
    assert set(bundle) == {"wape", "mase", "bias_pct", "pinball", "interval_coverage_80"}
    assert set(bundle["pinball"]) == {"p10", "p50", "p90"}


def test_metric_guards() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        wape([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="empty"):
        wape([], [])
    with pytest.raises(ValueError, match="sums to zero"):
        wape([0.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="seasonal period"):
        mase([1.0], [1.0], [1.0, 2.0], seasonality=7)
    with pytest.raises(ValueError, match="scale is zero"):
        mase([1.0], [1.0], [3.0] * 20)
    with pytest.raises(ValueError, match="quantile"):
        pinball_loss([1.0], [1.0], 1.0)
    with pytest.raises(ValueError, match="horizon must align"):
        bias_by_horizon([1.0], [1.0], [1, 2])
    with pytest.raises(ValueError, match="interval bounds"):
        interval_coverage([1.0], [0.0], [1.0, 2.0])
