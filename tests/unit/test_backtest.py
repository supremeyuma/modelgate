from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from modelgate.backtest.rolling_origin import (
    make_folds,
    recursive_forecast,
    run_backtest,
)
from modelgate.data.generator import generate_demand
from modelgate.features.build import build_features
from modelgate.features.contract import default_contract
from modelgate.train.baselines import moving_average, naive, seasonal_naive
from modelgate.train.pipeline import TrainingConfig, train_model


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return generate_demand(n_skus=3, n_days=220, seed=13)


def test_baselines_are_constructed_from_history_only() -> None:
    history = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert naive(history, 3).tolist() == [9.0, 9.0, 9.0]
    assert seasonal_naive(history, 8).tolist() == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 3.0]
    assert moving_average(history, 2, window=7) == pytest.approx([6.0, 6.0])
    with pytest.raises(ValueError):
        naive(np.array([]), 1)
    with pytest.raises(ValueError):
        seasonal_naive(np.array([1.0, 2.0]), 1)
    with pytest.raises(ValueError):
        moving_average(np.array([1.0, 2.0]), 1, window=7)
    from modelgate.train.baselines import forecast_baseline

    with pytest.raises(KeyError):
        forecast_baseline("nope", history, 1)


def test_backtest_folds_never_overlap_train_and_test(panel: pd.DataFrame) -> None:
    folds = make_folds(panel, horizon=14, n_folds=4)
    assert len(folds) == 4
    for fold in folds:
        assert fold.train_end < fold.test_start <= fold.test_end
        train_dates = set(panel.loc[panel["date"] <= fold.train_end, "date"])
        test_dates = set(
            panel.loc[(panel["date"] >= fold.test_start) & (panel["date"] <= fold.test_end), "date"]
        )
        assert train_dates.isdisjoint(test_dates)
    for previous, later in itertools.pairwise(folds):
        assert previous.test_end < later.test_start


def test_fold_guard_needs_enough_history(panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least"):
        make_folds(panel, horizon=28, n_folds=20)


def test_recursive_forecast_shape_and_ranges(panel: pd.DataFrame) -> None:
    config = TrainingConfig(n_estimators=15)
    origin = pd.Timestamp(panel["date"].unique()[-20])
    history = panel.loc[panel["date"] < origin]
    calendar = panel.loc[panel["date"] >= origin]
    model = train_model(history, config=config)
    forecast = recursive_forecast(model, history, calendar)
    assert len(forecast) == len(calendar)
    assert forecast["p10"].min() >= 0
    assert (forecast["p10"] <= forecast["p50"]).all()
    assert (forecast["p50"] <= forecast["p90"]).all()
    assert set(forecast["horizon"]) == set(range(1, 21))


def test_training_is_deterministic_for_seed(panel: pd.DataFrame) -> None:
    config = TrainingConfig(seed=7, n_estimators=20)
    first = train_model(panel, config=config)
    second = train_model(panel, config=config)
    matrix = build_features(panel, default_contract())
    assert first.dataset_hash == second.dataset_hash
    assert first.feature_version_hash == second.feature_version_hash
    np.testing.assert_allclose(
        first.predict_frame(matrix)["p50"], second.predict_frame(matrix)["p50"]
    )


def test_model_beats_naive_baseline_on_generated_data(panel: pd.DataFrame) -> None:
    result = run_backtest(
        panel,
        TrainingConfig(seed=3, n_estimators=40, horizon=14),
        n_folds=2,
        horizon=14,
    )
    model_wape = result.metrics["model"]["wape"]
    naive_wape = result.metrics["baselines"]["naive"]["wape"]
    seasonal_wape = result.metrics["baselines"]["seasonal_naive"]["wape"]
    assert model_wape < naive_wape, (model_wape, naive_wape)
    assert model_wape < seasonal_wape, (model_wape, seasonal_wape)


def test_backtest_report_contains_hashes_and_per_horizon_bias(panel: pd.DataFrame) -> None:
    result = run_backtest(
        panel, TrainingConfig(n_estimators=10, horizon=14), n_folds=2, horizon=14
    )
    report = result.to_report()
    assert len(report["dataset_hash"]) == 64
    assert len(report["feature_version_hash"]) == 64
    assert report["n_folds"] == 2
    for fold in report["metrics"]["per_fold"]:
        assert set(fold["model"]["bias_pct"]) == {str(step) for step in range(1, 15)}
        assert fold["model"]["pinball"]["p50"] >= 0
    assert set(report["metrics"]["baselines"]) == {"naive", "seasonal_naive", "moving_average"}
