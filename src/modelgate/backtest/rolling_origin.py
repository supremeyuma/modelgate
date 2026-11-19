"""Rolling-origin, expanding-window backtest with baselines.

Folds are contiguous blocks of ``horizon`` days. For fold *i* the model is trained on every row
strictly before the block and forecasts the block recursively, feeding its own median prediction
back into the lag features. Test blocks never overlap and never intersect the training window.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modelgate.config import REPORTS_DIR
from modelgate.data.generator import generate_demand
from modelgate.eval.metrics import bias_by_horizon, mase, pinball_loss, wape
from modelgate.features.contract import FeatureContract, default_contract
from modelgate.forecast import recursive_forecast
from modelgate.hashing import dataset_hash
from modelgate.train.baselines import forecast_baseline
from modelgate.train.pipeline import TrainingConfig, train_model


@dataclass(frozen=True)
class Fold:
    index: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    horizon: int


@dataclass
class BacktestResult:
    folds: list[Fold]
    metrics: dict[str, Any]
    observations: pd.DataFrame
    dataset_hash: str
    feature_version_hash: str
    config: TrainingConfig

    def to_report(self) -> dict[str, Any]:
        return {
            "dataset_hash": self.dataset_hash,
            "feature_version_hash": self.feature_version_hash,
            "horizon": self.config.horizon,
            "n_folds": len(self.folds),
            "seed": self.config.seed,
            "folds": [
                {
                    "index": fold.index,
                    "train_end": fold.train_end.strftime("%Y-%m-%d"),
                    "test_start": fold.test_start.strftime("%Y-%m-%d"),
                    "test_end": fold.test_end.strftime("%Y-%m-%d"),
                    "horizon": fold.horizon,
                }
                for fold in self.folds
            ],
            "metrics": self.metrics,
        }


def make_folds(
    frame: pd.DataFrame,
    horizon: int = 28,
    n_folds: int = 6,
    min_train_days: int = 120,
) -> list[Fold]:
    dates = pd.Series(pd.to_datetime(frame["date"]).unique()).sort_values().reset_index(drop=True)
    required = n_folds * horizon + min_train_days
    if len(dates) < required:
        raise ValueError(f"need at least {required} distinct dates, found {len(dates)}")
    folds: list[Fold] = []
    for index in range(n_folds):
        start = len(dates) - (n_folds - index) * horizon
        train_end = pd.Timestamp(dates.iloc[start - 1])
        test_start = pd.Timestamp(dates.iloc[start])
        test_end = pd.Timestamp(dates.iloc[start + horizon - 1])
        folds.append(
            Fold(
                index=index,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                horizon=horizon,
            )
        )
    return folds


def _baseline_predictions(
    history: pd.DataFrame,
    calendar: pd.DataFrame,
    horizon: int,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, list[pd.DataFrame]] = {
        "naive": [],
        "seasonal_naive": [],
        "moving_average": [],
    }
    days = pd.Series(pd.to_datetime(calendar["date"]).unique()).sort_values().reset_index(drop=True)
    for sku, group in history.groupby("sku", sort=True):
        values = group.sort_values("date")["demand"].to_numpy(dtype=float)
        for name in frames:
            forecast = forecast_baseline(name, values, len(days))
            frames[name].append(
                pd.DataFrame(
                    {
                        "sku": sku,
                        "date": days,
                        "horizon": np.arange(1, len(days) + 1),
                        name: forecast,
                    }
                )
            )
    return {name: pd.concat(parts, ignore_index=True) for name, parts in frames.items()}


def _mase_by_sku(
    observations: pd.DataFrame, history: pd.DataFrame, prediction: str
) -> float:
    values: list[float] = []
    for sku, group in observations.groupby("sku", sort=True):
        sample = history.loc[history["sku"] == sku].sort_values("date")["demand"].to_numpy(float)
        ordered = group.sort_values("date")
        values.append(
            mase(
                ordered["actual"].to_numpy(float),
                ordered[prediction].to_numpy(float),
                sample,
            )
        )
    return float(np.mean(values))


def _fold_metrics(
    observations: pd.DataFrame, history: pd.DataFrame, column: str, *, quantiles: bool
) -> dict[str, Any]:
    actual = observations["actual"].to_numpy(float)
    horizon = observations["horizon"].to_numpy(int)
    predictions = observations[column].to_numpy(float)
    bias = bias_by_horizon(actual, predictions, horizon)
    bundle: dict[str, Any] = {
        "wape": wape(actual, predictions),
        "mase": _mase_by_sku(observations, history, column),
        "bias_pct": {str(step): value for step, value in bias.items()},
        "bias_abs_max_pct": float(max(abs(value) for value in bias.values())),
    }
    if quantiles:
        bundle["pinball"] = {
            "p10": pinball_loss(actual, observations["p10"].to_numpy(float), 0.10),
            "p50": pinball_loss(actual, observations["p50"].to_numpy(float), 0.50),
            "p90": pinball_loss(actual, observations["p90"].to_numpy(float), 0.90),
        }
    return bundle


def run_backtest(
    frame: pd.DataFrame,
    config: TrainingConfig | None = None,
    n_folds: int = 6,
    contract: FeatureContract | None = None,
    horizon: int | None = None,
) -> BacktestResult:
    active = contract or default_contract()
    settings = config or TrainingConfig()
    steps = horizon or settings.horizon
    folds = make_folds(frame, horizon=steps, n_folds=n_folds)
    per_fold: list[dict[str, Any]] = []
    observations: list[pd.DataFrame] = []

    for fold in folds:
        history = frame.loc[frame["date"] < fold.test_start].copy()
        calendar = frame.loc[
            (frame["date"] >= fold.test_start) & (frame["date"] <= fold.test_end)
        ].copy()
        if history.empty or calendar.empty:
            raise ValueError(f"empty fold {fold.index}")

        model = train_model(history, active, settings)
        forecast = recursive_forecast(model, history, calendar, active)
        actuals = calendar.loc[:, ["sku", "date", "demand"]].rename(columns={"demand": "actual"})
        merged = forecast.merge(actuals, on=["sku", "date"], how="left", validate="one_to_one")

        baselines = _baseline_predictions(history, calendar, steps)
        for name, predictions in baselines.items():
            merged = merged.merge(
                predictions.loc[:, ["sku", "date", name]],
                on=["sku", "date"],
                how="left",
                validate="one_to_one",
            )
        if merged["actual"].isna().any():
            raise ValueError(f"fold {fold.index} is missing actuals")
        merged.insert(0, "fold", fold.index)
        observations.append(merged)

        model_metrics = _fold_metrics(merged, history, "mean", quantiles=True)
        baseline_metrics = {
            name: _fold_metrics(merged, history, name, quantiles=False)
            for name in ("naive", "seasonal_naive", "moving_average")
        }
        per_fold.append(
            {
                "index": fold.index,
                "origin": fold.train_end.strftime("%Y-%m-%d"),
                "test_start": fold.test_start.strftime("%Y-%m-%d"),
                "test_end": fold.test_end.strftime("%Y-%m-%d"),
                "train_rows": int(model.train_rows),
                "model": model_metrics,
                "baselines": baseline_metrics,
            }
        )

    observed = pd.concat(observations, ignore_index=True)
    weights = observed.groupby("fold")["actual"].sum()
    model_wape = float(
        sum(
            per_fold[index]["model"]["wape"] * weights.loc[per_fold[index]["index"]]
            for index in range(len(per_fold))
        )
        / weights.sum()
    )
    metrics: dict[str, Any] = {
        "model": {
            "wape": model_wape,
            "fold_wape": [fold["model"]["wape"] for fold in per_fold],
            "mase": float(np.mean([fold["model"]["mase"] for fold in per_fold])),
            "bias_abs_max_pct": float(
                max(
                    abs(value)
                    for fold in per_fold
                    for value in fold["model"]["bias_pct"].values()
                )
            ),
            "pinball": {
                quantile: float(
                    np.mean([fold["model"]["pinball"][quantile] for fold in per_fold])
                )
                for quantile in ("p10", "p50", "p90")
            },
        },
        "baselines": {
            name: {
                "wape": float(
                    sum(
                        fold["baselines"][name]["wape"] * weights.loc[fold["index"]]
                        for fold in per_fold
                    )
                    / weights.sum()
                ),
                "mase": float(
                    np.mean([fold["baselines"][name]["mase"] for fold in per_fold])
                ),
                "bias_abs_max_pct": float(
                    max(
                        abs(value)
                        for fold in per_fold
                        for value in fold["baselines"][name]["bias_pct"].values()
                    )
                ),
            }
            for name in ("naive", "seasonal_naive", "moving_average")
        },
        "per_fold": per_fold,
    }
    return BacktestResult(
        folds=folds,
        metrics=metrics,
        observations=observed,
        dataset_hash=dataset_hash(frame),
        feature_version_hash=active.version_hash,
        config=settings,
    )


def write_report(result: BacktestResult, path: Path | None = None) -> Path:
    target = path or REPORTS_DIR / "backtest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_report(), indent=2, sort_keys=True), encoding="utf-8")
    return target


def write_observations(result: BacktestResult, path: Path | None = None) -> Path:
    target = path or Path("models") / "backtest" / "observations.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    result.observations.to_parquet(target, index=False)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rolling-origin backtest")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from modelgate.data.ingest import read_demand

    try:
        frame = read_demand()
    except FileNotFoundError:
        frame = generate_demand()

    result = run_backtest(
        frame,
        TrainingConfig(seed=args.seed, horizon=args.horizon),
        n_folds=args.folds,
    )
    observations = write_observations(result)
    report = result.to_report()
    print(f"dataset hash {result.dataset_hash}")
    print(f"folds        {len(result.folds)} (h={args.horizon})")
    print(f"{'method':<16}{'WAPE':>8}{'MASE':>8}{'max|bias|%':>12}")
    print(
        f"{'model':<16}{report['metrics']['model']['wape']:>8.4f}"
        f"{report['metrics']['model']['mase']:>8.3f}"
        f"{report['metrics']['model']['bias_abs_max_pct']:>12.2f}"
    )
    for name, values in report["metrics"]["baselines"].items():
        print(f"{name:<16}{values['wape']:>8.4f}{values['mase']:>8.3f}"
              f"{values['bias_abs_max_pct']:>12.2f}")
    print(f"observations {observations}")
    if args.report:
        target = write_report(result)
        print(f"report       {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
