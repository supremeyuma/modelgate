"""Build the point-in-time feature matrix and prove that no future row leaks in.

Every lag and rolling statistic is computed from ``demand`` strictly before the target date.
The empirical guard (`check_point_in_time`) recomputes a sample of rows from a truncated history
and compares them with the full-matrix values. A single future reference makes the two differ and
fails the check.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from modelgate.config import DB_PATH
from modelgate.features.contract import FeatureContract, FeatureSpec, default_contract

WARMUP_COLUMNS = ("demand", "date", "sku", "promo", "holiday")


class LeakageError(RuntimeError):
    """Raised when a feature observes the future."""


@dataclass(frozen=True)
class PointInTimeReport:
    rows_checked: int
    features_checked: int
    max_abs_difference: float

    @property
    def ok(self) -> bool:
        return self.max_abs_difference == 0.0


def _require_history(frame: pd.DataFrame) -> None:
    missing = [column for column in WARMUP_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"feature build requires columns {missing}")


def _compute_one(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    grouped = frame.groupby("sku", sort=False)
    if spec.kind == "lag":
        assert spec.window is not None
        return grouped["demand"].shift(spec.window).astype("float64")
    if spec.kind == "rolling_mean":
        assert spec.window is not None
        window = spec.window
        return (
            grouped["demand"]
            .transform(
                lambda values: values.shift(1).rolling(window, min_periods=window).mean()
            )
            .astype("float64")
        )
    if spec.kind == "rolling_std":
        assert spec.window is not None
        window = spec.window
        return (
            grouped["demand"]
            .transform(
                lambda values: values.shift(1).rolling(window, min_periods=window).std()
            )
            .astype("float64")
        )
    if spec.kind == "calendar" and spec.name == "day_of_week":
        return cast(pd.Series, frame["date"].dt.day_of_week.astype("int8"))
    if spec.kind == "flag":
        return frame[spec.source].astype("int8")
    raise ValueError(f"unsupported feature spec: {spec.name}")


def compute_feature_frame(
    frame: pd.DataFrame, contract: FeatureContract | None = None
) -> pd.DataFrame:
    """Return every row (including warmup rows) with feature columns appended."""
    active = contract or default_contract()
    _require_history(frame)
    ordered = frame.sort_values(["sku", "date"], kind="mergesort").reset_index(drop=True)
    result = ordered.loc[:, ["sku", "date", "family", "demand", "promo", "holiday"]].copy()
    for spec in active.specs:
        result[spec.name] = _compute_one(ordered, spec)
    return result


def build_features(
    frame: pd.DataFrame, contract: FeatureContract | None = None
) -> pd.DataFrame:
    """Feature matrix with warmup and unlabelled rows removed and nulls rejected."""
    active = contract or default_contract()
    computed = compute_feature_frame(frame, active)
    names = list(active.feature_names)
    valid = computed.dropna(subset=names)
    if "demand" in valid.columns:
        valid = valid.loc[valid["demand"].notna()]
    valid = valid.loc[:, ["sku", "date", "family", *names, active.target]].reset_index(drop=True)
    nulls = valid[names].isna().sum()
    # dropna above makes this unreachable; the guard stays in case the builder changes.
    if int(nulls.sum()) > 0:  # pragma: no cover
        broken = {key: int(value) for key, value in nulls.items() if value}
        raise ValueError(f"null feature values after warmup: {broken}")
    return valid


def check_point_in_time(
    frame: pd.DataFrame,
    contract: FeatureContract | None = None,
    sample_size: int = 24,
    seed: int = 0,
    builder: Callable[[pd.DataFrame, FeatureContract], pd.DataFrame] | None = None,
) -> PointInTimeReport:
    """Recompute sampled rows from history truncated before the target date.

    The full matrix is allowed to use every row; the truncated rebuild is not. Any difference
    between the two is, by construction, future information. `builder` is injectable so the guard
    itself can be tested with a deliberately leaking implementation.
    """
    active = contract or default_contract()
    build = builder or compute_feature_frame
    full = build(frame, active)
    names = list(active.feature_names)
    candidates = full.dropna(subset=names)
    candidates = candidates.loc[candidates.groupby("sku").cumcount() >= 1]
    if candidates.empty:
        raise ValueError("no rows with usable history to check")

    rng = np.random.default_rng(seed)
    take = min(sample_size, len(candidates))
    positions = rng.choice(len(candidates), size=take, replace=False)
    sample = candidates.iloc[np.sort(positions)]

    max_difference = 0.0
    for _, target in sample.iterrows():
        history = frame.loc[
            (frame["sku"] == target["sku"]) & (frame["date"] < target["date"])
        ].copy()
        if history.empty:
            continue
        placeholder = pd.DataFrame(
            [
                {
                    "sku": target["sku"],
                    "date": target["date"],
                    "family": target["family"],
                    "demand": np.nan,
                    "promo": bool(target["promo"]),
                    "holiday": bool(target["holiday"]),
                }
            ]
        )
        truncated = pd.concat([history, placeholder], ignore_index=True)
        rebuilt = build(truncated, active).iloc[-1]
        for name in names:
            left = float(target[name])
            right = float(rebuilt[name])
            difference = abs(left - right)
            if np.isnan(right):
                difference = float("inf")
            max_difference = max(max_difference, difference)

    report = PointInTimeReport(
        rows_checked=len(sample), features_checked=len(names), max_abs_difference=max_difference
    )
    if not report.ok:
        raise LeakageError(
            f"features differ when rebuilt from truncated history: {report.max_abs_difference}"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build features or prove point-in-time safety")
    parser.add_argument("--check-point-in-time", action="store_true")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--sample-size", type=int, default=24)
    args = parser.parse_args(argv)

    from modelgate.data.ingest import read_demand

    frame = read_demand(args.db)
    contract = default_contract()
    if args.check_point_in_time:
        report = check_point_in_time(frame, contract, sample_size=args.sample_size)
        print(
            f"point-in-time OK: {report.rows_checked} rows x {report.features_checked} features, "
            f"max |difference| = {report.max_abs_difference}"
        )
        print(f"feature version {contract.version_hash}")
        return 0
    matrix = build_features(frame, contract)
    print(f"built {len(matrix):,} feature rows with {len(contract.feature_names)} features")
    print(f"feature version {contract.version_hash}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
