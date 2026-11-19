"""Validation gates protecting the raw panel and the feature matrix."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = ("sku", "date", "demand", "promo", "holiday")
MAX_NULL_RATE = 0.0
MAX_ROW_DROP = 0.25
MAX_MEAN_RATIO = 2.5


class ValidationError(ValueError):
    """Raised when a dataset violates a hard gate."""


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    skus: int
    null_rates: Mapping[str, float] = field(default_factory=dict)
    failures: tuple[str, ...] = ()


def validate(
    frame: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    reference_mean: float | None = None,
    max_row_drop: float = MAX_ROW_DROP,
    max_null_rate: float = MAX_NULL_RATE,
    max_mean_ratio: float = MAX_MEAN_RATIO,
) -> ValidationReport:
    """Run every hard gate and raise on the first class of failure."""
    failures: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValidationError(f"missing required columns: {missing}")
    if frame.empty:
        raise ValidationError("dataset is empty")

    null_rates = {column: float(frame[column].isna().mean()) for column in REQUIRED_COLUMNS}
    bad_nulls = {column: rate for column, rate in null_rates.items() if rate > max_null_rate}
    if bad_nulls:
        failures.append(f"null rate above {max_null_rate}: {bad_nulls}")

    duplicates = int(frame.duplicated(subset=["sku", "date"]).sum())
    if duplicates:
        raise ValidationError(f"duplicate (sku, date) rows: {duplicates}")

    if not pd.api.types.is_integer_dtype(frame["demand"]):
        failures.append(f"demand must be integer, got {frame['demand'].dtype}")
    negative = int((frame["demand"] < 0).sum())
    if negative:
        failures.append(f"negative demand rows: {negative}")

    if expected_rows is not None and expected_rows > 0:
        drop = 1.0 - len(frame) / expected_rows
        if drop > max_row_drop:
            failures.append(
                f"row-count drop {drop:.1%} exceeds {max_row_drop:.0%} "
                f"({len(frame)} of {expected_rows})"
            )

    if reference_mean is not None and reference_mean > 0:
        ratio = float(frame["demand"].mean()) / reference_mean
        if ratio > max_mean_ratio or ratio < 1.0 / max_mean_ratio:
            failures.append(f"mean demand ratio {ratio:.2f} outside [{1 / max_mean_ratio:.2f}, "
                            f"{max_mean_ratio:.2f}]")

    report = ValidationReport(
        rows=len(frame),
        skus=int(frame["sku"].nunique()),
        null_rates=null_rates,
        failures=tuple(failures),
    )
    if failures:
        raise ValidationError("; ".join(failures))
    return report
