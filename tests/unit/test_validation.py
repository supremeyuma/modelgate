from __future__ import annotations

import pandas as pd
import pytest

from modelgate.data.generator import generate_demand
from modelgate.validation.checks import ValidationError, validate


@pytest.fixture()
def frame() -> pd.DataFrame:
    return generate_demand(n_skus=2, n_days=60, seed=2)


def test_validation_accepts_clean_panel(frame: pd.DataFrame) -> None:
    report = validate(frame, expected_rows=len(frame))
    assert report.rows == len(frame)
    assert report.skus == 2
    assert report.failures == ()


def test_validation_rejects_duplicates_and_nulls(frame: pd.DataFrame) -> None:
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValidationError, match="duplicate"):
        validate(duplicated)

    with_nulls = frame.copy()
    with_nulls.loc[0, "demand"] = None
    with pytest.raises(ValidationError, match="null rate"):
        validate(with_nulls)


def test_validation_catches_injected_row_count_drop(frame: pd.DataFrame) -> None:
    kept = frame.sample(frac=0.70, random_state=1)
    with pytest.raises(ValidationError, match="row-count drop"):
        validate(kept, expected_rows=len(frame))
    # A 10% drop is tolerated.
    smaller = frame.sample(frac=0.90, random_state=1)
    report = validate(smaller, expected_rows=len(frame))
    assert report.rows == len(smaller)


def test_validation_rejects_negative_demand_and_mean_shift(frame: pd.DataFrame) -> None:
    negative = frame.copy()
    negative.loc[1, "demand"] = -3
    with pytest.raises(ValidationError, match="negative demand"):
        validate(negative)

    with pytest.raises(ValidationError, match="mean demand ratio"):
        validate(frame, reference_mean=float(frame["demand"].mean()) * 10)


def test_validation_requires_columns_and_rows(frame: pd.DataFrame) -> None:
    with pytest.raises(ValidationError, match="missing required columns"):
        validate(frame.drop(columns=["holiday"]))
    with pytest.raises(ValidationError, match="empty"):
        validate(frame.iloc[0:0])
