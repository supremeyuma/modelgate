from __future__ import annotations

import pandas as pd
import pytest

from modelgate.data.generator import generate_demand
from modelgate.features.build import (
    LeakageError,
    build_features,
    check_point_in_time,
    compute_feature_frame,
    compute_tail_features,
)
from modelgate.features.contract import (
    FeatureContract,
    FeatureSpec,
    contract_without,
    default_contract,
)


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    return generate_demand(n_skus=3, n_days=160, seed=5)


def test_as_of_join_never_uses_future_rows(panel: pd.DataFrame) -> None:
    """For every sampled row the truncated rebuild equals the full build."""
    report = check_point_in_time(panel, default_contract(), sample_size=20, seed=1)
    assert report.ok
    assert report.rows_checked == 20

    # An explicit truncation test on the last observed day of one SKU.
    sku = panel.loc[panel["sku"] == "SKU-000"].sort_values("date")
    cutoff = sku["date"].iloc[-1]
    full = compute_feature_frame(panel)
    truncated = compute_feature_frame(panel.loc[panel["date"] <= cutoff])
    here = full.loc[(full["sku"] == "SKU-000") & (full["date"] == cutoff)].iloc[0]
    there = truncated.loc[
        (truncated["sku"] == "SKU-000") & (truncated["date"] == cutoff)
    ].iloc[0]
    for name in default_contract().feature_names:
        assert here[name] == pytest.approx(there[name])

    # And the contract's declared rules never mention the target day or later.
    for spec in default_contract().specs:
        assert "t+" not in spec.pit_rule


def test_lag_features_match_manual_shift(panel: pd.DataFrame) -> None:
    matrix = build_features(panel, default_contract())
    sku = panel.loc[panel["sku"] == "SKU-002"].sort_values("date").reset_index(drop=True)
    built = matrix.loc[matrix["sku"] == "SKU-002"].sort_values("date").reset_index(drop=True)
    # The first feature row sits at index 28 (28 lags of warmup).
    assert built["date"].iloc[0] == sku["date"].iloc[28]
    for lag in (1, 7, 28):
        expected = sku["demand"].shift(lag).iloc[28:].to_numpy(dtype=float)
        assert built[f"lag_{lag}"].to_numpy() == pytest.approx(expected)
    expected_mean = sku["demand"].rolling(7).mean().shift(1).iloc[28:].to_numpy(dtype=float)
    assert built["rolling_mean_7"].to_numpy() == pytest.approx(expected_mean)


def test_feature_version_hash_changes_when_spec_changes() -> None:
    base = default_contract()
    trimmed = contract_without({"lag_28"})
    relabelled = FeatureContract(
        specs=tuple(
            FeatureSpec(**{**spec.to_dict(), "pit_rule": spec.pit_rule + " "})
            for spec in base.specs
        )
    )
    assert base.version_hash == default_contract().version_hash
    assert base.version_hash != trimmed.version_hash
    assert base.version_hash != relabelled.version_hash
    assert len(base.version_hash) == 64
    assert base.spec("lag_1").window == 1
    with pytest.raises(KeyError):
        base.spec("missing")
    assert base.to_dict()["version_hash"] == base.version_hash


def test_build_features_drops_warmup_and_has_no_nulls(panel: pd.DataFrame) -> None:
    matrix = build_features(panel, default_contract())
    assert len(matrix) == len(panel) - 28 * panel["sku"].nunique()
    assert not matrix[list(default_contract().feature_names)].isna().any().any()


def test_tail_features_match_full_matrix(panel: pd.DataFrame) -> None:
    full = compute_feature_frame(panel, default_contract())
    last = full.groupby("sku").tail(1).set_index("sku")
    tail = compute_tail_features(panel, default_contract()).set_index("sku")
    for name in default_contract().feature_names:
        assert tail[name].to_dict() == pytest.approx(last[name].to_dict())
    with pytest.raises(ValueError, match="not enough history"):
        compute_tail_features(panel.head(3), default_contract())


def test_leakage_guard_can_actually_fail(panel: pd.DataFrame) -> None:
    """A builder that peeks at tomorrow's demand must be caught by the truncated rebuild."""

    def leaking_builder(frame: pd.DataFrame, contract: FeatureContract) -> pd.DataFrame:
        leaked = compute_feature_frame(frame, contract)
        leaked["lag_1"] = leaked.groupby("sku")["demand"].shift(-1)
        return leaked

    with pytest.raises(LeakageError):
        check_point_in_time(panel, default_contract(), sample_size=3, builder=leaking_builder)
