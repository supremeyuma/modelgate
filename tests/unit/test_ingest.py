from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from modelgate.data.generator import generate_demand, write_raw
from modelgate.data.ingest import ingest, load_raw, read_demand
from modelgate.hashing import canonical_json, dataset_hash, sha256_file, sha256_json


def test_ingest_round_trips_through_duckdb_and_partitions(tmp_path: Path) -> None:
    raw = write_raw(generate_demand(n_skus=2, n_days=10, seed=4), tmp_path / "demand.parquet")
    result = ingest(raw, tmp_path / "test.duckdb", tmp_path / "partitioned")
    assert result.rows == 20
    assert result.skus == 2
    assert result.first_date == "2022-01-01"
    assert result.dataset_hash == dataset_hash(generate_demand(n_skus=2, n_days=10, seed=4))
    assert any((tmp_path / "partitioned").glob("date=*"))

    restored = read_demand(tmp_path / "test.duckdb")
    pd.testing.assert_frame_equal(
        restored.sort_values(["sku", "date"]).reset_index(drop=True),
        load_raw(raw).sort_values(["sku", "date"]).reset_index(drop=True),
    )


def test_missing_inputs_are_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="raw demand"):
        load_raw(tmp_path / "absent.parquet")
    with pytest.raises(FileNotFoundError, match="database"):
        read_demand(tmp_path / "absent.duckdb")


def test_hash_helpers_are_stable_and_order_independent() -> None:
    frame = generate_demand(n_skus=2, n_days=20, seed=9)
    shuffled = frame.sample(frac=1.0, random_state=0)
    assert dataset_hash(frame) == dataset_hash(shuffled)
    other = generate_demand(n_skus=2, n_days=20, seed=10)
    assert dataset_hash(frame) != dataset_hash(other)
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    path = Path(__file__).resolve()
    assert len(sha256_file(path)) == 64
