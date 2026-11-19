"""Load generated demand into DuckDB and date-partitioned parquet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from modelgate.config import DB_PATH, RAW_DIR
from modelgate.hashing import dataset_hash


@dataclass(frozen=True)
class IngestResult:
    rows: int
    skus: int
    first_date: str
    last_date: str
    dataset_hash: str
    db_path: Path
    partition_dir: Path


def load_raw(path: Path | None = None) -> pd.DataFrame:
    source = path or RAW_DIR / "demand.parquet"
    if not source.exists():
        raise FileNotFoundError(f"raw demand not found at {source}; run `make data`")
    frame = pd.read_parquet(source)
    frame["date"] = pd.to_datetime(frame["date"]).dt.as_unit("ns")
    return frame


def read_demand(db_path: Path | str | None = None) -> pd.DataFrame:
    target = Path(db_path) if db_path is not None else DB_PATH
    if not target.exists():
        raise FileNotFoundError(f"database not found at {target}; run `make data`")
    with duckdb.connect(str(target), read_only=True) as con:
        frame = con.execute("SELECT * FROM demand ORDER BY sku, date").df()
    frame["date"] = pd.to_datetime(frame["date"]).dt.as_unit("ns")
    return frame


def ingest(
    raw_path: Path | None = None,
    db_path: Path | None = None,
    partition_dir: Path | None = None,
) -> IngestResult:
    """Persist the raw panel twice: a queryable DuckDB table and Hive-partitioned parquet."""
    frame = load_raw(raw_path)
    target_db = db_path or DB_PATH
    partitions = partition_dir or RAW_DIR / "partitioned"
    target_db.parent.mkdir(parents=True, exist_ok=True)
    partitions.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(target_db)) as con:
        con.execute(
            "CREATE OR REPLACE TABLE demand AS SELECT * FROM read_parquet(?)",
            [str(raw_path or RAW_DIR / "demand.parquet")],
        )
        con.execute(
            "COPY demand TO ? (FORMAT PARQUET, PARTITION_BY (date), OVERWRITE_OR_IGNORE 1)",
            [str(partitions)],
        )

    return IngestResult(
        rows=len(frame),
        skus=int(frame["sku"].nunique()),
        first_date=frame["date"].min().strftime("%Y-%m-%d"),
        last_date=frame["date"].max().strftime("%Y-%m-%d"),
        dataset_hash=dataset_hash(frame),
        db_path=target_db,
        partition_dir=partitions,
    )


def main() -> int:
    result = ingest()
    print(
        f"ingested {result.rows:,} rows / {result.skus} SKUs "
        f"({result.first_date}..{result.last_date})"
    )
    print(f"dataset hash {result.dataset_hash}")
    print(f"duckdb        {result.db_path}")
    print(f"partitions    {result.partition_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
