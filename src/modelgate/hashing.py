"""Canonical hashing helpers used for datasets, artifacts and signed records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    """Stable JSON: sorted keys, no whitespace, UTF-8 friendly."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def dataframe_hash(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    """Hash the logical content of a frame, independent of row order and index."""
    use = list(columns) if columns is not None else list(frame.columns)
    ordered = frame.loc[:, use].sort_values(use, kind="mergesort").reset_index(drop=True)
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return sha256_bytes(payload)


def dataset_hash(frame: pd.DataFrame) -> str:
    """Hash the observed demand panel in a platform-independent way."""
    ordered = frame.sort_values(["sku", "date"], kind="mergesort").reset_index(drop=True)
    columns = ["sku", "date", "demand", "promo", "holiday"]
    normalised = ordered.loc[:, columns].copy()
    normalised["date"] = pd.to_datetime(normalised["date"]).dt.strftime("%Y-%m-%d")
    normalised["promo"] = normalised["promo"].astype(bool)
    normalised["holiday"] = normalised["holiday"].astype(bool)
    return sha256_bytes(normalised.to_csv(index=False, lineterminator="\n").encode("utf-8"))
