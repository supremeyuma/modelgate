"""Repository paths and global constants."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MODELGATE_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "modelgate.duckdb"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
MLRUNS_DIR = ROOT / "mlruns"

SEED = 42
N_SKUS = 12
N_DAYS = 900
START_DATE = "2022-01-01"

#: The only columns a model may treat as observations. `family` is static metadata.
OBSERVED_COLUMNS = ("sku", "date", "demand", "promo", "holiday")
