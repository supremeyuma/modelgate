"""Synthetic daily demand with a documented data generating process (DGP).

The DGP is deliberately *not* random noise around a constant. Every downstream claim depends on
these properties, so they are part of the contract:

* **Weekly seasonality** - Friday/Saturday/Sunday trade above Monday-Thursday. A lag-7
  autocorrelation of roughly 0.5 or more is expected on every SKU.
* **Local trend** - each SKU has a small positive or negative daily drift.
* **Annual seasonality** - a sinusoid peaking in early summer.
* **Promotions** - short bursts (1-3 days) with a multiplicative lift; promo days are recorded so
  features can use them, and they add heavy tails that a naive model must handle.
* **Holidays** - a fixed calendar of retail holidays with larger lifts.
* **Regime shift** - some SKUs change level at day 600 of the sample. This is what a monitor is
  supposed to notice, and what makes a single train/test split dishonest.
* **Overdispersed count noise** - lognormal multiplicative noise followed by a Poisson draw, so
  demand is integer, non-negative and noisier than Poisson.

Output schema (one row per SKU and calendar day):
``sku: str, family: str, date: datetime64[ns], demand: int64, promo: bool, holiday: bool``

Everything is reproducible from ``seed``: no wall clock, no global RNG state.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from modelgate.config import (
    N_DAYS,
    N_SKUS,
    RAW_DIR,
    SEED,
    START_DATE,
)

WEEKLY_FACTORS = np.array([0.76, 0.82, 0.90, 1.00, 1.26, 1.50, 1.14])
FAMILIES = ("beverage", "pantry", "household")
#: day-of-year (1-based) -> multiplicative holiday lift
HOLIDAYS: dict[int, float] = {
    1: 1.45,    # New Year
    15: 1.35,   # MLK weekend (approximate, fixed calendar)
    185: 1.55,  # Independence Day
    330: 1.35,  # Thanksgiving week
    335: 1.90,  # Black Friday
    359: 1.60,  # Christmas Eve
    360: 1.30,  # Christmas
}


def _seasonal_profile(days: np.ndarray) -> np.ndarray:
    """Annual sinusoid peaking around day 170."""
    return 1.0 + 0.13 * np.sin(2 * np.pi * (days - 80) / 365.25)


def generate_demand(
    n_skus: int = N_SKUS,
    n_days: int = N_DAYS,
    seed: int = SEED,
    start: str | date = START_DATE,
) -> pd.DataFrame:
    """Generate the synthetic panel described in the module docstring."""
    if n_skus < 1 or n_days < 1:
        raise ValueError("n_skus and n_days must be positive")
    rng = np.random.default_rng(seed)
    start_date = date.fromisoformat(start) if isinstance(start, str) else start
    dates = np.array([start_date + timedelta(days=i) for i in range(n_days)])
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    dow = np.array([d.weekday() for d in dates])
    frame_dates = pd.to_datetime(pd.Series(dates), format="%Y-%m-%d")

    holiday_lift = np.ones(n_days)
    for doy, lift in HOLIDAYS.items():
        holiday_lift[day_of_year == doy] = lift
    holiday_flag = holiday_lift > 1.0

    rows: list[pd.DataFrame] = []
    for sku_index in range(n_skus):
        sku_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        base = float(np.exp(sku_rng.normal(3.4, 0.35)))
        slope = float(sku_rng.normal(0.0006, 0.0005))
        weekly = WEEKLY_FACTORS * sku_rng.normal(1.0, 0.035, size=7)
        weekly = weekly / weekly.mean()

        promo = np.zeros(n_days, dtype=bool)
        promo_lift = np.ones(n_days)
        index = 0
        while index < n_days:
            gap = int(sku_rng.geometric(0.055)) - 1
            index += gap
            run = int(sku_rng.integers(1, 4))
            if index >= n_days:
                break
            lift = float(np.exp(sku_rng.normal(0.55, 0.18)))
            promo[index : index + run] = True
            promo_lift[index : index + run] = lift
            index += run

        regime = np.ones(n_days)
        if sku_index % 4 == 1:
            regime[600:] = 0.82
        elif sku_index % 4 == 3:
            regime[600:] = 1.18

        level = (
            base
            * (1.0 + slope * np.arange(n_days))
            * weekly[dow]
            * _seasonal_profile(day_of_year)
            * promo_lift
            * holiday_lift
            * regime
        )
        multiplicative_noise = sku_rng.lognormal(mean=0.0, sigma=0.07, size=n_days)
        mu = np.maximum(level * multiplicative_noise, 0.05)
        # Poisson draw at this mean keeps the series integer. The lognormal factor already
        # provides overdispersion relative to a pure Poisson process.
        demand = sku_rng.poisson(mu).astype(np.int64)

        rows.append(
            pd.DataFrame(
                {
                    "sku": f"SKU-{sku_index:03d}",
                    "family": FAMILIES[sku_index % len(FAMILIES)],
                    "date": frame_dates,
                    "demand": demand,
                    "promo": promo,
                    "holiday": holiday_flag.copy(),
                }
            )
        )

    frame = pd.concat(rows, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["sku", "date"], ignore_index=True)


def write_raw(frame: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the generated panel as a single parquet file (gitignored)."""
    target = path or RAW_DIR / "demand.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    return target


def main() -> int:
    frame = generate_demand()
    path = write_raw(frame)
    print(f"generated {len(frame):,} rows / {frame['sku'].nunique()} SKUs -> {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
