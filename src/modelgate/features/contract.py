"""Declarative feature contract: what a feature is, and how it is allowed to be computed.

The contract is the single source of truth for the feature matrix. Its hash is logged with every
model, recorded in every promotion record and exposed by the serving API. If the specification
changes, the hash changes and old artifacts cannot silently be paired with new features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from modelgate.hashing import sha256_json

FeatureKind = Literal["lag", "rolling_mean", "rolling_std", "calendar", "flag"]

LAGS = tuple(range(1, 29))
ROLLING_WINDOWS = (7, 14, 28)


@dataclass(frozen=True)
class FeatureSpec:
    """One column of the feature matrix."""

    name: str
    kind: FeatureKind
    source: str
    pit_rule: str
    dtype: str
    window: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "pit_rule": self.pit_rule,
            "dtype": self.dtype,
            "window": self.window,
        }


@dataclass(frozen=True)
class FeatureContract:
    specs: tuple[FeatureSpec, ...]
    target: str = "demand"
    version: str = "1.0.0"
    calendar: tuple[str, ...] = field(default=("day_of_week", "promo", "holiday"))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def version_hash(self) -> str:
        return sha256_json(
            {
                "version": self.version,
                "target": self.target,
                "calendar": list(self.calendar),
                "specs": [spec.to_dict() for spec in self.specs],
            }
        )

    def spec(self, name: str) -> FeatureSpec:
        for candidate in self.specs:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "target": self.target,
            "calendar": list(self.calendar),
            "version_hash": self.version_hash,
            "specs": [spec.to_dict() for spec in self.specs],
        }


def default_contract() -> FeatureContract:
    """Lags t-1..t-28, rolling moments, and calendar flags. No future information."""
    specs: list[FeatureSpec] = []
    for lag in LAGS:
        specs.append(
            FeatureSpec(
                name=f"lag_{lag}",
                kind="lag",
                source="demand",
                pit_rule=f"demand[t-{lag}]",
                dtype="float64",
                window=lag,
            )
        )
    for window in ROLLING_WINDOWS:
        specs.append(
            FeatureSpec(
                name=f"rolling_mean_{window}",
                kind="rolling_mean",
                source="demand",
                pit_rule=f"mean(demand[t-{window}..t-1])",
                dtype="float64",
                window=window,
            )
        )
    for window in ROLLING_WINDOWS:
        specs.append(
            FeatureSpec(
                name=f"rolling_std_{window}",
                kind="rolling_std",
                source="demand",
                pit_rule=f"std(demand[t-{window}..t-1])",
                dtype="float64",
                window=window,
            )
        )
    specs.extend(
        [
            FeatureSpec(
                name="day_of_week",
                kind="calendar",
                source="date",
                pit_rule="calendar(date[t])",
                dtype="int8",
            ),
            FeatureSpec(
                name="promo",
                kind="flag",
                source="promo",
                pit_rule="promo[date[t]]",
                dtype="int8",
            ),
            FeatureSpec(
                name="holiday",
                kind="flag",
                source="holiday",
                pit_rule="holiday[date[t]]",
                dtype="int8",
            ),
        ]
    )
    return FeatureContract(specs=tuple(specs))


def contract_without(names: set[str]) -> FeatureContract:
    """Derive a variant contract (used to prove hashes are sensitive to the spec)."""
    base = default_contract()
    return FeatureContract(
        specs=tuple(spec for spec in base.specs if spec.name not in names),
        target=base.target,
        version=base.version,
        calendar=base.calendar,
    )
