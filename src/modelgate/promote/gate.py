"""The promotion gate: every condition from PLAN.md section 6, plus a signed record.

A decision is never silent: PROMOTE, REJECT or INCONCLUSIVE always carries the metric bundle,
the failed checks and content hashes of both artifacts. The record's SHA-256 is a deterministic
function of its content, so a third party with the same inputs can reproduce it byte for byte.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from modelgate.config import REPORTS_DIR
from modelgate.eval.metrics import bias_by_horizon, wape
from modelgate.hashing import sha256_json
from modelgate.promote.stats import (
    bootstrap_wape_delta,
    diebold_mariano,
    tail_degradation,
)

ArrayLike = Sequence[float] | np.ndarray
SIGNATURE_PREFIX = "modelgate-promotion:"


class Decision(StrEnum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class GateThresholds:
    dm_p_value: float = 0.05
    ci_alpha: float = 0.05
    max_tail_degradation_pct: float = 5.0
    max_bias_pct: float = 2.0
    latency_budget_ms: float = 150.0
    min_samples: int = 30


@dataclass
class GateEvaluation:
    decision: Decision
    reasons: tuple[str, ...]
    checks: dict[str, bool]
    stats: dict[str, Any]
    record: dict[str, Any]
    record_sha256: str = field(default="")

    @property
    def promoted(self) -> bool:
        return self.decision is Decision.PROMOTE


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def evaluate_gate(
    actual: ArrayLike,
    champion: ArrayLike,
    challenger: ArrayLike,
    horizon: ArrayLike,
    *,
    champion_version: str,
    challenger_version: str,
    champion_artifact_sha256: str,
    challenger_artifact_sha256: str,
    feature_version_hash: str,
    dataset_hash: str,
    git_sha: str,
    latency_ms: float | None,
    thresholds: GateThresholds | None = None,
    decided_at: str | None = None,
) -> GateEvaluation:
    limits = thresholds or GateThresholds()
    y = np.asarray(actual, dtype=float).ravel()
    c = np.asarray(champion, dtype=float).ravel()
    h = np.asarray(challenger, dtype=float).ravel()
    steps = np.asarray(horizon, dtype=int).ravel()
    if not (y.size == c.size == h.size == steps.size):
        raise ValueError("actual, champion, challenger and horizon must align")
    if y.size == 0:
        raise ValueError("no observations to gate")
    # Canonical order makes every statistic, including the bootstrap, permutation invariant.
    order = np.lexsort((steps, h, c, y))
    y, c, h, steps = y[order], c[order], h[order], steps[order]

    wape_champion = wape(y, c)
    wape_challenger = wape(y, h)
    delta = wape_challenger - wape_champion
    loss_champion = np.abs(y - c)
    loss_challenger = np.abs(y - h)

    if y.size >= limits.min_samples:
        dm = diebold_mariano(loss_champion, loss_challenger, min_samples=limits.min_samples)
        dm_statistic, dm_p = dm.statistic, dm.p_value
    else:
        dm_statistic, dm_p = float("nan"), 1.0
    _, ci_low, ci_high = bootstrap_wape_delta(y, c, h, alpha=limits.ci_alpha)
    tail = tail_degradation(y, c, h)
    bias = bias_by_horizon(y, h, steps)
    max_bias = float(max(abs(value) for value in bias.values()))

    checks = {
        "mean_wape_improves": wape_challenger < wape_champion,
        "dm_significant": bool(dm_statistic > 0 and dm_p < limits.dm_p_value),
        "ci_excludes_zero": ci_high < 0.0,
        "tail_risk_bounded": tail <= limits.max_tail_degradation_pct,
        "bias_within_bound": max_bias <= limits.max_bias_pct,
        "latency_within_budget": latency_ms is not None
        and latency_ms <= limits.latency_budget_ms,
    }
    labels = {
        "mean_wape_improves": "mean WAPE did not improve",
        "dm_significant": f"Diebold-Mariano p={dm_p:.4f} >= {limits.dm_p_value}",
        "ci_excludes_zero": f"bootstrap 95% CI [{ci_low:.4f}, {ci_high:.4f}] contains 0",
        "tail_risk_bounded": (
            f"tail risk: worst-decile WAPE degraded {tail:.2f}% "
            f"> {limits.max_tail_degradation_pct}%"
        ),
        "bias_within_bound": (
            f"max |bias| {max_bias:.2f}% > {limits.max_bias_pct}% at a horizon"
        ),
        "latency_within_budget": (
            f"latency {latency_ms} ms not within {limits.latency_budget_ms} ms budget"
        ),
    }

    if y.size < limits.min_samples:
        decision = Decision.INCONCLUSIVE
        reasons: list[str] = [
            f"only {y.size} paired observations; need {limits.min_samples} for a decision"
        ]
    elif all(checks.values()):
        decision = Decision.PROMOTE
        reasons = ["all promotion conditions satisfied"]
    else:
        decision = Decision.REJECT
        reasons = [labels[name] for name, passed in checks.items() if not passed]

    stats: dict[str, Any] = {
        "champion_wape": wape_champion,
        "challenger_wape": wape_challenger,
        "delta_wape": delta,
        "dm_statistic": dm_statistic,
        "dm_p_value": dm_p,
        "bootstrap_ci": [ci_low, ci_high],
        "tail_degradation_pct": tail,
        "max_abs_bias_pct": max_bias,
        "bias_pct_by_horizon": {str(key): value for key, value in bias.items()},
        "latency_ms": latency_ms,
        "n_observations": int(y.size),
    }
    record: dict[str, Any] = {
        "schema": "modelgate.promotion/1",
        "decided_at": decided_at or _utc_now(),
        "decision": decision.value,
        "reasons": reasons,
        "checks": checks,
        "champion": {
            "version": champion_version,
            "artifact_sha256": champion_artifact_sha256,
        },
        "challenger": {
            "version": challenger_version,
            "artifact_sha256": challenger_artifact_sha256,
        },
        "feature_version_hash": feature_version_hash,
        "dataset_hash": dataset_hash,
        "git_sha": git_sha,
        "thresholds": asdict(limits),
        "metrics": stats,
    }
    digest = sha256_json(record)
    record["record_sha256"] = digest
    record["signature"] = sha256_json({"prefix": SIGNATURE_PREFIX, "record_sha256": digest})
    return GateEvaluation(
        decision=decision,
        reasons=tuple(reasons),
        checks=checks,
        stats=stats,
        record=record,
        record_sha256=digest,
    )


def verify_record(record: dict[str, Any]) -> bool:
    """Check that a record was not edited after it was signed."""
    claimed = record.get("record_sha256")
    signature = record.get("signature")
    if not claimed or not signature:
        return False
    body = {
        key: value for key, value in record.items() if key not in {"record_sha256", "signature"}
    }
    if sha256_json(body) != claimed:
        return False
    return bool(
        sha256_json({"prefix": SIGNATURE_PREFIX, "record_sha256": claimed}) == signature
    )


def record_filename(record: dict[str, Any]) -> str:
    stamp = str(record["decided_at"]).replace(":", "").replace("-", "").replace("+", "_")
    return f"promotion-{stamp}-{record['record_sha256'][:8]}.json"


def write_record(record: dict[str, Any], directory: Path | None = None) -> Path:
    target_dir = directory or REPORTS_DIR / "promotions"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / record_filename(record)
    target.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return target
