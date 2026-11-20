"""Injected-drift demo: PSI table, alert, performance decay, dry-run retrain PR.

Everything is deterministic. The current window is the last 28 days with demand scaled by a fixed
factor, so the drift signal is known in advance and the alert threshold is not tuned to the demo.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pandas as pd

from modelgate.config import REPORTS_DIR
from modelgate.data.generator import generate_demand
from modelgate.data.ingest import read_demand
from modelgate.features.build import build_features
from modelgate.features.contract import default_contract
from modelgate.monitor import metrics as prometheus
from modelgate.monitor.drift import (
    DriftThresholds,
    detect_feature_drift,
    detect_performance_decay,
    ks_residuals,
    rolling_wape,
)
from modelgate.promote.demo import build_scenario
from modelgate.promote.gate import evaluate_gate
from modelgate.retrain.schedule import build_pr_body, retrain_and_open_pr

DRIFT_FEATURES = ("lag_1", "lag_7", "rolling_mean_7", "rolling_mean_28", "rolling_std_7")
SHIFT = 1.40


class DryRunGit:
    """Records the would-be git/gh actions instead of executing them."""

    def __init__(self) -> None:
        self.branch: str | None = None
        self.message: str | None = None
        self.files: list[str] = []

    def create_branch(self, name: str) -> None:
        self.branch = name
        print(f"dry-run: git switch -c {name}")

    def commit(self, files: Sequence[str], message: str) -> None:
        self.files = list(files)
        self.message = message
        print(f"dry-run: git commit {len(self.files)} file(s): {message}")

    def open_pr(self, title: str, body: str) -> str:
        print(f"dry-run: gh pr create --title {title!r}")
        return "dry-run://pr/1"


def _load() -> pd.DataFrame:
    try:
        return read_demand()
    except FileNotFoundError:
        return generate_demand()


def main() -> int:
    frame = _load().sort_values(["sku", "date"]).reset_index(drop=True)
    dates = pd.Series(pd.to_datetime(frame["date"]).unique()).sort_values().reset_index(drop=True)
    split = pd.Timestamp(dates.iloc[-28])
    shifted = frame.copy()
    shifted.loc[shifted["date"] >= split, "demand"] = (
        shifted.loc[shifted["date"] >= split, "demand"] * SHIFT
    ).round()
    reference = build_features(frame)
    current = build_features(shifted)
    reference = reference.loc[reference["date"] < split]
    current = current.loc[current["date"] >= split]

    thresholds = DriftThresholds()
    report = detect_feature_drift(reference, current, DRIFT_FEATURES, thresholds)
    print(f"{'feature':<20}{'PSI':>8}{'threshold':>11}  alert")
    for feature, score in report.psi.items():
        flag = "YES" if score > thresholds.psi else "no"
        print(f"{feature:<20}{score:>8.3f}{thresholds.psi:>11.2f}  {flag}")

    ref_residual = reference["demand"] - reference["lag_1"]
    cur_residual = current["demand"] - current["lag_1"]
    ks_statistic, ks_p = ks_residuals(ref_residual, cur_residual)
    print(f"KS residuals       statistic {ks_statistic:.3f}  p={ks_p:.3e}")
    if ks_p < thresholds.ks_p_value:
        report.alerts.append(f"residual KS p={ks_p:.3e} < {thresholds.ks_p_value}")

    decay = detect_performance_decay(
        expected_wape=0.20,
        current=rolling_wape(current["demand"], current["lag_1"]),
        thresholds=thresholds,
    )
    report.alerts.extend(decay.alerts)
    print(f"rolling WAPE       {decay.rolling_wape:.4f} (expected 0.20)")

    for feature, score in report.psi.items():
        prometheus.set_feature_psi(feature, score)
        if score > thresholds.psi:
            prometheus.record_drift_alert(feature)
    if decay.rolling_wape is not None:
        prometheus.set_rolling_wape(decay.rolling_wape)

    print(f"drift alerts       {len(report.alerts)}")
    for alert in report.alerts:
        print(f"  - {alert}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "drift.json").write_text(
        json.dumps(
            {
                "shift": SHIFT,
                "psi": report.psi,
                "alerts": report.alerts,
                "detected": report.drifted,
                "window_days": 28,
                "rolling_wape": decay.rolling_wape,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if not report.drifted:
        print("no drift detected; nothing to do")
        return 0

    scenario = build_scenario("promote")
    evaluation = evaluate_gate(
        scenario.actual,
        scenario.champion,
        scenario.challenger,
        scenario.horizon,
        champion_version="champion",
        challenger_version="retrain-candidate",
        champion_artifact_sha256="a" * 64,
        challenger_artifact_sha256="b" * 64,
        feature_version_hash=default_contract().version_hash,
        dataset_hash="demo",
        git_sha="demo",
        latency_ms=10.0,
    )
    git = DryRunGit()
    outcome = retrain_and_open_pr(
        evaluation, report, git, "champion", "retrain-candidate"
    )
    print(f"retrain            {outcome.reason}")
    if outcome.opened:
        print("--- pull request body ---")
        print(build_pr_body(evaluation, report, "champion", "retrain-candidate"))
        print("-------------------------")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
