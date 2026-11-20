from __future__ import annotations

from collections.abc import Sequence

from modelgate.monitor.drift import DriftReport
from modelgate.monitor.metrics import (
    metrics_payload,
    record_drift_alert,
    record_forecast,
    set_feature_psi,
    set_rolling_wape,
)
from modelgate.promote.demo import build_scenario
from modelgate.promote.gate import Decision, evaluate_gate
from modelgate.retrain.schedule import build_pr_body, retrain_and_open_pr


class RecordingGit:
    def __init__(self) -> None:
        self.branch: str | None = None
        self.files: list[str] = []
        self.pr: tuple[str, str] | None = None

    def create_branch(self, name: str) -> None:
        self.branch = name

    def commit(self, files: Sequence[str], message: str) -> None:
        self.files = list(files)

    def open_pr(self, title: str, body: str) -> str:
        self.pr = (title, body)
        return "https://example.invalid/pr/1"


def make_evaluation(decision_scenario: str = "promote") -> object:
    scenario = build_scenario(decision_scenario)
    return evaluate_gate(
        scenario.actual,
        scenario.champion,
        scenario.challenger,
        scenario.horizon,
        champion_version="champion",
        challenger_version="challenger",
        champion_artifact_sha256="a" * 64,
        challenger_artifact_sha256="b" * 64,
        feature_version_hash="f" * 64,
        dataset_hash="d" * 64,
        git_sha="abc",
        latency_ms=10.0,
        decided_at="2026-11-20T00:00:00+00:00",
    )


def test_retrain_opens_pr_when_gate_passes() -> None:
    evaluation = make_evaluation("promote")
    drift = DriftReport(psi={"lag_1": 0.5}, alerts=["PSI drift on lag_1"])
    git = RecordingGit()
    outcome = retrain_and_open_pr(evaluation, drift, git, "champion", "challenger")
    assert outcome.opened
    assert outcome.pr_url == "https://example.invalid/pr/1"
    assert git.branch == "retrain/challenger"
    assert git.files == [f"reports/promotions/{evaluation.record_sha256}.json"]
    assert git.pr is not None
    assert evaluation.record_sha256 in git.pr[1]
    assert "PSI drift on lag_1" in git.pr[1]
    assert "PROMOTE" in outcome.body


def test_retrain_stays_quiet_when_gate_rejects() -> None:
    evaluation = make_evaluation("tail")
    assert evaluation.decision is Decision.REJECT
    git = RecordingGit()
    outcome = retrain_and_open_pr(
        evaluation, DriftReport(psi={}, alerts=["drift"]), git, "champion", "challenger"
    )
    assert not outcome.opened
    assert "REJECT" in outcome.reason
    assert git.branch is None
    assert git.pr is None


def test_metrics_endpoint_exposes_required_series() -> None:
    record_forecast("online")
    record_forecast("batch")
    set_feature_psi("lag_1", 0.42)
    set_rolling_wape(0.31)
    record_drift_alert("lag_1")
    payload = metrics_payload().decode("utf-8")
    for series in (
        "forecast_requests_total",
        "feature_psi",
        "rolling_wape",
        "drift_alerts_total",
    ):
        assert series in payload
    assert 'feature="lag_1"' in payload


def test_pr_body_contains_evidence() -> None:
    evaluation = make_evaluation("promote")
    body = build_pr_body(
        evaluation, DriftReport(psi={}, alerts=[]), "champion", "challenger"
    )
    assert "Champion: `champion`" in body
    assert "Decision: **PROMOTE**" in body
    assert "- none" in body
    assert "This PR was opened automatically" in body
