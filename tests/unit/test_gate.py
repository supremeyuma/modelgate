from __future__ import annotations

from pathlib import Path

import numpy as np

from modelgate.promote.demo import build_scenario, run_demo
from modelgate.promote.gate import (
    Decision,
    GateThresholds,
    evaluate_gate,
    verify_record,
    write_record,
)

FIXED_TIME = "2026-11-20T00:00:00+00:00"
HASHES = {
    "champion_version": "champion",
    "challenger_version": "challenger",
    "champion_artifact_sha256": "a" * 64,
    "challenger_artifact_sha256": "b" * 64,
    "feature_version_hash": "f" * 64,
    "dataset_hash": "d" * 64,
    "git_sha": "abc123",
}


def gate(actual: np.ndarray, champion: np.ndarray, challenger: np.ndarray, horizon: np.ndarray,
         latency: float | None = 10.0, **kwargs: object) -> object:
    return evaluate_gate(
        actual,
        champion,
        challenger,
        horizon,
        latency_ms=latency,
        decided_at=FIXED_TIME,
        **HASHES,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def test_gate_promotes_clearly_better_model() -> None:
    scenario = build_scenario("promote")
    evaluation = gate(
        scenario.actual, scenario.champion, scenario.challenger, scenario.horizon
    )
    assert evaluation.decision is Decision.PROMOTE, evaluation.reasons
    assert evaluation.checks["tail_risk_bounded"]
    assert evaluation.checks["bias_within_bound"]
    assert evaluation.stats["delta_wape"] < 0


def test_gate_rejects_model_with_no_significant_difference() -> None:
    rng = np.random.default_rng(1)
    actual = rng.uniform(30, 80, 300)
    errors = rng.normal(0, 4, 300)
    predictions = actual + errors
    evaluation = gate(
        actual, predictions, predictions, np.tile(np.arange(1, 29), 11)[:300]
    )
    assert evaluation.decision in {Decision.REJECT, Decision.INCONCLUSIVE}
    assert not evaluation.checks["mean_wape_improves"]
    assert not evaluation.checks["dm_significant"]


def test_gate_rejects_model_that_wins_on_mean_but_loses_tail() -> None:
    scenario = build_scenario("tail")
    evaluation = gate(
        scenario.actual, scenario.champion, scenario.challenger, scenario.horizon
    )
    assert evaluation.decision is Decision.REJECT
    assert evaluation.checks["mean_wape_improves"]
    assert not evaluation.checks["tail_risk_bounded"]
    assert any("tail" in reason for reason in evaluation.reasons)


def test_gate_inconclusive_on_small_sample() -> None:
    scenario = build_scenario("promote", n=28)
    evaluation = gate(
        scenario.actual, scenario.champion, scenario.challenger, scenario.horizon
    )
    assert evaluation.decision is Decision.INCONCLUSIVE
    assert "paired observations" in evaluation.reasons[0]


def test_gate_rejects_without_measured_latency() -> None:
    scenario = build_scenario("promote")
    evaluation = gate(
        scenario.actual, scenario.champion, scenario.challenger, scenario.horizon, latency=None
    )
    assert evaluation.decision is Decision.REJECT
    assert not evaluation.checks["latency_within_budget"]


def test_gate_is_invariant_to_row_order() -> None:
    scenario = build_scenario("promote")
    baseline = gate(scenario.actual, scenario.champion, scenario.challenger, scenario.horizon)
    order = np.random.default_rng(4).permutation(len(scenario.actual))
    shuffled = gate(
        scenario.actual[order],
        scenario.champion[order],
        scenario.challenger[order],
        scenario.horizon[order],
    )
    assert shuffled.decision is baseline.decision
    assert shuffled.stats["dm_statistic"] == baseline.stats["dm_statistic"]
    assert shuffled.stats["dm_p_value"] == baseline.stats["dm_p_value"]
    assert shuffled.stats["bootstrap_ci"] == baseline.stats["bootstrap_ci"]
    assert shuffled.record_sha256 == baseline.record_sha256


def test_promotion_record_is_fully_reproducible_from_hashes(tmp_path: Path) -> None:
    scenario = build_scenario("promote")
    first = gate(scenario.actual, scenario.champion, scenario.challenger, scenario.horizon)
    second = gate(scenario.actual, scenario.champion, scenario.challenger, scenario.horizon)
    assert first.record_sha256 == second.record_sha256
    assert verify_record(first.record)
    assert first.record["champion"]["artifact_sha256"] == "a" * 64
    assert first.record["feature_version_hash"] == "f" * 64

    tampered = dict(first.record)
    tampered["decision"] = "PROMOTE" if first.record["decision"] != "PROMOTE" else "REJECT"
    assert not verify_record(tampered)

    path = write_record(first.record, tmp_path)
    assert path.exists()
    assert path.name.startswith("promotion-")
    assert path.name.endswith(".json")


def test_gate_thresholds_are_explicit() -> None:
    limits = GateThresholds()
    assert limits.dm_p_value == 0.05
    assert limits.max_tail_degradation_pct == 5.0
    assert limits.max_bias_pct == 2.0
    assert limits.latency_budget_ms == 150.0


def test_demo_scenarios_produce_expected_decisions(tmp_path: Path) -> None:
    rejected, path = run_demo("tail", directory=tmp_path)
    assert rejected.decision is Decision.REJECT
    assert path is not None and path.exists()
    promoted, _ = run_demo("promote", dry_run=True)
    assert promoted.decision is Decision.PROMOTE
