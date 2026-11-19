"""Reproducible demo scenarios for the gate.

The scenarios operate on synthetic per-observation errors so the statistical behaviour is exactly
controlled and repeatable. They are deliberately *not* tuned to flatter the gate: the default
scenario improves the mean and loses the tail, and must be rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modelgate.features.contract import default_contract
from modelgate.promote.gate import Decision, GateEvaluation, evaluate_gate, write_record

DEMO_DATASET_HASH = "d" * 64


@dataclass
class DemoScenario:
    actual: np.ndarray
    champion: np.ndarray
    challenger: np.ndarray
    horizon: np.ndarray


def build_scenario(kind: str = "tail", n: int = 560, seed: int = 7) -> DemoScenario:
    rng = np.random.default_rng(seed)
    horizon = np.tile(np.arange(1, 29), n // 28 + 1)[:n]
    actual = 60.0 + rng.uniform(-10.0, 10.0, size=n)
    weekly = np.arange(n) // 7

    champion_error = rng.normal(0.0, 4.0, size=n)
    challenger_error = rng.normal(0.0, 2.0, size=n)

    # Remove per-horizon mean error from the challenger before injecting scenario effects so the
    # bias gate is decided on its merits, not on the random seed.
    for step in range(1, 29):
        mask = horizon == step
        challenger_error[mask] -= challenger_error[mask].mean()

    weekly = np.arange(n) // 7
    tail_weeks = weekly % 10 == 9
    if kind == "tail":
        # Challenger wins on the average, loses badly in one week out of ten.
        challenger_error[tail_weeks] += 12.0
    elif kind == "promote":
        # Champion carries a tail defect the challenger does not have.
        champion_error[tail_weeks] = champion_error[tail_weeks] * 3.5 + 10.0
    else:
        raise ValueError(f"unknown scenario {kind!r}")

    return DemoScenario(
        actual=actual,
        champion=actual + champion_error,
        challenger=actual + challenger_error,
        horizon=horizon,
    )


def run_demo(
    kind: str = "tail",
    dry_run: bool = False,
    directory: Path | None = None,
) -> tuple[GateEvaluation, Path | None]:
    scenario = build_scenario(kind)
    evaluation = evaluate_gate(
        scenario.actual,
        scenario.champion,
        scenario.challenger,
        scenario.horizon,
        champion_version="demo-champion",
        challenger_version="demo-challenger",
        champion_artifact_sha256="a" * 64,
        challenger_artifact_sha256="b" * 64,
        feature_version_hash=default_contract().version_hash,
        dataset_hash=DEMO_DATASET_HASH,
        git_sha="demo",
        latency_ms=9.5,
    )
    stats: dict[str, Any] = evaluation.stats
    print(f"scenario        {kind}")
    print(f"observations    {stats['n_observations']}")
    print(f"champion WAPE   {stats['champion_wape']:.4f}")
    print(f"challenger WAPE {stats['challenger_wape']:.4f}  (delta {stats['delta_wape']:+.4f})")
    print(f"DM statistic    {stats['dm_statistic']:.3f}  p={stats['dm_p_value']:.3e}")
    print(
        f"bootstrap 95%   [{stats['bootstrap_ci'][0]:+.4f}, {stats['bootstrap_ci'][1]:+.4f}]"
    )
    print(f"tail degradation{stats['tail_degradation_pct']:+.2f}%")
    print(f"max |bias|      {stats['max_abs_bias_pct']:.2f}%")
    print(f"checks          {evaluation.checks}")
    print(f"decision        {evaluation.decision.value}")
    for reason in evaluation.reasons:
        print(f"  - {reason}")
    path = None
    if not dry_run:
        path = write_record(evaluation.record, directory)
        print(f"record          {path}")
    return evaluation, path


def expected_decision(kind: str) -> Decision:
    return Decision.REJECT if kind == "tail" else Decision.PROMOTE
