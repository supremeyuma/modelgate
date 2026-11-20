"""Drift-triggered retraining that can only open a reviewed pull request.

Deployment stays manual: even a PROMOTE decision produces a branch and a PR, never a champion
mutation. The git/gh layer is injected so the behaviour is testable without a network.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from modelgate.monitor.drift import DriftReport
from modelgate.promote.gate import Decision, GateEvaluation


class GitOps(Protocol):
    def create_branch(self, name: str) -> None: ...

    def commit(self, files: Sequence[str], message: str) -> None: ...

    def open_pr(self, title: str, body: str) -> str: ...


@dataclass
class RetrainOutcome:
    opened: bool
    reason: str
    branch: str | None = None
    pr_url: str | None = None
    body: str | None = None


def build_pr_body(
    evaluation: GateEvaluation,
    drift: DriftReport,
    champion_version: str,
    challenger_version: str,
) -> str:
    stats = evaluation.stats
    drift_lines = (
        "\n".join(f"- {alert}" for alert in drift.alerts) if drift.alerts else "- none"
    )
    return "\n".join(
        [
            "## Retrain proposal",
            "",
            f"Champion: `{champion_version}`  ",
            f"Challenger: `{challenger_version}`  ",
            f"Decision: **{evaluation.decision.value}**",
            "",
            "### Drift evidence",
            drift_lines,
            "",
            "### Gate metrics",
            f"- WAPE: {stats['champion_wape']:.4f} -> {stats['challenger_wape']:.4f}",
            f"- DM p-value: {stats['dm_p_value']:.4g}",
            f"- Bootstrap 95% CI: {stats['bootstrap_ci']}",
            f"- Tail degradation: {stats['tail_degradation_pct']:.2f}%",
            f"- Max |bias|: {stats['max_abs_bias_pct']:.2f}%",
            f"- Promotion record: `{evaluation.record_sha256}`",
            "",
            "This PR was opened automatically. Promotion and deployment remain manual and require",
            "the signed record above.",
        ]
    )


def retrain_and_open_pr(
    evaluation: GateEvaluation,
    drift: DriftReport,
    git: GitOps,
    champion_version: str,
    challenger_version: str,
) -> RetrainOutcome:
    if evaluation.decision is not Decision.PROMOTE:
        return RetrainOutcome(
            opened=False,
            reason=f"gate decision {evaluation.decision.value}; no PR opened",
        )
    branch = f"retrain/{challenger_version}"
    body = build_pr_body(evaluation, drift, champion_version, challenger_version)
    git.create_branch(branch)
    git.commit(
        [f"reports/promotions/{evaluation.record_sha256}.json"],
        f"chore(retrain): propose {challenger_version}",
    )
    url = git.open_pr(
        f"Retrain proposal: {challenger_version} vs {champion_version}", body
    )
    return RetrainOutcome(
        opened=True,
        reason="gate promoted the challenger; PR opened for review",
        branch=branch,
        pr_url=url,
        body=body,
    )
