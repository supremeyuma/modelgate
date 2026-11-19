"""Command line entry point for promotion decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from modelgate.promote.demo import run_demo
from modelgate.promote.gate import evaluate_gate
from modelgate.registry.store import Registry


def _from_observations(path: Path, champion: str, challenger: str) -> int:
    frame = pd.read_parquet(path)
    required = {"actual", "champion", "challenger", "horizon"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"{path} must contain columns {sorted(required)}")
    registry = Registry()
    champion_manifest = registry.get(champion)
    challenger_manifest = registry.get(challenger)
    evaluation = evaluate_gate(
        frame["actual"].to_numpy(float),
        frame["champion"].to_numpy(float),
        frame["challenger"].to_numpy(float),
        frame["horizon"].to_numpy(int),
        champion_version=champion,
        challenger_version=challenger,
        champion_artifact_sha256=champion_manifest["artifact_sha256"],
        challenger_artifact_sha256=challenger_manifest["artifact_sha256"],
        feature_version_hash=challenger_manifest.get("feature_version_hash", ""),
        dataset_hash=challenger_manifest.get("dataset_hash", ""),
        git_sha=challenger_manifest.get("git_sha", "unknown"),
        latency_ms=challenger_manifest.get("latency_ms"),
    )
    print(json.dumps(evaluation.stats, indent=2, sort_keys=True))
    print(f"decision {evaluation.decision.value}: {'; '.join(evaluation.reasons)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Champion/challenger promotion gate")
    parser.add_argument("--challenger", default=None)
    parser.add_argument("--champion", default=None)
    parser.add_argument("--observations", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--demo", type=int, default=0, help="1 = tail rejection, 2 = promotion")
    args = parser.parse_args(argv)

    if args.demo:
        kind = "tail" if args.demo == 1 else "promote"
        evaluation, _ = run_demo(kind, dry_run=args.dry_run)
        # A rejected demo is a *successful* demonstration that the gate can say no.
        return 0 if evaluation.decision.value in {"REJECT", "PROMOTE"} else 1

    if not (args.challenger and args.champion and args.observations):
        parser.error("provide --challenger, --champion and --observations, or use --demo")
    _from_observations(Path(args.observations), args.champion, args.challenger)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
