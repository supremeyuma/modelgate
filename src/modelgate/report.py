"""Generate reports/metrics.json and the README results table from measured artefacts.

Nothing in the README table is typed by hand: every value comes from reports/backtest.json,
reports/latency.json, reports/drift.json or the signed promotion records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from modelgate.config import REPORTS_DIR, ROOT
from modelgate.promote.gate import verify_record

METRICS_START = "<!-- metrics:start -->"
METRICS_END = "<!-- metrics:end -->"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def collect() -> dict[str, Any]:
    backtest = _load(REPORTS_DIR / "backtest.json")
    latency = _load(REPORTS_DIR / "latency.json")
    drift = _load(REPORTS_DIR / "drift.json")
    if backtest is None:
        raise FileNotFoundError("reports/backtest.json missing; run `make evaluate`")
    if latency is None:
        raise FileNotFoundError("reports/latency.json missing; run `make smoke`")

    promotions = sorted((REPORTS_DIR / "promotions").glob("promotion-*.json"))
    decisions: list[str] = []
    rejected_tail = 0
    for path in promotions:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not verify_record(record):
            raise ValueError(f"promotion record failed verification: {path}")
        decisions.append(record["decision"])
        if record["decision"] == "REJECT" and not record["checks"].get(
            "tail_risk_bounded", True
        ):
            rejected_tail += 1

    metrics = backtest["metrics"]
    return {
        "dataset_hash": backtest["dataset_hash"],
        "feature_version_hash": backtest["feature_version_hash"],
        "horizon": backtest["horizon"],
        "folds": backtest["n_folds"],
        "wape": {
            "model": metrics["model"]["wape"],
            **{name: value["wape"] for name, value in metrics["baselines"].items()},
        },
        "mase": {
            "model": metrics["model"]["mase"],
            **{name: value["mase"] for name, value in metrics["baselines"].items()},
        },
        "latency": latency,
        "drift": drift,
        "gate": {
            "promotions": decisions.count("PROMOTE"),
            "rejections": decisions.count("REJECT"),
            "inconclusive": decisions.count("INCONCLUSIVE"),
            "rejected_for_tail_risk": rejected_tail,
            "records": len(decisions),
        },
    }


def render_table(metrics: dict[str, Any]) -> str:
    wape = metrics["wape"]
    mase = metrics["mase"]
    latency = metrics["latency"]
    drift = metrics["drift"] or {}
    gate = metrics["gate"]
    drift_value = (
        f"detected, {drift['window_days']}-day window" if drift.get("detected") else "not measured"
    )
    lines = [
        METRICS_START,
        "| Metric | Baseline | ModelGate | Status |",
        "|---|---|---|---|",
        (
            f"| WAPE (rolling-origin, h={metrics['horizon']}) | seasonal-naive "
            f"{wape['seasonal_naive']:.4f} | {wape['model']:.4f} | measured, "
            f"{metrics['folds']} folds |"
        ),
        (
            f"| MASE | seasonal-naive {mase['seasonal_naive']:.2f} (by definition 1.00) | "
            f"{mase['model']:.2f} | measured |"
        ),
        (
            f"| p95 inference latency | budget < {latency['budget_ms']:.0f} ms | "
            f"{latency['p95_ms']:.1f} ms | measured |"
        ),
        (
            f"| Challengers rejected by the gate | — | {gate['rejections']} of "
            f"{gate['records']} records | signed records |"
        ),
        (
            f"| Drift detection | 0 (no monitoring) | {drift_value} | injected-shift demo |"
        ),
        METRICS_END,
    ]
    return "\n".join(lines)


def update_readme(metrics: dict[str, Any], path: Path | None = None) -> Path:
    target = path or ROOT / "README.md"
    text = target.read_text(encoding="utf-8")
    if METRICS_START not in text or METRICS_END not in text:
        raise ValueError("README is missing the metrics markers")
    before = text.split(METRICS_START)[0]
    after = text.split(METRICS_END)[1]
    target.write_text(before + render_table(metrics) + after, encoding="utf-8")
    return target


def main() -> int:
    metrics = collect()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target = REPORTS_DIR / "metrics.json"
    target.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    readme = update_readme(metrics)
    print(render_table(metrics))
    print(f"metrics {target}")
    print(f"readme  {readme}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
