from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from modelgate import report
from modelgate.promote.demo import run_demo


def _backtest() -> dict[str, Any]:
    return {
        "dataset_hash": "d" * 64,
        "feature_version_hash": "f" * 64,
        "horizon": 28,
        "n_folds": 6,
        "metrics": {
            "model": {"wape": 0.15, "mase": 0.65},
            "baselines": {
                "naive": {"wape": 0.36, "mase": 1.5},
                "seasonal_naive": {"wape": 0.28, "mase": 1.2},
                "moving_average": {"wape": 0.30, "mase": 1.3},
            },
        },
    }


def test_report_collects_verifies_and_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports = tmp_path / "reports"
    (reports / "promotions").mkdir(parents=True)
    (reports / "backtest.json").write_text(json.dumps(_backtest()), encoding="utf-8")
    (reports / "latency.json").write_text(
        json.dumps({"p95_ms": 60.0, "budget_ms": 150.0, "passed": True}), encoding="utf-8"
    )
    (reports / "drift.json").write_text(
        json.dumps({"detected": True, "window_days": 28, "psi": {}}), encoding="utf-8"
    )
    run_demo("tail", directory=reports / "promotions")
    monkeypatch.setattr(report, "REPORTS_DIR", reports)

    metrics = report.collect()
    assert metrics["wape"]["model"] == 0.15
    assert metrics["gate"]["rejections"] == 1
    table = report.render_table(metrics)
    assert "0.1500" in table and "60.0 ms" in table and "detected" in table
    assert report.METRICS_START in table

    readme = tmp_path / "README.md"
    readme.write_text(
        f"# x\n\n{report.METRICS_START}\nOLDVALUE\n{report.METRICS_END}\n\nend\n",
        encoding="utf-8",
    )
    report.update_readme(metrics, readme)
    text = readme.read_text(encoding="utf-8")
    assert "0.2800" in text and "OLDVALUE" not in text

    monkeypatch.setattr(report, "ROOT", tmp_path)
    assert report.main() == 0
    stored = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    assert stored["gate"]["records"] == 1


def test_report_rejects_unverified_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports = tmp_path / "reports"
    (reports / "promotions").mkdir(parents=True)
    (reports / "backtest.json").write_text(json.dumps(_backtest()), encoding="utf-8")
    (reports / "latency.json").write_text(
        json.dumps({"p95_ms": 60.0, "budget_ms": 150.0}), encoding="utf-8"
    )
    _, path = run_demo("tail", directory=reports / "promotions")
    assert path is not None
    record = json.loads(path.read_text(encoding="utf-8"))
    record["decision"] = "PROMOTE"
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(report, "REPORTS_DIR", reports)
    with pytest.raises(ValueError, match="failed verification"):
        report.collect()


def test_report_requires_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "REPORTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="backtest"):
        report.collect()
