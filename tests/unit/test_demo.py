from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import modelgate.demo as demo
from modelgate.data.generator import generate_demand


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class _Client:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.entered = False

    def __enter__(self) -> _Client:
        self.entered = True
        return self

    def __exit__(self, *args: object) -> None:
        self.entered = False

    def get(self, path: str) -> _Response:
        return _Response(
            {
                "model_version": "v1",
                "feature_version_hash": "f" * 64,
                "artifact_sha256": "a" * 64,
            }
        )

    def post(self, path: str, json: dict[str, object]) -> _Response:
        return _Response(
            {
                "points": [
                    {"date": "2024-06-19", "mean": 42.0, "p10": 30.0, "p50": 41.0, "p90": 55.0}
                ]
            }
        )


def test_demo_runs_all_five_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    frame = generate_demand(n_skus=2, n_days=60, seed=3)
    evaluations = iter(
        [
            (
                SimpleNamespace(decision=demo.Decision.REJECT, reasons=["tail"]),
                SimpleNamespace(name="reject.json"),
            ),
            (
                SimpleNamespace(decision=demo.Decision.PROMOTE, reasons=["ok"]),
                SimpleNamespace(name="promote.json"),
            ),
        ]
    )
    monkeypatch.setattr(demo, "_ensure_data", lambda: None)
    monkeypatch.setattr(demo, "_train", lambda: ("v1", {"frame": frame}))
    monkeypatch.setattr(demo, "run_demo", lambda kind: next(evaluations))
    monkeypatch.setattr(demo, "TestClient", _Client)
    monkeypatch.setattr(demo, "create_app", lambda registry, demand: object())
    monkeypatch.setattr(demo.monitor_demo, "main", lambda: 0)
    monkeypatch.setattr(demo, "REPORTS_DIR", tmp_path)

    assert demo.main() == 0
    summary = json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))
    assert summary["rejected_record"] == "reject.json"
    assert summary["promoted_record"] == "promote.json"
    assert summary["champion"] == "v1"
    assert "demo complete" in capsys.readouterr().out


def test_ensure_data_generates_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(demo, "RAW_DIR", tmp_path / "raw")
    called: list[object] = []
    monkeypatch.setattr(
        demo,
        "write_raw",
        lambda frame: called.append("raw") or tmp_path / "raw" / "demand.parquet",
    )
    monkeypatch.setattr(demo, "ingest", lambda: called.append("ingest"))
    demo._ensure_data()
    assert called == ["raw", "ingest"]
    assert "step 0" in capsys.readouterr().out
