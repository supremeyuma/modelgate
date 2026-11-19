from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from modelgate.data.generator import generate_demand
from modelgate.registry.store import Registry
from modelgate.serve import smoke
from modelgate.serve.app import create_app
from modelgate.train.pipeline import TrainingConfig, save_model, train_model


@pytest.fixture(scope="module")
def demand() -> pd.DataFrame:
    return generate_demand(n_skus=2, n_days=140, seed=44)


def test_smoke_measures_contract_and_latency(
    tmp_path: Path,
    demand: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = train_model(demand, config=TrainingConfig(n_estimators=6))
    artifact = tmp_path / "artifact"
    save_model(model, artifact)
    registry = Registry(tmp_path / "registry")
    registry.register(artifact, metrics={}, created_at="2026-01-01T00:00:00+00:00")
    app = create_app(registry, demand)

    monkeypatch.setattr(smoke, "_ensure_champion", lambda: None)
    monkeypatch.setattr(smoke, "_client", lambda: TestClient(app))
    monkeypatch.setattr(smoke, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(smoke, "SAMPLES", 4)

    assert smoke.main() == 0
    report = json.loads((tmp_path / "latency.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["budget_ms"] == 150.0
    assert "contract OK" in capsys.readouterr().out
