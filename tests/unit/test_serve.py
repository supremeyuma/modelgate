from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from modelgate.data.generator import generate_demand
from modelgate.features.contract import default_contract
from modelgate.registry.store import Registry
from modelgate.serve.app import MAX_BATCH, create_app
from modelgate.train.pipeline import TrainingConfig, save_model, train_model


@pytest.fixture(scope="module")
def demand() -> pd.DataFrame:
    return generate_demand(n_skus=3, n_days=150, seed=31)


@pytest.fixture(scope="module")
def registry(tmp_path_factory: pytest.TempPathFactory, demand: pd.DataFrame) -> Registry:
    root = tmp_path_factory.mktemp("serving")
    model = train_model(demand, config=TrainingConfig(n_estimators=10, horizon=28))
    artifact = root / "artifact"
    save_model(model, artifact)
    registry = Registry(root / "registry")
    registry.register(artifact, metrics={"wape": 0.2}, created_at="2026-01-01T00:00:00+00:00")
    return registry


@pytest.fixture(scope="module")
def client(registry: Registry, demand: pd.DataFrame) -> Iterator[TestClient]:
    with TestClient(create_app(registry, demand)) as test_client:
        yield test_client


def test_api_contract_schema(client: TestClient) -> None:
    version = client.get("/version")
    assert version.status_code == 200
    assert set(version.json()) == {
        "model_version",
        "feature_version_hash",
        "artifact_sha256",
        "dataset_hash",
    }
    assert version.json()["feature_version_hash"] == default_contract().version_hash

    health = client.get("/health").json()
    assert health["status"] == "ok"

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    for series in ("forecast_requests_total", "feature_psi", "rolling_wape"):
        assert series in metrics.text

    response = client.post("/forecast", json={"sku": "SKU-001", "horizon": 14})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"sku", "model_version", "feature_version_hash", "points"}
    assert payload["sku"] == "SKU-001"
    assert len(payload["points"]) == 14
    assert set(payload["points"][0]) == {"date", "mean", "p10", "p50", "p90"}
    assert version.json()["model_version"] == payload["model_version"]


def test_forecast_is_deterministic(client: TestClient) -> None:
    body = {"sku": "SKU-002", "horizon": 28, "promo_days": [3, 4]}
    first = client.post("/forecast", json=body).json()
    second = client.post("/forecast", json=body).json()
    assert first == second


def test_forecast_validates_requests(client: TestClient) -> None:
    assert client.post("/forecast", json={"sku": "SKU-000", "horizon": 29}).status_code == 422
    assert client.post("/forecast", json={"sku": "SKU-000", "horizon": 0}).status_code == 422
    assert (
        client.post("/forecast", json={"sku": "SKU-000", "promo_days": [40]}).status_code == 422
    )
    assert client.post("/forecast", json={"sku": "NOPE"}).status_code == 404


def test_batch_larger_than_limit_is_rejected(client: TestClient) -> None:
    too_many = {"requests": [{"sku": "SKU-000", "horizon": 1}] * (MAX_BATCH + 1)}
    assert client.post("/forecast/batch", json=too_many).status_code == 422
    ok = {"requests": [{"sku": "SKU-000", "horizon": 7}] * 3}
    response = client.post("/forecast/batch", json=ok)
    assert response.status_code == 200
    assert len(response.json()["forecasts"]) == 3


def test_startup_rejects_tampered_artifact(
    tmp_path: Path, registry: Registry, demand: pd.DataFrame
) -> None:
    tampered = Registry(tmp_path / "registry")
    artifact = tmp_path / "artifact"
    save_model(train_model(demand, config=TrainingConfig(n_estimators=5)), artifact)
    version = tampered.register(artifact, metrics={}, created_at="2026-01-01T00:00:00+00:00")
    tampered.path_for(version["version_id"]).joinpath("model.joblib").write_bytes(b"tampered")
    with (
        pytest.raises(RuntimeError, match="refusing to start"),
        TestClient(create_app(tampered, demand)),
    ):
        pass


@pytest.mark.slow
def test_p95_latency_under_budget(client: TestClient) -> None:
    body = {"sku": "SKU-000", "horizon": 28}
    for _ in range(5):
        client.post("/forecast", json=body)
    latencies = []
    for _ in range(40):
        started = time.perf_counter()
        assert client.post("/forecast", json=body).status_code == 200
        latencies.append((time.perf_counter() - started) * 1000.0)
    assert float(np.percentile(latencies, 95)) < 150.0
