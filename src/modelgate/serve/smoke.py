"""Contract and latency smoke test for the serving API.

Runs in-process by default so it works without a network; set MODELGATE_BASE_URL to probe a live
server instead. The measured p95 is written to reports/latency.json so the README never carries a
hand-typed latency number.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import numpy as np

from modelgate.config import REPORTS_DIR, SEED
from modelgate.data.generator import generate_demand
from modelgate.data.ingest import read_demand
from modelgate.features.contract import default_contract
from modelgate.registry.store import Registry
from modelgate.train.pipeline import TrainingConfig, save_model, train_model

BUDGET_MS = 150.0
SAMPLES = 40


def _client() -> httpx.Client:
    base = os.environ.get("MODELGATE_BASE_URL")
    if base:
        return httpx.Client(base_url=base, timeout=10.0)
    from fastapi.testclient import TestClient

    from modelgate.serve.app import create_app

    return TestClient(create_app())


def _ensure_champion() -> None:
    registry = Registry()
    if registry.champion_id is not None:
        return
    print("no champion registered; training a small model for the smoke test")
    try:
        frame = read_demand()
    except FileNotFoundError:
        frame = generate_demand(n_skus=3, n_days=160, seed=SEED)
    model = train_model(frame, config=TrainingConfig(n_estimators=20))
    directory = Path("models") / "smoke"
    save_model(model, directory)
    registry.register(directory, metrics={"train_wape": 0.0})


def main() -> int:
    _ensure_champion()
    client = _client()
    with client:
        version = client.get("/version").json()
        assert {"model_version", "feature_version_hash", "artifact_sha256"} <= set(version)
        assert version["feature_version_hash"] == default_contract().version_hash
        sku = "SKU-000"
        body = {"sku": sku, "horizon": 28}
        first = client.post("/forecast", json=body)
        first.raise_for_status()
        payload = first.json()
        assert set(payload) == {"sku", "model_version", "feature_version_hash", "points"}
        assert len(payload["points"]) == 28
        assert set(payload["points"][0]) == {"date", "mean", "p10", "p50", "p90"}

        batch = client.post("/forecast/batch", json={"requests": [body, body]})
        batch.raise_for_status()
        assert len(batch.json()["forecasts"]) == 2

        # Warm up, then measure the single-SKU endpoint.
        for _ in range(5):
            client.post("/forecast", json=body)
        latencies: list[float] = []
        for _ in range(SAMPLES):
            started = time.perf_counter()
            response = client.post("/forecast", json=body)
            response.raise_for_status()
            latencies.append((time.perf_counter() - started) * 1000.0)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    report = {
        "samples": len(latencies),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "budget_ms": BUDGET_MS,
        "passed": p95 < BUDGET_MS,
        "model_version": version["model_version"],
    }
    target = REPORTS_DIR / "latency.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"contract OK; p50 {p50:.1f} ms, p95 {p95:.1f} ms (budget {BUDGET_MS:.0f} ms)")
    print(f"latency report {target}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
