"""FastAPI serving with a champion pinned at process start.

The artifact is verified against its manifest SHA-256 before the first request; a mismatch stops
startup instead of serving an unverified model. Every response carries the model version and the
feature-contract hash that produced it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date as date_type
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

from modelgate.config import MODELS_DIR
from modelgate.features.contract import default_contract
from modelgate.forecast import recursive_forecast
from modelgate.monitor import metrics as prometheus
from modelgate.registry.store import Registry, RegistryError
from modelgate.train.pipeline import TrainedModel, TrainingConfig

MAX_BATCH = 100
MAX_HORIZON = 28


class ForecastRequest(BaseModel):
    sku: str = Field(min_length=1, max_length=64)
    horizon: int = Field(default=28, ge=1, le=MAX_HORIZON)
    promo_days: list[int] = Field(default_factory=list)
    holiday_days: list[int] = Field(default_factory=list)

    @field_validator("promo_days", "holiday_days")
    @classmethod
    def _valid_offsets(cls, values: list[int]) -> list[int]:
        if any(not 1 <= value <= MAX_HORIZON for value in values):
            raise ValueError(f"day offsets must be between 1 and {MAX_HORIZON}")
        return values


class BatchForecastRequest(BaseModel):
    requests: list[ForecastRequest] = Field(min_length=1, max_length=MAX_BATCH)


class ForecastPoint(BaseModel):
    date: date_type
    mean: float
    p10: float
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    sku: str
    model_version: str
    feature_version_hash: str
    points: list[ForecastPoint]


class BatchForecastResponse(BaseModel):
    forecasts: list[ForecastResponse]


class ModelNotLoadedError(RuntimeError):
    pass


def _load_model(registry: Registry) -> tuple[TrainedModel, dict[str, Any]]:
    try:
        artifact, manifest = registry.champion_artifact()
    except RegistryError as error:
        raise ModelNotLoadedError(str(error)) from error
    payload = joblib.load(artifact)
    if manifest.get("artifact_sha256") is None:
        raise ModelNotLoadedError("manifest has no artifact hash")
    contract = default_contract()
    if tuple(payload["feature_names"]) != contract.feature_names:
        raise ModelNotLoadedError("artifact feature names do not match the active contract")
    if payload["feature_version_hash"] != contract.version_hash:
        raise ModelNotLoadedError("artifact feature hash does not match the active contract")
    model = TrainedModel(
        boosters=payload["boosters"],
        feature_names=tuple(payload["feature_names"]),
        feature_version_hash=payload["feature_version_hash"],
        dataset_hash=payload["dataset_hash"],
        config=TrainingConfig(**payload["config"]),
        train_rows=int(payload["train_rows"]),
        calibration_factors=tuple(payload.get("calibration_factors", ())),
    )
    return model, manifest


def forecast_points(
    model: TrainedModel,
    history: pd.DataFrame,
    sku: str,
    horizon: int,
    promo_days: list[int] | None = None,
    holiday_days: list[int] | None = None,
) -> pd.DataFrame:
    history = history.loc[history["sku"] == sku].sort_values("date").copy()
    if history.empty:
        raise KeyError(sku)
    last_date = pd.Timestamp(history["date"].max())
    future_dates = [last_date + pd.Timedelta(days=offset) for offset in range(1, horizon + 1)]
    family = str(history["family"].iloc[-1])
    promo = set(promo_days or [])
    holiday = set(holiday_days or [])
    calendar = pd.DataFrame(
        {
            "sku": sku,
            "date": future_dates,
            "family": family,
            "promo": [offset in promo for offset in range(1, horizon + 1)],
            "holiday": [offset in holiday for offset in range(1, horizon + 1)],
        }
    )
    return recursive_forecast(model, history, calendar)


def create_app(
    registry: Registry | None = None,
    demand: pd.DataFrame | None = None,
) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_registry = registry or Registry(MODELS_DIR / "registry")
        try:
            model, manifest = _load_model(active_registry)
        except ModelNotLoadedError as error:
            raise RuntimeError(f"refusing to start: {error}") from error
        if demand is None:
            from modelgate.data.ingest import read_demand

            frame = read_demand()
        else:
            frame = demand
        state.update(
            {
                "model": model,
                "manifest": manifest,
                "demand": frame,
                "version": str(manifest.get("version_id", "unknown")),
                "feature_hash": model.feature_version_hash,
            }
        )
        app.state.ready = True
        yield
        state.clear()
        app.state.ready = False

    app = FastAPI(title="ModelGate", version="1.0.0", lifespan=lifespan)

    def _require() -> dict[str, Any]:
        if not state:
            raise HTTPException(status_code=503, detail="model is not loaded")
        return state

    def _forecast(request: ForecastRequest) -> ForecastResponse:
        current = _require()
        prometheus.record_forecast("online")
        try:
            points = forecast_points(
                current["model"],
                current["demand"],
                request.sku,
                request.horizon,
                request.promo_days,
                request.holiday_days,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"unknown sku {request.sku!r}") from error
        return ForecastResponse(
            sku=request.sku,
            model_version=current["version"],
            feature_version_hash=current["feature_hash"],
            points=[
                ForecastPoint(
                    date=day, mean=mean, p10=p10, p50=p50, p90=p90
                )
                for day, mean, p10, p50, p90 in zip(
                    pd.to_datetime(points["date"]).dt.date.tolist(),
                    points["mean"].astype(float).tolist(),
                    points["p10"].astype(float).tolist(),
                    points["p50"].astype(float).tolist(),
                    points["p90"].astype(float).tolist(),
                    strict=True,
                )
            ],
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        current = _require()
        return {"status": "ok", "model_version": current["version"]}

    @app.get("/version")
    def version() -> dict[str, Any]:
        current = _require()
        manifest = current["manifest"]
        return {
            "model_version": current["version"],
            "feature_version_hash": current["feature_hash"],
            "artifact_sha256": manifest["artifact_sha256"],
            "dataset_hash": manifest.get("dataset_hash"),
        }

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(request: ForecastRequest) -> ForecastResponse:
        return _forecast(request)

    @app.post("/forecast/batch", response_model=BatchForecastResponse)
    def forecast_batch(request: BatchForecastRequest) -> BatchForecastResponse:
        prometheus.record_forecast("batch")
        return BatchForecastResponse(forecasts=[_forecast(item) for item in request.requests])

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(
            content=prometheus.metrics_payload(),
            media_type="text/plain; version=0.0.4",
        )

    return app


app = create_app()


def main() -> int:
    import uvicorn

    uvicorn.run("modelgate.serve.app:app", host="127.0.0.1", port=8000, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
