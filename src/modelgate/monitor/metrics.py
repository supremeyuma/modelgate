"""Prometheus series exposed by the service and the monitor."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest

REGISTRY = CollectorRegistry()

FORECAST_REQUESTS = Counter(
    "forecast_requests_total",
    "Forecast requests served",
    ["endpoint"],
    registry=REGISTRY,
)
FEATURE_PSI = Gauge(
    "feature_psi",
    "Population stability index for a feature",
    ["feature"],
    registry=REGISTRY,
)
ROLLING_WAPE = Gauge(
    "rolling_wape",
    "Rolling WAPE of the latest served forecasts",
    registry=REGISTRY,
)
DRIFT_ALERTS = Counter(
    "drift_alerts_total",
    "Drift alerts raised",
    ["feature"],
    registry=REGISTRY,
)


def record_forecast(endpoint: str = "online") -> None:
    FORECAST_REQUESTS.labels(endpoint=endpoint).inc()


def set_feature_psi(feature: str, value: float) -> None:
    FEATURE_PSI.labels(feature=feature).set(value)


def set_rolling_wape(value: float) -> None:
    ROLLING_WAPE.set(value)


def record_drift_alert(feature: str) -> None:
    DRIFT_ALERTS.labels(feature=feature).inc()


def metrics_payload() -> bytes:
    return generate_latest(REGISTRY)
