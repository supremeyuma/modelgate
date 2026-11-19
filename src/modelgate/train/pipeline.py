"""LightGBM quantile pipeline with MLflow tracking and content hashes.

The model is a next-day forecaster on lag/rolling features (the feature contract is the only source
of columns). Quantile boosters at p10/p50/p90 give interval forecasts; the median is the point
forecast. Deterministic settings are explicit so two runs over the same data hash produce the same
metrics.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from modelgate.config import MLRUNS_DIR, MODELS_DIR
from modelgate.features.build import build_features
from modelgate.features.contract import FeatureContract, default_contract
from modelgate.hashing import dataset_hash, sha256_file, sha256_json
from modelgate.provenance import git_sha

QUANTILES: dict[str, float] = {"p10": 0.10, "p50": 0.50, "p90": 0.90}

# MLflow 3 puts the filesystem backend behind an explicit opt-in. PLAN.md specifies a local file
# store for this repo, so opt in before any tracking call is made.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    n_estimators: int = 80
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 20
    horizon: int = 28
    num_threads: int = 1

    def to_params(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class TrainedModel:
    boosters: dict[str, Any]
    feature_names: tuple[str, ...]
    feature_version_hash: str
    dataset_hash: str
    config: TrainingConfig
    train_rows: int
    calibration_factors: tuple[float, ...] = ()

    def factor_for(self, horizon: int) -> float:
        if 1 <= horizon <= len(self.calibration_factors):
            return float(self.calibration_factors[horizon - 1])
        return 1.0

    def predict_frame(
        self, features: pd.DataFrame, horizons: Sequence[int] | None = None
    ) -> dict[str, np.ndarray]:
        matrix = features.loc[:, list(self.feature_names)].to_numpy(dtype=float)
        if horizons is None:
            factor = np.ones(len(features), dtype=float)
        else:
            factor = np.array([self.factor_for(int(step)) for step in horizons], dtype=float)
        return {
            name: np.clip(np.asarray(booster.predict(matrix), dtype=float) * factor, 0.0, None)
            for name, booster in self.boosters.items()
        }

    def predict_one(self, features: pd.DataFrame) -> dict[str, float]:
        predictions = self.predict_frame(features, [1] * len(features))
        return {name: float(values[0]) for name, values in predictions.items()}

    def to_manifest(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "feature_version_hash": self.feature_version_hash,
            "dataset_hash": self.dataset_hash,
            "config": self.config.to_params(),
            "train_rows": self.train_rows,
            "calibration_factors": list(self.calibration_factors),
        }


@dataclass
class TrainingRun:
    model: TrainedModel
    run_id: str
    artifact_dir: Path
    artifact_sha256: str
    manifest: dict[str, Any] = field(default_factory=dict)
    train_metrics: dict[str, float] = field(default_factory=dict)


def _params(config: TrainingConfig, quantile: float) -> dict[str, Any]:
    return {
        "objective": "quantile",
        "alpha": quantile,
        "metric": "quantile",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_child_samples": config.min_child_samples,
        "seed": config.seed,
        "bagging_seed": config.seed,
        "feature_fraction_seed": config.seed,
        "data_random_seed": config.seed,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": config.num_threads,
        "verbosity": -1,
    }


def _mean_params(config: TrainingConfig) -> dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_child_samples": config.min_child_samples,
        "seed": config.seed,
        "bagging_seed": config.seed,
        "feature_fraction_seed": config.seed,
        "data_random_seed": config.seed,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": config.num_threads,
        "verbosity": -1,
    }


def train_model(
    frame: pd.DataFrame,
    contract: FeatureContract | None = None,
    config: TrainingConfig | None = None,
) -> TrainedModel:
    """Fit the mean and quantile boosters on the feature matrix derived from ``frame``."""
    active = contract or default_contract()
    settings = config or TrainingConfig()
    matrix = build_features(frame, active)
    names = active.feature_names
    features = matrix.loc[:, list(names)].to_numpy(dtype=float)
    target = matrix["demand"].to_numpy(dtype=float)

    dataset = lgb.Dataset(features, label=target, feature_name=list(names), free_raw_data=False)
    boosters: dict[str, Any] = {
        "mean": lgb.train(_mean_params(settings), dataset, num_boost_round=settings.n_estimators)
    }
    for name, quantile in QUANTILES.items():
        boosters[name] = lgb.train(
            _params(settings, quantile),
            dataset,
            num_boost_round=settings.n_estimators,
        )

    return TrainedModel(
        boosters=boosters,
        feature_names=names,
        feature_version_hash=active.version_hash,
        dataset_hash=dataset_hash(frame),
        config=settings,
        train_rows=len(matrix),
    )


def predict_history(model: TrainedModel, frame: pd.DataFrame) -> pd.DataFrame:
    """In-sample predictions for diagnostics (not used in backtests)."""
    matrix = build_features(frame, default_contract())
    predictions = model.predict_frame(matrix, [1] * len(matrix))
    result = matrix.loc[:, ["sku", "date", "demand"]].copy()
    for name, values in predictions.items():
        result[name] = values
    return result


def save_model(model: TrainedModel, directory: Path | None = None) -> tuple[Path, str]:
    target = directory or MODELS_DIR / "latest"
    target.mkdir(parents=True, exist_ok=True)
    artifact = target / "model.joblib"
    joblib.dump(
        {
            "boosters": model.boosters,
            "feature_names": list(model.feature_names),
            "feature_version_hash": model.feature_version_hash,
            "dataset_hash": model.dataset_hash,
            "config": model.config.to_params(),
            "train_rows": model.train_rows,
            "calibration_factors": list(model.calibration_factors),
        },
        artifact,
    )
    digest = sha256_file(artifact)
    manifest = model.to_manifest()
    manifest["artifact_sha256"] = digest
    manifest["artifact"] = artifact.name
    manifest["manifest_sha256"] = sha256_json(manifest)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return artifact, digest


def load_model(directory: Path | None = None) -> TrainedModel:
    target = directory or MODELS_DIR / "latest"
    payload = joblib.load(target / "model.joblib")
    return TrainedModel(
        boosters=payload["boosters"],
        feature_names=tuple(payload["feature_names"]),
        feature_version_hash=payload["feature_version_hash"],
        dataset_hash=payload["dataset_hash"],
        config=TrainingConfig(**payload["config"]),
        train_rows=int(payload["train_rows"]),
        calibration_factors=tuple(payload.get("calibration_factors", ())),
    )


def train_and_log(
    frame: pd.DataFrame,
    config: TrainingConfig | None = None,
    contract: FeatureContract | None = None,
    tracking_uri: str | Path | None = None,
    experiment: str = "modelgate",
    output_dir: Path | None = None,
) -> TrainingRun:
    """Train, persist an immutable-shaped artifact and log hashes/metrics to MLflow."""
    active = contract or default_contract()
    settings = config or TrainingConfig()
    model = train_model(frame, active, settings)

    diagnostics = predict_history(model, frame)
    error = np.abs(diagnostics["demand"].to_numpy(float) - diagnostics["mean"].to_numpy(float))
    actual = diagnostics["demand"].to_numpy(float)
    train_metrics = {
        "train_wape": float(error.sum() / actual.sum()),
        "train_bias_pct": float(
            (diagnostics["mean"].sum() - diagnostics["demand"].sum())
            / diagnostics["demand"].sum()
            * 100.0
        ),
    }

    location = output_dir or MODELS_DIR / "latest"
    artifact, digest = save_model(model, location)
    manifest = json.loads((location / "manifest.json").read_text(encoding="utf-8"))
    manifest["git_sha"] = git_sha()
    (location / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if tracking_uri is None:
        uri = MLRUNS_DIR.as_uri()
    else:
        raw = str(tracking_uri)
        uri = raw if raw.startswith("file:") else Path(raw).absolute().as_uri()
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"quantile-{settings.seed}") as run:
        mlflow.log_params({**settings.to_params(), "quantiles": ",".join(QUANTILES)})
        mlflow.set_tags(
            {
                "feature_version_hash": model.feature_version_hash,
                "dataset_hash": model.dataset_hash,
                "git_sha": manifest["git_sha"],
                "artifact_sha256": digest,
            }
        )
        mlflow.log_metrics(train_metrics)
        mlflow.log_artifact(str(artifact), artifact_path="model")
        mlflow.log_dict(active.to_dict(), "feature_contract.json")
        run_id = run.info.run_id

    return TrainingRun(
        model=model,
        run_id=run_id,
        artifact_dir=location,
        artifact_sha256=digest,
        manifest=manifest,
        train_metrics=train_metrics,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    from modelgate.data.ingest import read_demand

    parser = argparse.ArgumentParser(description="Train the quantile forecaster")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--register", action="store_true", help="register the trained model")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args(argv)

    frame = read_demand()
    run = train_and_log(
        frame,
        TrainingConfig(seed=args.seed),
        tracking_uri=args.tracking_uri,
    )
    print(f"trained {run.model.train_rows:,} rows, {len(QUANTILES)} quantile boosters")
    print(f"feature version {run.model.feature_version_hash}")
    print(f"dataset hash    {run.model.dataset_hash}")
    print(f"artifact        {run.artifact_dir} (sha256 {run.artifact_sha256[:16]}...)")
    print(f"mlflow run      {run.run_id}")
    for key, value in run.train_metrics.items():
        print(f"{key:<16}{value:.4f}")

    if args.register:
        from modelgate.registry.store import Registry

        registry = Registry()
        version = registry.register(run.artifact_dir, metrics=run.train_metrics, run_id=run.run_id)
        print(f"registered      {version['version_id']} (champion={registry.champion_id})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
