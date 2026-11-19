from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from modelgate.data.generator import generate_demand
from modelgate.features.build import build_features
from modelgate.features.contract import default_contract
from modelgate.train.pipeline import (
    TrainingConfig,
    load_model,
    save_model,
    train_and_log,
    train_model,
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return generate_demand(n_skus=3, n_days=140, seed=21)


def test_metrics_logged_to_mlflow_include_feature_hash(
    panel: pd.DataFrame, tmp_path: Path
) -> None:
    tracking = tmp_path / "mlruns"
    output = tmp_path / "model"
    run = train_and_log(
        panel,
        TrainingConfig(n_estimators=12, horizon=14),
        tracking_uri=tracking,
        output_dir=output,
        experiment="test-experiment",
    )
    assert run.run_id
    experiment = mlflow.get_experiment_by_name("test-experiment")
    assert experiment is not None
    client = MlflowClient(tracking_uri=tracking.as_uri())
    stored = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(stored) == 1
    tags = stored[0].data.tags
    assert tags["feature_version_hash"] == default_contract().version_hash
    assert tags["dataset_hash"] == run.model.dataset_hash
    assert tags["artifact_sha256"] == run.artifact_sha256
    assert "train_wape" in stored[0].data.metrics

    saved = load_model(output)
    assert saved.feature_version_hash == run.model.feature_version_hash
    assert saved.boosters.keys() == {"mean", "p10", "p50", "p90"}


def test_save_and_load_round_trip(panel: pd.DataFrame, tmp_path: Path) -> None:
    model = train_model(panel, config=TrainingConfig(n_estimators=8))
    directory = tmp_path / "artifact"
    artifact, digest = save_model(model, directory)
    assert artifact.exists()
    assert len(digest) == 64
    restored = load_model(directory)
    matrix = build_features(panel, default_contract())
    np.testing.assert_allclose(
        model.predict_frame(matrix)["p50"], restored.predict_frame(matrix)["p50"]
    )
    assert restored.config == model.config
