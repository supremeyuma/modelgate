from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modelgate.monitor.drift import (
    DriftThresholds,
    detect_feature_drift,
    detect_performance_decay,
    ks_residuals,
    population_stability_index,
    rolling_wape,
)


def test_psi_flags_injected_mean_shift() -> None:
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, 2000)
    for shift in (1.0, 1.5, 2.0):
        current = rng.normal(shift, 1, 2000)
        score = population_stability_index(reference, current)
        assert score > 0.20, (shift, score)


def test_psi_does_not_flag_stationary_sample() -> None:
    rng = np.random.default_rng(4)
    false_positives = 0
    draws = 200
    for _ in range(draws):
        reference = rng.normal(10, 3, 300)
        current = rng.normal(10, 3, 300)
        if population_stability_index(reference, current) > 0.20:
            false_positives += 1
    assert false_positives / draws < 0.05


def test_psi_guards_degenerate_input() -> None:
    assert population_stability_index([5.0] * 10, [5.0] * 10) == 0.0
    assert population_stability_index([5.0] * 10, [7.0] * 10) == float("inf")
    with pytest.raises(ValueError, match="non-empty"):
        population_stability_index([], [1.0])


def test_ks_and_rolling_wape() -> None:
    rng = np.random.default_rng(5)
    statistic, p_value = ks_residuals(rng.normal(0, 1, 500), rng.normal(2, 1, 500))
    assert statistic > 0.5
    assert p_value < 0.01
    values = rolling_wape([10, 10, 10, 10, 10, 10, 20], [10, 10, 10, 10, 10, 10, 10], window=7)
    assert values.shape == (1,)
    assert values[0] == pytest.approx(10 / 80)
    with pytest.raises(ValueError):
        rolling_wape([1.0], [1.0], window=7)


def test_feature_drift_and_decay_reports() -> None:
    rng = np.random.default_rng(6)
    reference = pd.DataFrame({"lag_1": rng.normal(0, 1, 500), "promo": rng.integers(0, 2, 500)})
    current = pd.DataFrame({"lag_1": rng.normal(3, 1, 500), "promo": rng.integers(0, 2, 500)})
    report = detect_feature_drift(reference, current, ["lag_1"])
    assert report.drifted
    assert report.psi["lag_1"] > 0.2
    assert any("lag_1" in alert for alert in report.alerts)
    with pytest.raises(KeyError):
        detect_feature_drift(reference, current, ["missing"])

    decay = detect_performance_decay(0.20, [0.20, 0.30])
    assert decay.rolling_wape == 0.30
    assert decay.drifted
    calm = detect_performance_decay(0.20, np.array([0.20, 0.21]))
    assert not calm.drifted
    with pytest.raises(ValueError, match="positive"):
        detect_performance_decay(0.0, [0.1])
    with pytest.raises(ValueError, match="empty"):
        detect_performance_decay(0.1, [])
    assert DriftThresholds().psi == 0.2


def test_monitor_demo_injects_drift_and_opens_dry_run_pr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from modelgate.data.generator import generate_demand
    from modelgate.monitor import demo

    frame = generate_demand(n_skus=2, n_days=150, seed=1)
    monkeypatch.setattr(demo, "_load", lambda: frame)
    assert demo.main() == 0
    output = capsys.readouterr().out
    assert "feature" in output and "PSI" in output
    assert "alert" in output.lower()
    assert "dry-run: gh pr create" in output
    assert "PROMOTE" in output
