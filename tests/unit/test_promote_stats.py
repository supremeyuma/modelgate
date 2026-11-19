from __future__ import annotations

import numpy as np
import pytest

from modelgate.promote.stats import (
    bootstrap_wape_delta,
    diebold_mariano,
    tail_degradation,
    window_wapes,
    worst_decile_wape,
)


def test_dm_test_matches_reference_values() -> None:
    # d = champion loss - challenger loss = [1, 2, 3, 4, 5]
    # mean 3, sample sd sqrt(2.5)=1.5811, statistic 3 / (1.5811 / sqrt(5)) = 4.2426
    result = diebold_mariano([1, 2, 3, 4, 5], [0, 0, 0, 0, 0], min_samples=5)
    assert result.statistic == pytest.approx(4.242640687)
    assert result.p_value == pytest.approx(1.105e-5, rel=1e-3)
    assert result.mean_differential == pytest.approx(3.0)
    assert result.n == 5


def test_dm_degenerate_cases() -> None:
    identical = diebold_mariano([1.0] * 40, [1.0] * 40, min_samples=30)
    assert identical.statistic == 0.0
    assert identical.p_value == 0.5
    always_better = diebold_mariano([2.0] * 40, [1.0] * 40, min_samples=30)
    assert always_better.statistic == float("inf")
    assert always_better.p_value == 0.0
    with pytest.raises(ValueError, match="at least"):
        diebold_mariano([1.0], [1.0], min_samples=30)


def test_bootstrap_ci_is_seeded_and_signed() -> None:
    rng = np.random.default_rng(2)
    actual = rng.uniform(20, 80, 200)
    champion = actual + rng.normal(0, 6, 200)
    challenger = actual + rng.normal(0, 1, 200)
    first = bootstrap_wape_delta(actual, champion, challenger, n_boot=400, seed=5)
    second = bootstrap_wape_delta(actual, champion, challenger, n_boot=400, seed=5)
    assert first == second
    observed, low, high = first
    assert observed < 0
    assert low < observed < high


def test_tail_risk_detects_degradation() -> None:
    actual = np.full(70, 50.0)
    champion = actual + 1.0
    challenger = actual + 1.0
    challenger[-7:] = actual[-7:] + 30.0
    assert worst_decile_wape(actual, challenger) > worst_decile_wape(actual, champion)
    assert tail_degradation(actual, champion, challenger) > 5.0
    assert window_wapes(actual, champion).size == 10
    with pytest.raises(ValueError, match="one window"):
        window_wapes([1.0], [1.0], window=7)
    with pytest.raises(ValueError, match=r"same length|shorter than one window"):
        tail_degradation([1.0], [1.0], [1.0, 2.0])
