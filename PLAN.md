# PLAN — ModelGate (training, promotion and drift platform)

Status: **M0** · Target: 7 milestones / ~4 weeks

The contract for the coding agent. Work one milestone at a time; a milestone is done only when its
**acceptance commands** pass from a clean clone. Update checkboxes in the same commit.

---

## 1. What the repo must prove

1. End-to-end loop: data → features → train → backtest → **promote** → serve → monitor → retrain.
2. Promotion is a **statistical decision**, not a manual one: champion/challenger, paired test, tail
   risk, signed record.
3. Features cannot see the future; a failing model cannot reach production; drift is detected before
   accuracy breaches an SLA.

## 2. Non-goals

- No Kubernetes, no feature store product, no cloud vendor lock-in. Docker Compose only.
- No model that can't be explained in one commit message.
- No metric in the README typed by hand — `make report` emits them.
- No test that calls a live external service.

## 3. Repo layout (target)

```
modelgate/
  pyproject.toml
  Makefile                    # setup | data | train | evaluate | promote | serve | report | ci
  docker-compose.yml          # postgres + (optional) mlflow + api
  src/modelgate/
    data/ingest.py            # CSV/parquet -> DuckDB, partitioned by date
    data/generator.py         # documented synthetic demand DGP (seasonality, promo, holidays)
    features/contract.py      # feature spec + version hash + point-in-time rules
    features/build.py         # as-of joins, lags, rolling stats
    validation/checks.py      # schema, ranges, missingness, row-count, distribution
    train/pipeline.py         # fit + log to MLflow (params, metrics, dataset hash)
    train/baselines.py        # naive, seasonal-naive, moving average
    backtest/rolling_origin.py# expanding/rolling windows, horizon h
    eval/metrics.py           # WAPE, MASE, bias, pinball loss
    registry/store.py         # model store (MLflow registry or file+manifest)
    promote/gate.py           # champion/challenger decision + signed record
    promote/stats.py          # Diebold-Mariano on loss differentials, bootstrap CI
    serve/app.py              # FastAPI: /forecast (online) + batch endpoint
    monitor/drift.py          # PSI (features), KS (residuals), performance decay
    monitor/metrics.py        # Prometheus exporters
    retrain/schedule.py       # scheduled retrain -> branch + PR
  tests/{unit,properties,integration,fixtures}
  docs/{case-study.md,architecture.md,decisions/}
  reports/
```

## 4. Agent operating rules

- **Gate discipline:** milestone N+1 starts only when `make ci` is green for N.
- **Leakage is the primary risk.** Any feature builder must pass the point-in-time test suite before
  it is used anywhere.
- **Determinism:** seeds everywhere; training twice on the same data hash produces an identical
  metric tuple (assert it).
- **Commits:** Conventional Commits, one per meaningful slice (listed per milestone below).
- **Honesty:** the gate must be able to say no. A demo where every challenger is promoted is a
  failed demo, not a good one.

## 5. Milestones

### M0 — Scaffold (0.5 day)

**Build**
- `pyproject.toml` (uv, ruff, mypy strict, pytest), `Makefile`, CI (lint → types → tests).
- `docker-compose.yml` placeholder (postgres, api) — no logic yet.

**Acceptance**
```bash
make setup && make ci
```

**Commit** `chore: scaffold package, tooling, compose and CI`

---

### M1 — Data, feature contract, validation (3 days)

**Build**
- `data/generator.py`: synthetic daily demand per SKU with weekly seasonality, trend, promo spikes,
  holiday effects and noise. Document the DGP in the module docstring — every property tested later
  depends on it.
- `data/ingest.py`: load to DuckDB, partitioned by date; `data/raw` gitignored, generator output
  reproducible from a seed.
- `features/contract.py`: declarative feature spec — name, window, source, point-in-time rule,
  dtype. Compute a **feature version hash** from the spec; log it with every model.
- `features/build.py`: lags (t−1…t−28), rolling mean/std, day-of-week, promo flags, holiday flags,
  as-of joins only.
- `validation/checks.py`: schema, dtype, null rate, negative-demand check, row-count drift, and a
  hard failure on duplicate (sku, date).

**Tests**
- `test_generator_is_deterministic_for_seed` and `test_generator_produces_expected_seasonality`
  (autocorrelation at lag 7 above threshold).
- `test_as_of_join_never_uses_future_rows` (property: for every row, max source timestamp < target
  date).
- `test_lag_features_match_manual_shift` (golden values).
- `test_feature_version_hash_changes_when_spec_changes`.
- `test_validation_rejects_duplicates_and_nulls`.
- `test_validation_catches_injected_row_count_drop` (30% missing rows fails).

**Acceptance**
```bash
make data && make ci
python -m modelgate.features.build --check-point-in-time   # exits 0 only if no leakage
```

**Commit**
- `feat(data): synthetic demand generator with documented DGP`
- `feat(features): declarative feature contract, as-of joins and version hash`
- `feat(validation): schema, range and row-count gates`

---

### M2 — Training and rolling-origin backtest (3 days)

**Build**
- `train/baselines.py`: naive (last value), seasonal-naive (same weekday, 7/28), moving average.
- `train/pipeline.py`: LightGBM on lag/rolling features; global model per SKU family; log params,
  metrics, feature version hash and dataset hash to MLflow (local file store is fine).
- `backtest/rolling_origin.py`: expanding-window, horizon h=28, ≥6 folds, per-fold and aggregate
  metrics.
- `eval/metrics.py`: WAPE, MASE (vs seasonal-naive), bias by horizon, pinball loss at p10/p50/p90.

**Tests**
- `test_backtest_folds_never_overlap_train_and_test`.
- `test_seasonal_naive_scores_mase_one_by_definition` (sanity of the metric).
- `test_wape_matches_hand_computed_example` (golden).
- `test_training_is_deterministic_for_seed` (same data hash → same metrics).
- `test_model_beats_naive_baseline_on_generated_data` (the generator has structure; failing this
  means the pipeline is broken).
- `test_metrics_logged_to_mlflow_include_feature_hash`.

**Acceptance**
```bash
make train && make evaluate
# reports/backtest.json: per-fold WAPE, MASE, bias, pinball, dataset hash
```

**Commit**
- `feat(train): LightGBM pipeline with MLflow tracking and dataset hashing`
- `feat(backtest): rolling-origin evaluation with baselines and per-horizon bias`
- `test(eval): metric golden values and MASE sanity checks`

---

### M3 — Registry and the promotion gate (4 days) — the point of the repo

**Build**
- `registry/store.py`: versions with immutable artifacts, metadata (metrics, hashes, git sha,
  timestamp), `champion` pointer, `promote()`/`rollback()`.
- `promote/stats.py`: Diebold-Mariano test on per-observation loss differentials (absolute error),
  bootstrap CI on the WAPE delta, and a **tail-risk** check (worst-decile window WAPE must not
  degrade by more than X%).
- `promote/gate.py`: decision function returning `PROMOTE | REJECT | INCONCLUSIVE` plus a signed
  record (`promotion-<ts>.json` with data hash, metric deltas, p-value, CI, git sha, decision).
- CLI: `python -m modelgate.promote --challenger <id> --champion <id> --dry-run`.

**Tests**
- `test_gate_rejects_model_with_no_significant_difference` (same model twice → REJECT/INCONCLUSIVE).
- `test_gate_rejects_model_that_wins_on_mean_but_loses_tail` (constructed distributions).
- `test_gate_promotes_clearly_better_model` (inject a real improvement).
- `test_dm_test_matches_reference_values` (fixed arrays, known statistic).
- `test_gate_is_invariant_to_row_order` (property).
- `test_promotion_record_is_fully_reproducible_from_hashes`.
- `test_rollback_restores_previous_champion`.

**Acceptance**
```bash
make promote DEMO=1
# demo: candidate B is better on mean, worse on tail -> REJECT, record written
```

**Commit**
- `feat(registry): versioned model store with champion pointer and rollback`
- `feat(promote): Diebold-Mariano gate with tail-risk bound`
- `feat(promote): CLI with dry-run and signed promotion records`
- `test(promote): negative, tail-risk and reproducibility cases`

---

### M4 — Serving (2 days)

**Build**
- `serve/app.py`: `POST /forecast` (online, single SKU, h≤28) and `POST /forecast/batch`.
- Model version pinned per process start; `/health` and `/version` report the loaded champion and
  feature hash.
- Startup fails loudly if the champion artifact hash doesn't match its record.
- Request validation (pydantic), timeouts, max batch size.

**Tests**
- `test_api_contract_schema` (golden response shape).
- `test_startup_rejects_tampered_artifact`.
- `test_forecast_is_deterministic`.
- `test_batch_larger_than_limit_is_rejected`.
- `test_p95_latency_under_budget` (locust/k6 script; assert p95 < 150 ms on the reference machine,
  recorded in the report, marked `slow` so CI can opt in).

**Acceptance**
```bash
docker compose up -d && make serve &
curl -s localhost:8000/version        # champion + feature hash
make smoke                            # contract + latency smoke
```

**Commit**
- `feat(serve): FastAPI online and batch endpoints with pinned version`
- `test(serve): contract, tamper detection and latency budget`

---

### M5 — Monitoring, drift and retraining (3 days)

**Build**
- `monitor/drift.py`: PSI per feature (binned, reference window = training distribution), KS on
  residuals, and performance decay (rolling WAPE vs backtest expectation).
- Alert thresholds configurable; injected shift must trigger, stationary data must not.
- `monitor/metrics.py`: Prometheus counters/gauges: `forecast_requests_total`,
  `feature_psi{feature}`, `rolling_wape`, `drift_alerts_total`.
- `retrain/schedule.py`: on drift breach or schedule → pull fresh data, retrain, run the gate, and
  open a **PR** with the promotion record attached (never auto-deploy).

**Tests**
- `test_psi_flags_injected_mean_shift` (property over shift magnitude).
- `test_psi_does_not_flag_stationary_sample` (false-positive rate < 5% over 200 draws).
- `test_retrain_opens_pr_when_gate_passes` and `test_retrain_stays_quiet_when_gate_rejects`
  (mock the git/gh layer).
- `test_metrics_endpoint_exposes_required_series`.

**Acceptance**
```bash
make monitor-demo   # injects drift, prints PSI table, raises alert, dry-run retrain PR body
```

**Commit**
- `feat(monitor): PSI/KS drift detection with false-positive budget`
- `feat(monitor): Prometheus metrics and alert thresholds`
- `feat(retrain): drift-triggered retrain that opens a reviewed PR`

---

### M6 — Documentation, demo, release (2 days)

**Build**
- `docs/architecture.md` with a diagram (Mermaid) of the loop and the gate.
- `docs/case-study.md`: problem, data, method, results vs baseline, what the gate rejected and why,
  limits, next step.
- Recorded demo script `make demo` that runs train → gate reject → gate promote → serve → drift.
- Fill README results table from `reports/`.
- Tag `v1.0.0`; only then add the CV block.

**Acceptance**
```bash
git clone <repo> /tmp/modelgate && cd /tmp/modelgate && make setup && make ci && make demo
```

**Commit**
- `docs: architecture, case study and measured README`
- `chore(release): v1.0.0`

---

## 6. Gate specification (fixed now)

A challenger is promoted only if **all** hold:

1. Mean WAPE improves on rolling-origin backtest.
2. Diebold-Mariano p < 0.05 on the loss differential (one-sided, challenger better).
3. Bootstrap 95% CI on ΔWAPE excludes 0.
4. Worst-decile window WAPE does not degrade by more than 5%.
5. Bias by horizon does not exceed ±2% at any horizon.
6. Latency budget respected.

Otherwise `REJECT`, and if the sample is too small to decide, `INCONCLUSIVE` — never a silent
promotion. Every decision writes a record with hashes, git sha, metrics and the reason.

## 7. Test strategy summary

| Layer | What it protects |
|---|---|
| Point-in-time property tests | no future leakage in features |
| Golden-metric tests | metric implementations don't drift silently |
| Synthetic improvement/rejection | the gate can actually say no |
| Determinism tests | reproducibility claims are true |
| Drift power/false-alarm tests | monitoring is not theatre |
| API contract + tamper tests | serving can't load an unverified artifact |
| Latency budget | the SLA claim is measured, not asserted |

Coverage target: 85% on `src/modelgate`, enforced in CI.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Synthetic data is too easy — model trivially wins | DGP includes regime shifts and promo noise; baselines reported honestly |
| Gate over-fits to one backtest window | ≥6 rolling folds; tail-risk check |
| MLflow adds ops weight | file-store backend by default; server optional in compose |
| Auto-retrain ships junk | retrain only opens a PR; promotion requires the gate |

## 9. Done checklist

- [x] M0 scaffold + CI
- [x] M1 generator, feature contract, leakage gates
- [x] M2 training, MLflow, rolling-origin backtest
- [x] M3 registry + statistical promotion gate + signed records
- [x] M4 FastAPI serving with pinned versions
- [x] M5 drift monitoring + PR-based retraining
- [x] M6 architecture, case study, demo, v1.0.0, CV block
