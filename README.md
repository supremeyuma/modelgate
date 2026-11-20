# ModelGate — ML training, promotion and drift platform

Repository status: **complete (M0–M6)**. The loop runs offline on a clean checkout:
`make setup && make ci && make demo`.

Forecast demand for small-business POS data, then let the system — not a human — decide whether a
new model ships, and notice when the shipped one starts to rot.

- Build plan: [PLAN.md](PLAN.md)
- Architecture (Mermaid): [docs/architecture.md](docs/architecture.md)
- Case study with measured results: [docs/case-study.md](docs/case-study.md)
- Decisions: [docs/decisions/](docs/decisions/)
- Agent rules: [AGENTS.md](AGENTS.md)

Commands: `make data | train | evaluate | promote DEMO=1 | serve | smoke | monitor-demo | demo |
report | ci`.

## Why this exists

Most portfolio ML projects stop at a notebook that reports a score. The hard part is what happens
next: a candidate model must be tested against the incumbent, promoted only on evidence, served with
a pinned version, monitored for drift, and retrained without a human copying files. This repo is
that loop, end to end.

The promotion gate is the differentiator: champion/challenger with a paired significance test on
forecast errors and a tail-risk bound — the same discipline as a UI eval gate, applied to tabular
models.

## Target results (to be filled from `make report`, never typed by hand)

<!-- metrics:start -->
| Metric | Baseline | ModelGate | Status |
|---|---|---|---|
| WAPE (rolling-origin, h=28) | seasonal-naive 0.2871 | 0.1565 | measured, 6 folds |
| MASE | seasonal-naive 1.20 (by definition 1.00) | 0.66 | measured |
| p95 inference latency | budget < 150 ms | 64.8 ms | measured |
| Challengers rejected by the gate | — | 2 of 3 records | signed records |
| Drift detection | 0 (no monitoring) | detected, 28-day window | injected-shift demo |
<!-- metrics:end -->

## CV block

- Built an end-to-end forecasting platform where promotion is a paired statistical decision
  (Diebold-Mariano, bootstrap interval, tail-risk bound) with signed, content-addressed records.
- Enforced point-in-time feature construction with a property test that rebuilds rows from truncated
  history, and determinism with explicit seeds and dataset/feature hashes.
- Shipped a version-pinned FastAPI service that refuses tampered artifacts, plus PSI/KS drift
  monitoring and PR-based retraining that cannot auto-deploy.

## Not in scope
- A general-purpose feature store or Kubernetes. Docker Compose and one host are enough.
- Deep sequence models. A boosted tree must be the bar to beat; only add a second family if the
  comparison is interesting.
- Multi-tenant SaaS concerns.
