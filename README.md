# ModelGate — ML training, promotion and drift platform

Repository scaffold. **Status: M0 — plan only, no implementation yet.**

Forecast demand for small-business POS data, then let the system — not a human — decide whether a
new model ships, and notice when the shipped one starts to rot.

- Build plan: [PLAN.md](PLAN.md)
- Agent rules: [AGENTS.md](AGENTS.md)

## Why this exists

Most portfolio ML projects stop at a notebook that reports a score. The hard part is what happens
next: a candidate model must be tested against the incumbent, promoted only on evidence, served with
a pinned version, monitored for drift, and retrained without a human copying files. This repo is
that loop, end to end.

The promotion gate is the differentiator: champion/challenger with a paired significance test on
forecast errors and a tail-risk bound — the same discipline as a UI eval gate, applied to tabular
models.

## Target results (to be filled from `make report`, never typed by hand)

| Metric | Baseline | ModelGate | Status |
|---|---|---|---|
| WAPE (rolling-origin, h=28) | seasonal-naive | — | not measured |
| MASE | seasonal-naive (MASE=1) | — | not measured |
| p95 inference latency | budget < 150 ms | — | not measured |
| Challengers rejected by the gate | — | — | not measured |
| Drift detection lead time | 0 (no monitoring) | — | not measured |

## Not in scope

- A general-purpose feature store or Kubernetes. Docker Compose and one host are enough.
- Deep sequence models. A boosted tree must be the bar to beat; only add a second family if the
  comparison is interesting.
- Multi-tenant SaaS concerns.
