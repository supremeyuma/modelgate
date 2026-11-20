# 0001 - Promotion is a statistical decision

Status: accepted.

## Context

A candidate that improves WAPE on average can still be worse where it matters: the worst weeks.
A single train/test split cannot distinguish a real improvement from noise. The repository exists to
make the promotion decision honest and auditable.

## Decision

A challenger is promoted only when all of the following hold on paired rolling-origin observations:

1. Mean WAPE improves.
2. One-sided Diebold-Mariano p < 0.05 on absolute-error differentials.
3. The seeded 95% bootstrap interval on ΔWAPE excludes zero.
4. Worst-decile weekly WAPE degrades by no more than 5%.
5. Pooled signed bias is within ±2% at every horizon.
6. The measured p95 latency is inside the budget.

Otherwise the decision is `REJECT`; when fewer than 30 paired observations exist it is
`INCONCLUSIVE`. Every decision writes a signed record with both artifact hashes, the dataset and
feature-contract hashes, the git SHA, the statistics and the failed checks.

## Consequences

* The gate can say no, and the demo exercises that path.
* Changing any metric definition requires a `feat!:` commit and a note here.
* The bias bound is the strictest condition. It is measured on pooled per-horizon observations;
  six folds of twelve SKUs still leave sampling error, so a model that is only marginally better
  will be rejected. That is intentional: the default is to keep the incumbent.
* Retraining may open a PR but never moves the champion pointer automatically.
