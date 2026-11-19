# AGENTS.md

Rules for any coding agent working in this repository.

## Read first

`PLAN.md` is the source of truth for milestones, acceptance commands and commit boundaries. Do not
invent scope; if something is missing, amend `PLAN.md` in the same commit as the change and say why.

## Workflow

1. Take the lowest incomplete milestone. Never skip a gate.
2. Read the full milestone section before coding.
3. Write the listed tests first; confirm they fail for the intended reason.
4. Implement until `make ci` is green.
5. Tick the checkbox in `PLAN.md`, commit, move on.

## Commit rules

- Conventional Commits. One commit per meaningful slice (loader, gate, endpoint, monitor), never a
  single dump at the end.
- Subject states the behaviour; body states why, plus any measured number and the command used.
- Never commit `data/raw/`, `mlruns/`, model binaries, secrets, or `.env`.
- If you change a metric's definition, it is a `feat!:` with a note in `docs/decisions/`.

## Engineering rules

- Leakage first: any new feature must pass the point-in-time suite before anything else uses it.
- Determinism: explicit seeds; same dataset hash + params must reproduce the same metrics.
- The gate must be able to reject. A demo where everything promotes is a bug.
- No network in tests. MLflow uses a local file store; API tests use fixtures.
- Statistical decisions carry an interval and a stated threshold.
- `mypy --strict` and `ruff` clean; coverage ≥ 85% on `src/modelgate`.

## Definition of done

- Tests added, `make ci` green, docs updated, `PLAN.md` ticked.
- Any number quoted in README or a doc comes from `reports/` generated in that commit.
- No silent TODOs: fix, or add to the current milestone in `PLAN.md`.
