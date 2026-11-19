# ModelGate - one entry point for the whole loop.
# Recipes are deliberately portable: no POSIX-only shell builtins.

PY := uv run
DEMO ?= 0

.PHONY: setup ci lint types test data train evaluate promote serve smoke monitor-demo demo report clean

setup:
	uv sync --extra dev --python 3.12

lint:
	$(PY) ruff check src tests

types:
	$(PY) mypy --strict src/modelgate

test:
	$(PY) pytest --cov=modelgate --cov-report=term-missing --cov-fail-under=85

ci: lint types test

data:
	$(PY) python -m modelgate.data.generator
	$(PY) python -m modelgate.data.ingest

train:
	$(PY) python -m modelgate.train.pipeline --register

evaluate:
	$(PY) python -m modelgate.backtest.rolling_origin --report

promote:
	$(PY) python -m modelgate.promote --demo $(DEMO)

serve:
	$(PY) python -m modelgate.serve.app

smoke:
	$(PY) python -m modelgate.serve.smoke

monitor-demo:
	$(PY) python -m modelgate.monitor.demo

demo:
	$(PY) python -m modelgate.demo

report:
	$(PY) python -m modelgate.report

clean:
	$(PY) python -m modelgate.clean
