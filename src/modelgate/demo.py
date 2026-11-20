"""End-to-end demo: data -> train -> gate reject -> gate promote -> serve -> drift.

Runs offline on a clean checkout. The two gate scenarios are the same controlled error
distributions used by the unit tests; they exist to show that the gate can both say no and say yes.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from modelgate.config import RAW_DIR, REPORTS_DIR
from modelgate.data.generator import generate_demand, write_raw
from modelgate.data.ingest import ingest, read_demand
from modelgate.monitor import demo as monitor_demo
from modelgate.promote.demo import run_demo
from modelgate.promote.gate import Decision
from modelgate.registry.store import Registry
from modelgate.serve.app import create_app
from modelgate.train.pipeline import TrainingConfig, train_and_log


def _ensure_data() -> None:
    if not (RAW_DIR / "demand.parquet").exists():
        print("step 0: generating demand data")
        write_raw(generate_demand())
        ingest()


def _train() -> tuple[str, dict[str, Any]]:
    print("step 1: train and register a champion")
    frame = read_demand()
    run = train_and_log(frame, TrainingConfig(n_estimators=60))
    registry = Registry()
    manifest = registry.register(
        run.artifact_dir, metrics=run.train_metrics, run_id=run.run_id
    )
    registry.promote(manifest["version_id"])
    print(
        f"  champion {manifest['version_id']} WAPE={run.train_metrics['train_wape']:.4f} "
        f"feature={run.model.feature_version_hash[:12]}..."
    )
    return manifest["version_id"], {"frame": frame}


def main() -> int:
    _ensure_data()
    champion, context = _train()
    frame = context["frame"]

    print("step 2: gate rejects a challenger that wins on mean and loses the tail")
    rejected, rejected_path = run_demo("tail")
    if rejected.decision is not Decision.REJECT:
        raise SystemExit("demo invariant broken: tail scenario was not rejected")
    assert rejected_path is not None

    print("step 3: gate promotes a clearly better, tail-safe challenger")
    promoted, promoted_path = run_demo("promote")
    if promoted.decision is not Decision.PROMOTE:
        raise SystemExit("demo invariant broken: promotion scenario was rejected")
    assert promoted_path is not None

    print("step 4: serve the pinned champion")
    with TestClient(create_app(Registry(), frame)) as client:
        version = client.get("/version").json()
        forecast = client.post("/forecast", json={"sku": "SKU-000", "horizon": 7}).json()
        print(
            f"  /version {version['model_version']} "
            f"feature={version['feature_version_hash'][:12]}..."
        )
        print(
            f"  /forecast SKU-000 h=7 first={forecast['points'][0]['date']} "
            f"mean={forecast['points'][0]['mean']:.1f}"
        )

    print("step 5: inject drift and prepare a reviewed retrain PR")
    monitor_demo.main()

    summary = {
        "champion": champion,
        "rejected_record": rejected_path.name,
        "promoted_record": promoted_path.name,
        "rejected_reasons": list(rejected.reasons),
        "promoted_reasons": list(promoted.reasons),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "demo.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("demo complete")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
