from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelgate.hashing import sha256_file
from modelgate.registry.store import Registry, RegistryError


def make_artifact(root: Path, name: str, payload: bytes, feature_hash: str = "f" * 64) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    artifact = directory / "model.joblib"
    artifact.write_bytes(payload)
    manifest = {
        "artifact_sha256": sha256_file(artifact),
        "feature_version_hash": feature_hash,
        "dataset_hash": "d" * 64,
        "git_sha": "abc123",
        "config": {"seed": 42},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def test_register_promote_and_rollback(tmp_path: Path) -> None:
    first = make_artifact(tmp_path, "first", b"first-model")
    second = make_artifact(tmp_path, "second", b"second-model")
    registry = Registry(tmp_path / "registry")

    one = registry.register(first, metrics={"wape": 0.2}, created_at="2026-01-01T00:00:00+00:00")
    assert registry.champion_id == one["version_id"]
    two = registry.register(second, metrics={"wape": 0.18}, created_at="2026-01-02T00:00:00+00:00")
    assert two["version_id"] != one["version_id"]
    assert len(registry.versions()) == 2

    registry.promote(two["version_id"])
    assert registry.champion_id == two["version_id"]
    assert registry.verify(two["version_id"])

    restored = registry.rollback()
    assert restored["version_id"] == one["version_id"]
    assert registry.champion_id == one["version_id"]
    with pytest.raises(RegistryError, match="no previous champion"):
        registry.rollback()


def test_tampered_artifact_is_rejected(tmp_path: Path) -> None:
    source = make_artifact(tmp_path, "model", b"original")
    registry = Registry(tmp_path / "registry")
    version = registry.register(source, metrics={}, created_at="2026-01-01T00:00:00+00:00")
    registry.path_for(version["version_id"]).joinpath("model.joblib").write_bytes(b"tampered")
    assert not registry.verify(version["version_id"])
    with pytest.raises(RegistryError, match="integrity"):
        registry.champion_artifact()
    with pytest.raises(RegistryError, match="integrity"):
        registry.promote(version["version_id"])


def test_registry_guards_bad_inputs(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry")
    with pytest.raises(RegistryError, match="incomplete"):
        registry.register(tmp_path / "missing")
    with pytest.raises(RegistryError, match="unknown version"):
        registry.get("nope")
    with pytest.raises(RegistryError, match="no champion"):
        Registry(tmp_path / "empty").champion_artifact()

    bad = make_artifact(tmp_path, "bad", b"payload")
    manifest = json.loads((bad / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifact_sha256"] = "0" * 64
    (bad / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RegistryError, match="does not match"):
        registry.register(bad)


def test_register_is_idempotent_for_same_content(tmp_path: Path) -> None:
    source = make_artifact(tmp_path, "model", b"same")
    registry = Registry(tmp_path / "registry")
    first = registry.register(
        source, metrics={"wape": 0.1}, created_at="2026-01-01T00:00:00+00:00"
    )
    second = registry.register(
        source, metrics={"wape": 0.1}, created_at="2026-01-01T00:00:00+00:00"
    )
    assert first["version_id"] == second["version_id"]
    assert len(registry.versions()) == 1
