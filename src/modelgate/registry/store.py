"""File-backed model registry with immutable artifacts and a champion pointer.

Layout::

    models/registry/
      index.json                 # ordered versions, champion, rollback history
      versions/<id>/model.joblib # byte-identical copy, integrity checked on read
      versions/<id>/manifest.json

Every read of a champion artifact verifies its SHA-256 against the manifest; serving refuses to
start otherwise. ``rollback`` pops the previous champion from an explicit history stack.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from modelgate.config import MODELS_DIR
from modelgate.hashing import sha256_file, sha256_json
from modelgate.provenance import git_sha

SCHEMA = "modelgate.registry/1"


class RegistryError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class Registry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else MODELS_DIR / "registry"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        if not self.index_path.exists():
            self._write_index({"schema": SCHEMA, "order": [], "versions": {}, "champion": None,
                               "history": []})

    def _read_index(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.index_path.read_text(encoding="utf-8")))

    def _write_index(self, index: dict[str, Any]) -> None:
        self.index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")

    @property
    def champion_id(self) -> str | None:
        return self._read_index().get("champion")

    def versions(self) -> list[dict[str, Any]]:
        index = self._read_index()
        return [index["versions"][version_id] for version_id in index["order"]]

    def get(self, version_id: str) -> dict[str, Any]:
        index = self._read_index()
        if version_id not in index["versions"]:
            raise RegistryError(f"unknown version {version_id!r}")
        return cast(dict[str, Any], index["versions"][version_id])

    def path_for(self, version_id: str) -> Path:
        return self.root / "versions" / version_id

    def register(
        self,
        artifact_dir: Path,
        metrics: dict[str, float] | None = None,
        run_id: str | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        source = Path(artifact_dir)
        artifact = source / "model.joblib"
        source_manifest_path = source / "manifest.json"
        if not artifact.exists() or not source_manifest_path.exists():
            raise RegistryError(f"artifact directory {source} is incomplete")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        digest = sha256_file(artifact)
        declared = source_manifest.get("artifact_sha256", digest)
        if declared != digest:
            raise RegistryError("source artifact does not match its manifest")

        stamp = created_at or _utc_now()
        version_id = "v" + sha256_json(
            {"artifact_sha256": digest, "created_at": stamp, "metrics": metrics or {}}
        )[:12]
        index = self._read_index()
        if version_id in index["versions"]:
            return cast(dict[str, Any], index["versions"][version_id])

        target = self.path_for(version_id)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(artifact, target / "model.joblib")
        manifest = {
            **source_manifest,
            "version_id": version_id,
            "artifact_sha256": digest,
            "created_at": stamp,
            "git_sha": source_manifest.get("git_sha", git_sha()),
            "run_id": run_id,
            "metrics": metrics or {},
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        index["versions"][version_id] = manifest
        index["order"].append(version_id)
        if index["champion"] is None:
            index["champion"] = version_id
        self._write_index(index)
        return manifest

    def verify(self, version_id: str) -> bool:
        manifest = self.get(version_id)
        artifact = self.path_for(version_id) / "model.joblib"
        if not artifact.exists():
            return False
        return bool(sha256_file(artifact) == manifest["artifact_sha256"])

    def promote(self, version_id: str) -> dict[str, Any]:
        manifest = self.get(version_id)
        if not self.verify(version_id):
            raise RegistryError(f"artifact for {version_id} failed integrity check")
        index = self._read_index()
        current = index.get("champion")
        if current == version_id:
            return manifest
        if current is not None:
            index.setdefault("history", []).append(current)
        index["champion"] = version_id
        self._write_index(index)
        return manifest

    def rollback(self) -> dict[str, Any]:
        index = self._read_index()
        history: list[str] = index.get("history", [])
        if not history:
            raise RegistryError("no previous champion to roll back to")
        previous = history.pop()
        if not self.verify(previous):
            raise RegistryError(f"rollback target {previous} failed integrity check")
        index["history"] = history
        index["champion"] = previous
        self._write_index(index)
        return self.get(previous)

    def champion_artifact(self) -> tuple[Path, dict[str, Any]]:
        champion = self.champion_id
        if champion is None:
            raise RegistryError("registry has no champion")
        if not self.verify(champion):
            raise RegistryError(f"champion artifact {champion} failed integrity check")
        return self.path_for(champion) / "model.joblib", self.get(champion)
