from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from memory import (
    CachingEmbeddingModel,
    HashEmbeddingModel,
    SentenceTransformerEmbedder,
)


class ReusableEmbeddingFactory:
    """Load model weights once and return a fresh case-local cache per call."""

    def __init__(
        self,
        backend: str,
        model_name: str,
        *,
        hash_dim: int = 384,
        backend_factory: Callable[[str], object] | None = None,
    ):
        if backend not in {"hash", "sentence-transformers"}:
            raise ValueError(f"unsupported embedding backend: {backend}")
        self.backend = backend
        self.model_name = model_name
        self.hash_dim = hash_dim
        self.backend_factory = backend_factory or SentenceTransformerEmbedder
        self._shared_backend = None

    def _load_backend(self):
        if self._shared_backend is None:
            self._shared_backend = (
                HashEmbeddingModel(dim=self.hash_dim)
                if self.backend == "hash"
                else self.backend_factory(self.model_name)
            )
        return self._shared_backend

    def __call__(self):
        return CachingEmbeddingModel(self._load_backend())


def run_row_key(row: dict) -> tuple[str, str, int]:
    return str(row["case_id"]), str(row["agent"]), int(row.get("repeat", 1))


def require_frozen_benchmark(path: str | Path) -> dict:
    """Verify that a split is covered by a human-approved frozen manifest."""
    benchmark_path = Path(path).resolve()
    manifest_path = benchmark_path.parent / "frozen_manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            f"formal run requires a frozen benchmark manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        raise ValueError("benchmark manifest is not frozen")
    split = benchmark_path.stem
    split_data = manifest.get("splits", {}).get(split)
    if not split_data:
        raise ValueError(f"benchmark split is not present in frozen manifest: {split}")
    if split_data.get("sha256") != _sha256_file(benchmark_path):
        raise ValueError(f"benchmark hash no longer matches frozen manifest: {split}")
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class JsonlRunCheckpoint:
    """Append-only per-row checkpoint with configuration compatibility checks."""

    def __init__(self, path: str | Path, configuration: dict):
        self.path = Path(path)
        self.meta_path = self.path.with_suffix(self.path.suffix + ".meta.json")
        canonical = json.dumps(
            configuration, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        self.configuration = configuration
        self.configuration_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def prepare(self, *, resume: bool, retry_failed: bool = True) -> list[dict]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume:
            if not self.path.exists() or not self.meta_path.exists():
                raise ValueError(f"resume checkpoint is incomplete: {self.path}")
            metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if metadata.get("configuration_sha256") != self.configuration_sha256:
                raise ValueError("checkpoint configuration does not match this run")
            rows = self._load_rows()
            if retry_failed:
                rows = [row for row in rows if row.get("status") != "failed"]
                self._rewrite(rows)
            return rows

        self._rewrite([])
        self.meta_path.write_text(
            json.dumps(
                {
                    "configuration_sha256": self.configuration_sha256,
                    "configuration": self.configuration,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return []

    def _load_rows(self) -> list[dict]:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        rows: list[dict] = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise ValueError(f"corrupt checkpoint row {index + 1}")
        keys = [run_row_key(row) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("checkpoint contains duplicate run keys")
        return rows

    def _rewrite(self, rows: list[dict]) -> None:
        with self.path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def append(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
