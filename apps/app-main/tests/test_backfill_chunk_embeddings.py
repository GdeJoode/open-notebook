"""Unit tests for the R.0 chunk-embedding backfill script (DB-free).

Fakes stand in for the embedding service / source repo / discovery query so
these run without Docker or a configured model. They cover the chunk-backfill
loop (idempotent dry-run, real embed, dimension logging) and the source-level
aggregate populate (mean-pool + graceful empty). Container-level coverage that
exercises the real SurrealDB queries lives in
``apps/app-main/tests/test_backfill_chunk_embeddings_db.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

# Load the script module by path (scripts/ is not an importable package).
_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "backfill_chunk_embeddings.py"
)
_spec = importlib.util.spec_from_file_location("backfill_chunk_embeddings", _SCRIPT)
backfill = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(backfill)


class FakeEmbedResult:
    def __init__(self, n: int) -> None:
        self.embeddings_created = n


class FakeRepo:
    """In-memory: source_id -> {chunks: [text], vectors: [[float]]}."""

    def __init__(self, store: Dict[str, Dict[str, Any]]) -> None:
        self._store = store
        self.aggregate_writes: Dict[str, Optional[List[float]]] = {}

    async def get_chunks(self, source_id: str) -> List[Any]:
        return list(self._store.get(source_id, {}).get("chunks", []))

    async def get_embedding_count(self, source_id: str) -> int:
        return len(self._store.get(source_id, {}).get("vectors", []))

    async def get_embedding_vectors(self, source_id: str) -> List[List[float]]:
        return list(self._store.get(source_id, {}).get("vectors", []))

    async def set_aggregate_embedding(
        self, source_id: str, embedding: Optional[List[float]]
    ) -> bool:
        self.aggregate_writes[source_id] = embedding
        return True


class FakeEmbedSvc:
    """Embeds a source by populating its ``vectors`` from ``chunks`` (dim=4)."""

    def __init__(self, repo: FakeRepo, dim: int = 4) -> None:
        self.source_repo = repo
        self.dim = dim

    async def embed_source(self, source_id: str) -> FakeEmbedResult:
        entry = self.source_repo._store[source_id]
        entry["vectors"] = [[1.0] * self.dim for _ in entry["chunks"]]
        return FakeEmbedResult(len(entry["chunks"]))


@pytest.mark.asyncio
async def test_chunk_dry_run_reports_missing_and_writes_nothing() -> None:
    repo = FakeRepo(
        {
            "source:a": {"chunks": ["x", "y"], "vectors": []},
            "source:b": {"chunks": ["z"], "vectors": []},
        }
    )

    async def fake_missing() -> List[str]:
        return ["source:a", "source:b"]

    async def no_svc():
        raise AssertionError("dry-run must not resolve the embedding model")

    with patch.object(backfill, "SourceRepository", return_value=repo), patch.object(
        backfill, "list_sources_missing_chunk_embeddings", new=fake_missing
    ), patch("app_main.dependencies.get_embedding_service", new=no_svc):
        summary = await backfill._backfill_chunks(limit=None, dry_run=True)

    assert summary["sources"] == 2
    assert summary["chunks_missing"] == 3
    assert summary["embedded_sources"] == 0


@pytest.mark.asyncio
async def test_chunk_backfill_embeds_and_is_idempotent() -> None:
    store = {
        "source:a": {"chunks": ["x", "y"], "vectors": []},
        "source:b": {"chunks": ["z"], "vectors": []},
    }
    repo = FakeRepo(store)
    svc = FakeEmbedSvc(repo, dim=4)

    # Discovery returns sources still missing vectors (recomputed each call).
    async def fake_missing() -> List[str]:
        return [sid for sid, e in store.items() if not e["vectors"]]

    async def fake_svc_factory():
        return svc

    with patch.object(backfill, "SourceRepository", return_value=repo), patch.object(
        backfill, "list_sources_missing_chunk_embeddings", new=fake_missing
    ), patch("app_main.dependencies.get_embedding_service", new=fake_svc_factory):
        summary = await backfill._backfill_chunks(limit=None, dry_run=False)
        assert summary["embedded_sources"] == 2
        assert summary["failed"] == 0

        # Re-run: everything now has vectors -> nothing missing (idempotent).
        rerun = await backfill._backfill_chunks(limit=None, dry_run=False)
        assert rerun["sources"] == 0
        assert rerun["embedded_sources"] == 0

    # Every source got 4-dim vectors.
    assert all(len(v) == 4 for v in store["source:a"]["vectors"])
    assert len(store["source:b"]["vectors"]) == 1


@pytest.mark.asyncio
async def test_chunk_limit_caps_one_run() -> None:
    store = {f"source:{i}": {"chunks": ["c"], "vectors": []} for i in range(5)}
    repo = FakeRepo(store)
    svc = FakeEmbedSvc(repo, dim=4)

    async def fake_missing() -> List[str]:
        return [sid for sid, e in store.items() if not e["vectors"]]

    async def fake_svc_factory():
        return svc

    with patch.object(backfill, "SourceRepository", return_value=repo), patch.object(
        backfill, "list_sources_missing_chunk_embeddings", new=fake_missing
    ), patch("app_main.dependencies.get_embedding_service", new=fake_svc_factory):
        summary = await backfill._backfill_chunks(limit=2, dry_run=False)

    assert summary["embedded_sources"] == 2
    # 3 still missing -> a follow-up run finishes them (resumable).
    assert len([e for e in store.values() if not e["vectors"]]) == 3


@pytest.mark.asyncio
async def test_populate_source_embedding_mean_pools() -> None:
    repo = FakeRepo(
        {"source:a": {"chunks": ["x"], "vectors": [[1.0, 3.0], [3.0, 5.0]]}}
    )
    dim = await backfill.populate_source_embedding("source:a", source_repo=repo)
    assert dim == 2
    assert repo.aggregate_writes["source:a"] == [2.0, 4.0]


@pytest.mark.asyncio
async def test_populate_source_embedding_empty_writes_none() -> None:
    repo = FakeRepo({"source:a": {"chunks": [], "vectors": []}})
    dim = await backfill.populate_source_embedding("source:a", source_repo=repo)
    assert dim is None
    # NONE is written (graceful empty), not skipped.
    assert "source:a" in repo.aggregate_writes
    assert repo.aggregate_writes["source:a"] is None
