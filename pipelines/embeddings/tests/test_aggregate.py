"""Unit tests for the source-level aggregate embedding (PL.1, DB-free).

Cover ``populate_source_embedding`` (mean-pool + graceful empty) directly and
its wiring into ``EmbeddingService.embed_source`` (the live embed step now
writes ``source.embedding``). A fake repo records the aggregate writes so we can
assert the value without a container; the ``@requires_docker`` roundtrip in
``apps/app-main/tests/test_backfill_chunk_embeddings_db.py`` validates the real
SurrealDB write.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from embeddings.aggregate import populate_source_embedding
from embeddings.config import EmbeddingConfig
from embeddings.service import EmbeddingService
from shared.models import Chunk, SourceEmbedding


class FakeAggregateRepo:
    """In-memory repo recording per-chunk vectors + aggregate writes.

    ``embed_source`` exercises the live path: ``delete_embeddings`` clears, the
    model produces per-chunk vectors that ``add_embedding`` records, and the new
    aggregate step reads them back via ``get_embedding_vectors`` and writes the
    mean-pool via ``set_aggregate_embedding``.
    """

    def __init__(self, chunks: Optional[List[Chunk]] = None) -> None:
        self.config = None
        self._chunks = chunks or []
        self._vectors: List[List[float]] = []
        self.aggregate_writes: List[Optional[List[float]]] = []
        self._source: Optional[Any] = None

    async def delete_embeddings(self, source_id: str) -> int:
        self._vectors = []
        return 0

    async def get_chunks(self, source_id: str) -> List[Chunk]:
        return list(self._chunks)

    async def get(self, source_id: str) -> Any:
        return self._source

    async def add_embedding(self, **kwargs: Any) -> SourceEmbedding:
        self._vectors.append(kwargs.get("embedding"))
        return SourceEmbedding(
            source=kwargs.get("source_id"),
            content=kwargs.get("content"),
            order=kwargs.get("order", 0),
            embedding=kwargs.get("embedding"),
            chunk=kwargs.get("chunk_id"),
        )

    async def get_embedding_vectors(self, source_id: str) -> List[List[float]]:
        return [v for v in self._vectors if v]

    async def set_aggregate_embedding(
        self, source_id: str, embedding: Optional[List[float]]
    ) -> bool:
        self.aggregate_writes.append(embedding)
        return True


def _chunk(text: str, order: int) -> Chunk:
    return Chunk(text=text, order=order, element_type="paragraph", physical_page=1)


class _StoreRepo:
    """Minimal repo holding fixed vectors for direct populate_* tests."""

    def __init__(self, vectors: List[List[float]]) -> None:
        self._vectors = vectors
        self.aggregate_writes: Dict[str, Optional[List[float]]] = {}

    async def get_embedding_vectors(self, source_id: str) -> List[List[float]]:
        return list(self._vectors)

    async def set_aggregate_embedding(
        self, source_id: str, embedding: Optional[List[float]]
    ) -> bool:
        self.aggregate_writes[source_id] = embedding
        return True


@pytest.mark.asyncio
async def test_populate_source_embedding_mean_pools() -> None:
    repo = _StoreRepo([[1.0, 3.0], [3.0, 5.0]])
    dim = await populate_source_embedding("source:a", source_repo=repo)
    assert dim == 2
    assert repo.aggregate_writes["source:a"] == [2.0, 4.0]


@pytest.mark.asyncio
async def test_populate_source_embedding_empty_writes_none() -> None:
    repo = _StoreRepo([])
    dim = await populate_source_embedding("source:a", source_repo=repo)
    assert dim is None
    # NONE is written (graceful empty), not skipped.
    assert "source:a" in repo.aggregate_writes
    assert repo.aggregate_writes["source:a"] is None


@pytest.mark.asyncio
async def test_embed_source_writes_aggregate_mean_pool() -> None:
    """AC1: the live embed step writes ``source.embedding`` = mean_pool(chunks)."""
    from unittest.mock import AsyncMock

    chunks = [_chunk("alpha", 0), _chunk("beta", 1)]
    repo = FakeAggregateRepo(chunks)
    model = AsyncMock()
    # Distinct per-chunk vectors so the mean is non-trivial.
    model.aembed = AsyncMock(side_effect=lambda texts: [[1.0, 3.0], [3.0, 5.0]])

    service = EmbeddingService(repo, model, EmbeddingConfig(batch_size=10))
    result = await service.embed_source("source:agg")

    assert result.embeddings_created == 2
    # Exactly one aggregate write, equal to the mean-pool of the two vectors.
    assert repo.aggregate_writes == [[2.0, 4.0]]


@pytest.mark.asyncio
async def test_embed_source_recomputes_aggregate_on_reembed() -> None:
    """AC2: re-embedding recomputes the aggregate (no dup/stale)."""
    from unittest.mock import AsyncMock

    chunks = [_chunk("alpha", 0)]
    repo = FakeAggregateRepo(chunks)
    model = AsyncMock()
    model.aembed = AsyncMock(side_effect=lambda texts: [[2.0, 2.0]])

    service = EmbeddingService(repo, model, EmbeddingConfig(batch_size=10))
    await service.embed_source("source:re")
    assert repo.aggregate_writes[-1] == [2.0, 2.0]

    # Re-embed with a different model output -> aggregate recomputed afresh.
    model.aembed = AsyncMock(side_effect=lambda texts: [[6.0, 8.0]])
    await service.embed_source("source:re")
    assert repo.aggregate_writes[-1] == [6.0, 8.0]


@pytest.mark.asyncio
async def test_embed_source_no_chunks_no_text_clears_aggregate() -> None:
    """AC2 (graceful): a source with no chunks and no text writes NONE, no crash."""
    from unittest.mock import AsyncMock

    repo = FakeAggregateRepo(chunks=[])  # no structural chunks
    repo._source = None  # get() -> None -> no full_text fallback
    model = AsyncMock()

    service = EmbeddingService(repo, model, EmbeddingConfig(batch_size=10))
    result = await service.embed_source("source:empty")

    assert result.embeddings_created == 0
    # Aggregate cleared to NONE (no vectors to pool), no exception raised.
    assert repo.aggregate_writes == [None]


@pytest.mark.asyncio
async def test_populate_source_embedding_clears_stale_populated_aggregate() -> None:
    """PL.2 fold-in of the PL.1 minor: a previously-POPULATED aggregate is
    cleared to NONE when the source loses its chunk vectors.

    Seed a non-null ``source.embedding`` (a real prior aggregate), then strip the
    chunk vectors and re-run the aggregate step: the stale populated value MUST
    be overwritten with NONE — not left dangling. ``populate_source_embedding``
    always writes (NONE on empty), so a re-embed of a now-empty source never
    leaves a stale vector behind that retrieval would treat as live.
    """
    repo = _StoreRepo([])  # no chunk vectors now
    # Seed a pre-existing populated aggregate (the "stale" value).
    repo.aggregate_writes["source:stale"] = [9.0, 9.0, 9.0]

    dim = await populate_source_embedding("source:stale", source_repo=repo)

    assert dim is None
    # The stale populated aggregate has been overwritten to NONE.
    assert repo.aggregate_writes["source:stale"] is None
