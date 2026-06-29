"""Source-level aggregate embedding (mean-pool of chunk vectors).

The retrieval/analytical layer (R.1/R.2 "Verwante", ``find_related_by_embedding``,
the hybrid/KG retrieval, contradiction candidate generation) reads the
source-level aggregate ``source.embedding`` — a mean-pool of the source's
per-chunk vectors (migration 63). The per-chunk vectors alone are invisible to
all of those: a source needs the aggregate to be related-retrievable.

This helper is the single, reusable definition of "compute + write the
aggregate". It is called by ``EmbeddingService.embed_source`` right after the
per-chunk vectors are written (so every fresh embed produces the aggregate) and
re-exported by ``scripts/backfill_chunk_embeddings.py`` for the one-time
backfill — neither imports the logic from a script.

Kept dependency-light: it only touches the ``SourceRepository`` (vector read +
aggregate write) and the pure ``mean_pool`` helper, so it never pulls the model
layer in.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger
from shared.utils.vectors import mean_pool
from surrealdb_service.repositories.source import SourceRepository


async def populate_source_embedding(
    source_id: str, source_repo: Optional[SourceRepository] = None
) -> Optional[int]:
    """Mean-pool a source's chunk vectors into ``source.embedding`` (R.0).

    Reads the source's per-chunk vectors from ``source_embedding``, mean-pools
    them, and writes the aggregate onto ``source.embedding`` (migration 63). The
    aggregate dimension is whatever the stored chunk vectors are (the configured
    model's dim, 1024 today) — never hardcoded.

    Idempotent: it recomputes purely from the current chunk vectors, so a
    re-embed (which rewrites the per-chunk rows) yields a fresh, correct
    aggregate with no duplication. Graceful: a source with no/empty chunk
    vectors writes ``NONE`` (the honest "no aggregate" state the
    ``option<array<float>>`` field stores), returning ``None`` rather than
    crashing.

    Returns the aggregate vector's dimension on success, or ``None`` when there
    was nothing to pool.
    """
    repo = source_repo or SourceRepository()
    vectors = await repo.get_embedding_vectors(source_id)
    aggregate = mean_pool(vectors)
    await repo.set_aggregate_embedding(source_id, aggregate)
    if aggregate is None:
        logger.info(f"source {source_id}: no chunk vectors -> aggregate=NONE")
        return None
    logger.info(
        f"source {source_id}: aggregate embedding set "
        f"(pooled {len(vectors)} chunk vectors, dim={len(aggregate)})"
    )
    return len(aggregate)
