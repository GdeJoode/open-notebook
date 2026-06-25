"""R.0 backfill: embed source chunks + populate source-level aggregate vectors.

Background. The embedding layer was *decoupled-and-unrun*: ``SourceProcessor``
created chunks but nothing embedded them (the R.0 forward-fix wires that for new
ingests). Existing sources therefore have ``chunk`` rows but zero
``source_embedding`` rows, so ``fn::vector_search`` and the R.1 retrieval layer
have nothing to search. This script backfills the gap.

Two modes (run chunk embeddings first, then aggregates):

* **default (chunk embeddings)** — iterate sources that have chunks but no
  ``source_embedding`` rows and call ``EmbeddingService.embed_source`` (which
  reads the chunks, batches, and writes ``source_embedding``). ``embed_source``
  is itself idempotent (it deletes existing embeddings first), and we only ever
  select sources that are still missing vectors, so the script is idempotent +
  resumable at the source granularity.

* ``--source-embeddings`` — populate the source-level aggregate ``source.embedding``
  (migration 63) by mean-pooling each source's chunk vectors. No model calls.
  Reusable: the live orchestrator imports ``populate_source_embedding`` /
  ``populate_all_source_embeddings`` to run this as a gated step.

Properties (mirrors ``scripts/backfill_entity_embeddings.py``, the Track-P pattern):

* **Idempotent** — chunk mode selects only sources still missing vectors;
  aggregate mode recomputes from the current chunk vectors (a stable function of
  the data). Re-running is safe.
* **Resumable** — each source is embedded + committed before moving on; a crash
  loses at most the in-flight source, and the next run re-selects the rest.
* **Batched** — ``embed_source`` batches the model calls internally
  (``EmbeddingConfig.batch_size``); this script's ``--limit`` caps sources/run.
* **--dry-run** — reports how many sources / chunks are missing vectors (or, in
  aggregate mode, how many sources would get an aggregate) and writes nothing.

Usage (inside the app environment, e.g. ``uv run --project apps/app-main``)::

    python scripts/backfill_chunk_embeddings.py --dry-run
    python scripts/backfill_chunk_embeddings.py
    python scripts/backfill_chunk_embeddings.py --limit 2
    python scripts/backfill_chunk_embeddings.py --source-embeddings --dry-run
    python scripts/backfill_chunk_embeddings.py --source-embeddings

Args:
    --limit N           Process at most N sources this run (default: all).
    --dry-run           Report the missing set; write nothing.
    --source-embeddings Populate source-level aggregate vectors (mean-pool)
                        instead of embedding chunks.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger
from shared.utils.vectors import mean_pool
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.source import SourceRepository


async def list_sources_missing_chunk_embeddings() -> List[str]:
    """Return source ids that HAVE chunks but NO ``source_embedding`` rows.

    A source qualifies for the chunk backfill only when it has at least one
    ``chunk`` and zero ``source_embedding`` rows — i.e. extracted-but-unembedded.
    Sources with no chunks (nothing to embed) and sources already embedded are
    both excluded, which is what makes the default mode idempotent.
    """
    rows = await execute_query(
        """
        SELECT VALUE id FROM source WHERE
            count(SELECT id FROM chunk WHERE source = $parent.id) > 0
            AND count(SELECT id FROM source_embedding WHERE source = $parent.id) = 0;
        """
    )
    return [str(r) for r in (rows or [])]


async def count_missing_chunks(source_ids: List[str]) -> int:
    """Total chunk rows across ``source_ids`` (the would-be embedding count)."""
    total = 0
    repo = SourceRepository()
    for sid in source_ids:
        chunks = await repo.get_chunks(sid)
        total += len(chunks)
    return total


async def populate_source_embedding(
    source_id: str, source_repo: Optional[SourceRepository] = None
) -> Optional[int]:
    """Mean-pool a source's chunk vectors into ``source.embedding`` (R.0).

    Reusable by the live orchestrator. Reads the source's per-chunk vectors from
    ``source_embedding``, mean-pools them, and writes the aggregate onto
    ``source.embedding`` (migration 63). A source with no chunk vectors gets
    ``NONE`` written (graceful empty), returning ``None``.

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


async def populate_all_source_embeddings(
    limit: Optional[int] = None, dry_run: bool = False
) -> Dict[str, int]:
    """Populate aggregate ``source.embedding`` for every source with chunk vectors.

    Reusable by the live orchestrator (gated). ``--dry-run`` reports how many
    sources HAVE chunk vectors (and would therefore get an aggregate) without
    writing.
    """
    repo = SourceRepository()
    rows = await execute_query(
        "SELECT VALUE id FROM source WHERE "
        "count(SELECT id FROM source_embedding WHERE source = $parent.id) > 0;"
    )
    source_ids = [str(r) for r in (rows or [])]
    if limit is not None:
        source_ids = source_ids[:limit]

    logger.info(
        f"source-embedding populate: {len(source_ids)} source(s) have chunk "
        f"vectors to aggregate"
    )
    if dry_run:
        return {"sources": len(source_ids), "populated": 0, "empty": 0}

    populated = 0
    empty = 0
    dim_seen: Optional[int] = None
    for sid in source_ids:
        dim = await populate_source_embedding(sid, source_repo=repo)
        if dim is None:
            empty += 1
        else:
            populated += 1
            if dim_seen is None:
                dim_seen = dim
    logger.info(
        f"source-embedding populate DONE: populated={populated} empty={empty} "
        f"(aggregate dim={dim_seen})"
    )
    return {"sources": len(source_ids), "populated": populated, "empty": empty}


async def _backfill_chunks(
    limit: Optional[int], dry_run: bool
) -> Dict[str, int]:
    """Embed chunks for sources missing ``source_embedding`` rows."""
    pending = await list_sources_missing_chunk_embeddings()
    if limit is not None:
        pending = pending[:limit]
    total_sources = len(pending)
    total_chunks = await count_missing_chunks(pending)
    logger.info(
        f"backfill_chunk_embeddings: {total_sources} source(s) with chunks but "
        f"no embeddings ({total_chunks} chunk(s) to embed)"
    )
    if dry_run:
        return {
            "sources": total_sources,
            "chunks_missing": total_chunks,
            "embedded_sources": 0,
            "failed": 0,
        }
    if total_sources == 0:
        logger.info("Nothing to backfill — every source with chunks is embedded.")
        return {
            "sources": 0,
            "chunks_missing": 0,
            "embedded_sources": 0,
            "failed": 0,
        }

    # Resolve the SAME embedding model the pipeline uses (mxbai-embed-large,
    # 1024-dim, resolved via ModelManager — never hardcoded). Imported lazily so
    # a --dry-run never needs the app DI container / model layer wired up.
    from app_main.dependencies import get_embedding_service

    embed_svc = await get_embedding_service()

    embedded_sources = 0
    failed = 0
    dim_seen: Optional[int] = None
    repo = embed_svc.source_repo

    for sid in pending:
        try:
            result = await embed_svc.embed_source(sid)
        except Exception as e:  # noqa: BLE001 — isolate one bad source
            logger.error(f"Failed to embed source {sid}: {e}")
            failed += 1
            continue
        embedded_sources += 1
        if dim_seen is None:
            vectors = await repo.get_embedding_vectors(sid)
            if vectors:
                dim_seen = len(vectors[0])
                logger.info(f"Embedding dimension = {dim_seen}")
        logger.info(
            f"Progress: embedded {embedded_sources}/{total_sources} sources "
            f"(source {sid}: {result.embeddings_created} embeddings)"
        )

    remaining = len(await list_sources_missing_chunk_embeddings())
    logger.info(
        f"backfill_chunk_embeddings DONE: embedded_sources={embedded_sources} "
        f"failed={failed} remaining={remaining} (embedding dim={dim_seen})"
    )
    return {
        "sources": total_sources,
        "chunks_missing": total_chunks,
        "embedded_sources": embedded_sources,
        "failed": failed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill source chunk embeddings + aggregate vectors (R.0)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N sources this run (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the missing set; write nothing.",
    )
    parser.add_argument(
        "--source-embeddings",
        action="store_true",
        help="Populate source-level aggregate vectors (mean-pool) instead of "
        "embedding chunks.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.source_embeddings:
        asyncio.run(
            populate_all_source_embeddings(limit=args.limit, dry_run=args.dry_run)
        )
    else:
        asyncio.run(_backfill_chunks(limit=args.limit, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
