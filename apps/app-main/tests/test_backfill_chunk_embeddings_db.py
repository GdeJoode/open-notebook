"""Container tests for the R.0 backfill against a real SurrealDB (Track R.0).

These exercise the actual SurrealQL the script + repo run — the discovery query
(sources with chunks but no embeddings), the per-chunk vector read, the
aggregate ``source.embedding`` write, and the mean-pool populate — which the
DB-free fakes in ``test_backfill_chunk_embeddings.py`` cannot validate. Gated on
``requires_docker``; skipped cleanly without Docker.
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.source import SourceRepository

_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "backfill_chunk_embeddings.py"
)
_spec = importlib.util.spec_from_file_location("backfill_chunk_embeddings", _SCRIPT)
backfill = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(backfill)


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


async def _make_source_with_chunks(
    cfg: SurrealDBConfig, n_chunks: int
) -> str:
    title = _uid("r0-src")
    created = await execute_query(
        "CREATE source SET title = $title, full_text = 'body';",
        {"title": title},
        config=cfg,
    )
    sid = str(created[0]["id"])
    for order in range(n_chunks):
        await execute_query(
            "CREATE chunk SET source = $src, text = $text, order = $order, "
            "physical_page = 0, element_type = 'paragraph', positions = [], "
            "metadata = {};",
            {
                "src": ensure_record_id(sid),
                "text": f"chunk {order} of {title}",
                "order": order,
            },
            config=cfg,
        )
    return sid


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_discovery_query_finds_unembedded_sources(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """``list_sources_missing_chunk_embeddings`` returns chunked-but-unembedded.

    A source with chunks + no ``source_embedding`` rows appears; once it has an
    embedding row it disappears (idempotency at the discovery layer); a source
    with no chunks never appears.
    """
    import surrealdb_service.connection as conn

    # The script's helpers call the bare global ``execute_query()`` (no config),
    # which resolves the global pool. Bind that pool to the test container so the
    # script's own SurrealQL runs against the fixture DB.
    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        repo = SourceRepository(config=live_surrealdb)
        with_chunks = await _make_source_with_chunks(live_surrealdb, 3)
        no_chunks_created = await execute_query(
            "CREATE source SET title = $t, full_text = 'x';",
            {"t": _uid("r0-empty")},
            config=live_surrealdb,
        )
        no_chunks = str(no_chunks_created[0]["id"])

        missing = await backfill.list_sources_missing_chunk_embeddings()
        assert with_chunks in missing
        assert no_chunks not in missing  # no chunks -> nothing to embed

        # Add an embedding row -> source drops out of the missing set.
        await repo.add_embedding(
            source_id=with_chunks,
            content="chunk 0",
            order=0,
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        missing_after = await backfill.list_sources_missing_chunk_embeddings()
        assert with_chunks not in missing_after
    finally:
        conn._pool = orig


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_repo_vectors_and_aggregate_roundtrip(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """``get_embedding_vectors`` + ``set_aggregate_embedding`` roundtrip on real DB."""
    repo = SourceRepository(config=live_surrealdb)
    sid = await _make_source_with_chunks(live_surrealdb, 2)

    await repo.add_embedding(sid, "c0", 0, [1.0, 3.0, 5.0])
    await repo.add_embedding(sid, "c1", 1, [3.0, 5.0, 7.0])

    vectors = await repo.get_embedding_vectors(sid)
    assert len(vectors) == 2
    assert all(len(v) == 3 for v in vectors)

    # Aggregate write (mean-pool) persists onto source.embedding.
    from shared.utils.vectors import mean_pool

    pooled = mean_pool(vectors)
    assert pooled == pytest.approx([2.0, 4.0, 6.0])
    ok = await repo.set_aggregate_embedding(sid, pooled)
    assert ok is True

    src = await repo.get(sid)
    assert src is not None
    assert src.embedding == pytest.approx([2.0, 4.0, 6.0])

    # NONE path: a source with no chunk vectors writes NONE gracefully.
    empty_sid = await _make_source_with_chunks(live_surrealdb, 0)
    assert await repo.get_embedding_vectors(empty_sid) == []
    assert await repo.set_aggregate_embedding(empty_sid, None) is True
    empty_src = await repo.get(empty_sid)
    assert empty_src is not None
    assert empty_src.embedding is None


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_populate_all_source_embeddings_db(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """``populate_all_source_embeddings`` aggregates every embedded source."""
    import surrealdb_service.connection as conn

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        repo = SourceRepository(config=live_surrealdb)
        sid = await _make_source_with_chunks(live_surrealdb, 2)
        await repo.add_embedding(sid, "c0", 0, [2.0, 2.0])
        await repo.add_embedding(sid, "c1", 1, [4.0, 6.0])

        # Dry-run reports the candidate count, writes nothing.
        dry = await backfill.populate_all_source_embeddings(dry_run=True)
        assert dry["sources"] >= 1
        assert dry["populated"] == 0
        assert (await repo.get(sid)).embedding is None

        # Real run mean-pools the chunk vectors onto source.embedding.
        summary = await backfill.populate_all_source_embeddings(dry_run=False)
        assert summary["populated"] >= 1
        src = await repo.get(sid)
        assert src.embedding == pytest.approx([3.0, 4.0])
    finally:
        conn._pool = orig
