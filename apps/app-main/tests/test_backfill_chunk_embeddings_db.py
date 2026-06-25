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

from unittest.mock import patch

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


async def _add_chunk(
    cfg: SurrealDBConfig,
    source_id: str,
    text,
    order: int,
    element_type: str = "paragraph",
) -> None:
    """Create a single chunk (``text`` may be ``""``/whitespace/NONE)."""
    if text is None:
        await execute_query(
            "CREATE chunk SET source = $src, text = NONE, order = $order, "
            "physical_page = 0, element_type = $et, positions = [], metadata = {};",
            {"src": ensure_record_id(source_id), "order": order, "et": element_type},
            config=cfg,
        )
    else:
        await execute_query(
            "CREATE chunk SET source = $src, text = $text, order = $order, "
            "physical_page = 0, element_type = $et, positions = [], metadata = {};",
            {
                "src": ensure_record_id(source_id),
                "text": text,
                "order": order,
                "et": element_type,
            },
            config=cfg,
        )


class _FakeModel:
    """Local-only fake embedding model (deterministic 8-dim vector per text)."""

    async def aembed(self, texts):
        return [[float(len(t) % 7)] * 8 for t in texts]


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_empty_chunks_skipped_and_not_eternally_flagged(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """R.0d AC1+AC2: empty/whitespace chunks are skipped AND don't re-flag.

    Builds the live failure shape: 3 normal chunks + 1 empty-text chunk
    (``element_type='table'``, the staging case) + 1 whitespace-only chunk.

    * AC1 — ``embed_source`` writes vectors for the 3 normal chunks and NO
      ``source_embedding`` row for the 2 empties; it does not raise (old code
      raised ``Text cannot be empty``).
    * AC2 — the eternal-incomplete guard: detection does NOT flag the source
      even though ``chunk_count (5) > source_embedding_count (3)``, because the
      difference is exactly the skipped empties.
    """
    import surrealdb_service.connection as conn
    from embeddings.service import EmbeddingService

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        repo = SourceRepository(config=live_surrealdb)
        created = await execute_query(
            "CREATE source SET title = $t, full_text = 'body';",
            {"t": _uid("r0d-mixed")},
            config=live_surrealdb,
        )
        sid = str(created[0]["id"])
        await _add_chunk(live_surrealdb, sid, "normal chunk zero", 0)
        await _add_chunk(live_surrealdb, sid, "normal chunk one", 1)
        await _add_chunk(live_surrealdb, sid, "", 2, element_type="table")
        await _add_chunk(live_surrealdb, sid, "   \n\t ", 3)
        await _add_chunk(live_surrealdb, sid, "normal chunk two", 4)

        # Pre-embed: source is flagged (3 non-empty chunks, 0 embeddings).
        assert sid in await backfill.list_sources_missing_chunk_embeddings()
        assert await backfill.count_missing_chunks([sid]) == 3

        svc = EmbeddingService(source_repo=repo, embedding_model=_FakeModel())
        result = await svc.embed_source(sid)  # must not raise

        # AC1: exactly 3 rows (empties skipped), all 8-dim.
        assert result.embeddings_created == 3
        assert await repo.get_embedding_count(sid) == 3
        vectors = await repo.get_embedding_vectors(sid)
        assert all(len(v) == 8 for v in vectors)

        # AC2: NOT flagged despite chunk_count(5) > embedding_count(3).
        assert sid not in await backfill.list_sources_missing_chunk_embeddings()
        assert await backfill.count_missing_chunks([sid]) == 0
    finally:
        conn._pool = orig


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_partial_nonempty_source_still_flagged_with_empties(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """R.0d AC3: a genuinely partial source is STILL flagged (no R.0c regress).

    3 non-empty chunks + 1 empty chunk, only 1 non-empty embedded -> 2 non-empty
    chunks still lack vectors, so the source must remain flagged; the empty
    chunk does not inflate the count.
    """
    import surrealdb_service.connection as conn

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        repo = SourceRepository(config=live_surrealdb)
        created = await execute_query(
            "CREATE source SET title = $t, full_text = 'body';",
            {"t": _uid("r0d-partial")},
            config=live_surrealdb,
        )
        sid = str(created[0]["id"])
        await _add_chunk(live_surrealdb, sid, "alpha", 0)
        await _add_chunk(live_surrealdb, sid, "beta", 1)
        await _add_chunk(live_surrealdb, sid, "", 2, element_type="table")
        await _add_chunk(live_surrealdb, sid, "gamma", 3)

        # Only 1 of 3 non-empty chunks embedded.
        await repo.add_embedding(sid, "alpha", 0, [0.1] * 8)

        assert sid in await backfill.list_sources_missing_chunk_embeddings()
        # 3 non-empty - 1 embedded = 2 (the empty chunk is not counted).
        assert await backfill.count_missing_chunks([sid]) == 2
    finally:
        conn._pool = orig


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_discovery_query_finds_unembedded_sources(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """``list_sources_missing_chunk_embeddings`` returns chunked-but-unembedded.

    A source with chunks + no ``source_embedding`` rows appears; once *every*
    chunk has an embedding it disappears (idempotency at the discovery layer); a
    source with no chunks never appears.

    R.0c: dropping out requires the embedding count to REACH the chunk count.
    Adding a single embedding to a 3-chunk source no longer removes it (it is
    partially embedded — the old ``= 0`` detection wrongly dropped it here,
    silently leaving 2 chunks unembedded).
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

        # One embedding on a 3-chunk source = PARTIAL -> still flagged (R.0c).
        await repo.add_embedding(
            source_id=with_chunks,
            content="chunk 0",
            order=0,
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        partial_missing = await backfill.list_sources_missing_chunk_embeddings()
        assert with_chunks in partial_missing

        # Embed the remaining 2 chunks -> counts match -> drops out (idempotent).
        for order in (1, 2):
            await repo.add_embedding(
                source_id=with_chunks,
                content=f"chunk {order}",
                order=order,
                embedding=[0.1, 0.2, 0.3, 0.4],
            )
        missing_after = await backfill.list_sources_missing_chunk_embeddings()
        assert with_chunks not in missing_after
    finally:
        conn._pool = orig


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_discovery_query_flags_partially_embedded_source(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """R.0c: a PARTIALLY-embedded source must be flagged as needing embedding.

    The original ``= 0`` detection (any ``>= 1`` embeddings -> "done") would NOT
    flag a source with 1-of-3 chunks embedded, silently leaving 2 chunks without
    a vector — exactly the live staging bug. This proves the four states:

    * partial (1 of 3 embedded)   -> flagged   (AC1; fails old, passes new)
    * fully embedded (3 of 3)     -> NOT flagged (AC2)
    * zero embeddings (0 of 3)    -> flagged   (AC3; no regression)
    * count_missing_chunks counts only UNembedded chunks (partial = 2, not 3)
    """
    import surrealdb_service.connection as conn

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        repo = SourceRepository(config=live_surrealdb)

        # Partial: 3 chunks, 1 embedding row.
        partial = await _make_source_with_chunks(live_surrealdb, 3)
        await repo.add_embedding(partial, "c0", 0, [0.1, 0.2, 0.3, 0.4])

        # Full: 3 chunks, 3 embedding rows.
        full = await _make_source_with_chunks(live_surrealdb, 3)
        for order in range(3):
            await repo.add_embedding(full, f"c{order}", order, [0.1, 0.2, 0.3, 0.4])

        # Zero: 3 chunks, 0 embedding rows.
        zero = await _make_source_with_chunks(live_surrealdb, 3)

        missing = await backfill.list_sources_missing_chunk_embeddings()
        assert partial in missing  # AC1 — the bug fix
        assert full not in missing  # AC2 — no redundant re-embed
        assert zero in missing  # AC3 — no regression

        # count_missing_chunks reports only the unembedded chunks of a partial
        # source (3 chunks - 1 embedded = 2), not the whole chunk set.
        only_partial = await backfill.count_missing_chunks([partial])
        assert only_partial == 2
    finally:
        conn._pool = orig


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_embed_source_completes_partial_then_clean(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """R.0c AC4: after ``embed_source`` runs on a partial source, re-detection = 0.

    ``embed_source`` deletes the partial embeddings and rebuilds all chunks, so a
    previously-partial source drops out of the missing set entirely (idempotent /
    complete). Uses a local-only fake model — no Ollama.
    """
    import surrealdb_service.connection as conn
    from embeddings.service import EmbeddingService

    class _FakeModel:
        async def aembed(self, texts):
            return [[float(len(t) % 7)] * 8 for t in texts]

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        repo = SourceRepository(config=live_surrealdb)
        partial = await _make_source_with_chunks(live_surrealdb, 3)
        # Seed a stale partial embedding row (1 of 3).
        await repo.add_embedding(partial, "c0", 0, [9.0] * 8)
        assert await repo.get_embedding_count(partial) == 1
        assert partial in await backfill.list_sources_missing_chunk_embeddings()

        svc = EmbeddingService(source_repo=repo, embedding_model=_FakeModel())
        await svc.embed_source(partial)

        # All 3 chunks now embedded (the stale row was deleted + rebuilt).
        assert await repo.get_embedding_count(partial) == 3
        assert partial not in await backfill.list_sources_missing_chunk_embeddings()
        assert await backfill.count_missing_chunks([partial]) == 0
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
async def test_backfill_chunks_real_run_creates_embeddings_and_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """End-to-end: ``_backfill_chunks`` writes ``source_embedding`` rows.

    Drives the real ``EmbeddingService.embed_source`` (with a fake, local-only
    embedding MODEL so no Ollama is needed) against a container. Proves the AC:
    dry-run reports the missing set without writing; a real run populates
    ``source_embedding`` for the target sources; a re-run dry-run sees 0 missing
    (idempotent); the observed dimension is the model-derived value (here 8).
    """
    import surrealdb_service.connection as conn
    from embeddings.service import EmbeddingService

    class _FakeModel:
        """Local-only fake: deterministic 8-dim vector per text (stands in for
        the configured 1024-dim mxbai-embed-large; dimension is model-derived,
        not hardcoded in the script)."""

        async def aembed(self, texts):
            return [[float(len(t) % 7)] * 8 for t in texts]

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        repo = SourceRepository(config=live_surrealdb)
        sid = await _make_source_with_chunks(live_surrealdb, 3)

        # Dry-run: reports 1 source / 3 chunks missing, writes nothing.
        dry = await backfill._backfill_chunks(limit=None, dry_run=True)
        assert sid in await backfill.list_sources_missing_chunk_embeddings()
        assert dry["chunks_missing"] >= 3
        assert await repo.get_embedding_count(sid) == 0

        # Real run via a patched embedding service (real EmbeddingService +
        # real repo writes, fake local model).
        svc = EmbeddingService(source_repo=repo, embedding_model=_FakeModel())

        async def _fake_factory():
            return svc

        with patch(
            "app_main.dependencies.get_embedding_service", new=_fake_factory
        ):
            summary = await backfill._backfill_chunks(limit=None, dry_run=False)
        assert summary["embedded_sources"] >= 1

        # source_embedding rows now exist for the source's chunks.
        assert await repo.get_embedding_count(sid) == 3
        vectors = await repo.get_embedding_vectors(sid)
        assert len(vectors) == 3
        assert all(len(v) == 8 for v in vectors)  # model-derived dim

        # Idempotent: the source is no longer in the missing set on re-run.
        assert sid not in await backfill.list_sources_missing_chunk_embeddings()
    finally:
        conn._pool = orig


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
