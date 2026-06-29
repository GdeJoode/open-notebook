"""Container tests for note→source similarity + the idempotent relate helper (NS.1).

`NoteRepository.find_related_sources_by_embedding` is the cross-type analogue of
the note→note `find_related_by_embedding`: cosine of `note.embedding` vs every
populated aggregate `source.embedding`, exclude no-embedding sources, rank score
DESC. `relate_note_source` is the idempotent clear-before-relate helper for the
`note_about` edge (RELATE is not idempotent — Track W/Y/Z) that enforces
note→source typing and refuses injection / wrong-table ids.

These tests prove the NS.1 acceptance against a real SurrealDB container:

* ranks sources by cosine DESC against the note's embedding (the cross-type
  ranking — same 1024-dim mxbai space for note.embedding and the source
  mean-pool source.embedding);
* a query note with an empty (`[]`) embedding → graceful `[]`, no crash;
* sources without an aggregate embedding never rank;
* `relate_note_source` is idempotent (same pair twice → exactly ONE edge row,
  latest score);
* `relate_note_source` enforces note→source typing (note-in / source-out);
* a SurrealQL-injection id in EITHER position is refused — `note`/`source`/
  `note_about` row counts unchanged.

Run:

    uv run --project packages/surrealdb-service \\
        pytest packages/surrealdb-service/tests/test_note_source_similarity_roundtrip.py -v
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.notebook import NoteRepository


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


async def _make_note(
    config: SurrealDBConfig, *, embedding: list[float], title: str | None = None
) -> str:
    """Create a note with an explicit embedding; return its record id (str).

    `note.embedding` is strict `array<float>` (no default), so it must always be
    supplied — an unembedded note carries `[]`, never NONE.
    """
    rows = await execute_query(
        "CREATE note SET title = $t, content = $t, embedding = $e RETURN AFTER;",
        {"t": title or _unique("note"), "e": embedding},
        config=config,
    )
    return str(rows[0]["id"])


async def _make_source(
    config: SurrealDBConfig, *, embedding: list[float] | None, title: str | None = None
) -> str:
    """Create a source with an explicit aggregate embedding (or NONE)."""
    if embedding is None:
        rows = await execute_query(
            "CREATE source SET title = $t RETURN AFTER;",
            {"t": title or _unique("src")},
            config=config,
        )
    else:
        rows = await execute_query(
            "CREATE source SET title = $t, embedding = $e RETURN AFTER;",
            {"t": title or _unique("src"), "e": embedding},
            config=config,
        )
    return str(rows[0]["id"])


# ---------------------------------------------------------------------------
# AC1: ranks sources by cosine DESC against the note's embedding.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_find_related_sources_ranks_by_cosine(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NoteRepository(config=live_surrealdb)

    # The `live_surrealdb` fixture is session-scoped with no per-test cleanup, so
    # sources seeded by other suites (and this track's other tests) accumulate in
    # the same `source` table. Real sources on staging are 1024-dim; isolate this
    # test's candidate set by seeding in a UNIQUE 5-dim space — no other suite
    # embeds sources at 5 dims, and `find_related_sources_by_embedding` only
    # considers candidates whose `array::len(embedding)` equals the query length
    # (the dim guard), so the 5-dim note query's candidate set is exactly this
    # test's sources. The Y.2 lesson: do not assert absolute top-k membership
    # over a shared table.
    #
    # Vectors point along the first axis; a near (mostly-axis-0) source outranks
    # a far (orthogonal) source, with the 45-degree source in between.
    note = await _make_note(live_surrealdb, embedding=[1.0, 0.0, 0.0, 0.0, 0.0])
    near = await _make_source(live_surrealdb, embedding=[0.9, 0.1, 0.0, 0.0, 0.0])
    mid = await _make_source(live_surrealdb, embedding=[0.5, 0.5, 0.0, 0.0, 0.0])
    far = await _make_source(live_surrealdb, embedding=[0.0, 1.0, 0.0, 0.0, 0.0])

    results = await repo.find_related_sources_by_embedding(note, k=10)
    ids = [r["id"] for r in results]

    # The length guard isolates candidates to this test's 5-dim sources, so the
    # full result set is exactly {near, mid, far}, ranked near > mid > far.
    assert ids == [near, mid, far], (
        f"cross-type cosine ranking wrong: expected [near, mid, far], got {ids}"
    )
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert set(results[0].keys()) == {"id", "title", "score"}


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_find_related_sources_respects_k(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NoteRepository(config=live_surrealdb)
    # Unique 6-dim space so other suites' sources can't contaminate the LIMIT.
    note = await _make_note(live_surrealdb, embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(4):
        await _make_source(
            live_surrealdb, embedding=[0.8, 0.2, 0.0, 0.0, 0.0, 0.0]
        )

    results = await repo.find_related_sources_by_embedding(note, k=2)
    assert len(results) == 2, f"k=2 should cap the result set, got {len(results)}"


# ---------------------------------------------------------------------------
# AC1: a no/empty-embedding query note is graceful (no crash).
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_find_related_sources_empty_note_embedding_returns_empty(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NoteRepository(config=live_surrealdb)
    # A real source exists with an embedding.
    await _make_source(live_surrealdb, embedding=[1.0, 0.0, 0.0])
    # The query note has the strict-but-empty embedding (`[]`).
    unembedded = await _make_note(live_surrealdb, embedding=[])

    results = await repo.find_related_sources_by_embedding(unembedded, k=10)
    assert results == [], (
        "an unembedded note must yield [] (needs-embedding), not crash"
    )


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_find_related_sources_excludes_no_embedding_sources(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NoteRepository(config=live_surrealdb)
    # Unique 7-dim space to isolate from other suites.
    note = await _make_note(
        live_surrealdb, embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    good = await _make_source(
        live_surrealdb, embedding=[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    # A source with NO aggregate embedding (embedding is NONE) must never rank.
    no_emb = await _make_source(live_surrealdb, embedding=None)

    results = await repo.find_related_sources_by_embedding(note, k=50)
    ids = [r["id"] for r in results]
    assert good in ids
    assert no_emb not in ids, "a source without an embedding must never rank"


# ---------------------------------------------------------------------------
# AC3: relate_note_source is idempotent + enforces typing + fields round-trip.
# ---------------------------------------------------------------------------


async def _edge_rows(
    config: SurrealDBConfig, note_id: str, source_id: str
) -> list[dict]:
    return await execute_query(
        "SELECT in, out, similarity_score, method, created_at "
        "FROM note_about WHERE in = $f AND out = $t",
        {"f": ensure_record_id(note_id), "t": ensure_record_id(source_id)},
        config=config,
    )


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relate_note_source_idempotent_single_edge(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NoteRepository(config=live_surrealdb)
    note = await _make_note(live_surrealdb, embedding=[1.0, 0.0])
    src = await _make_source(live_surrealdb, embedding=[0.9, 0.1])

    assert await repo.relate_note_source(note, src, similarity_score=0.91) is True
    assert await repo.relate_note_source(note, src, similarity_score=0.95) is True

    rows = await _edge_rows(live_surrealdb, note, src)
    assert len(rows) == 1, (
        f"relating the same pair twice must yield exactly ONE edge, got {len(rows)}"
    )
    # The surviving row carries the latest score (clear-before-relate replaces).
    assert abs(float(rows[0]["similarity_score"]) - 0.95) < 1e-9


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relate_note_source_fields_round_trip(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NoteRepository(config=live_surrealdb)
    note = await _make_note(live_surrealdb, embedding=[1.0, 0.0])
    src = await _make_source(live_surrealdb, embedding=[0.5, 0.5])

    assert await repo.relate_note_source(
        note, src, similarity_score=0.707, method="embedding"
    ) is True
    rows = await _edge_rows(live_surrealdb, note, src)
    assert len(rows) == 1
    edge = rows[0]
    assert str(edge["in"]) == note and str(edge["out"]) == src
    assert abs(float(edge["similarity_score"]) - 0.707) < 1e-9
    assert edge["method"] == "embedding"
    assert edge["created_at"] is not None, "created_at default must populate"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relate_note_source_enforces_typing(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """note→source typing: a source in the 'in' slot or a note in the 'out' slot
    is refused (and the swapped/wrong-table call writes no edge)."""
    repo = NoteRepository(config=live_surrealdb)
    note = await _make_note(live_surrealdb, embedding=[1.0, 0.0])
    src = await _make_source(live_surrealdb, embedding=[0.9, 0.1])

    # Swapped: source as 'in' (note_id slot), note as 'out' (source_id slot).
    assert await repo.relate_note_source(src, note, similarity_score=0.9) is False
    # A note in the 'out' (source) slot is rejected.
    other_note = await _make_note(live_surrealdb, embedding=[0.5, 0.5])
    assert await repo.relate_note_source(
        note, other_note, similarity_score=0.9
    ) is False
    # No edge written for the swapped call.
    assert await _edge_rows(live_surrealdb, src, note) == []


# ---------------------------------------------------------------------------
# SECURITY (AC3): a SurrealQL-injection id in EITHER position is REFUSED — no
# rogue edge, no dropped table. `RecordID.parse` splits on the first colon and
# round-trips an injection payload verbatim, so a `startswith("note:")` check
# would pass it; the strict `_RECORD_ID_RE` validator must reject it BEFORE
# interpolation. Mirrors `test_relate_note_refuses_sql_injection_id` (Y.1).
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relate_note_source_refuses_sql_injection_id(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NoteRepository(config=live_surrealdb)
    note = await _make_note(live_surrealdb, embedding=[1.0, 0.0])
    src = await _make_source(live_surrealdb, embedding=[0.5, 0.5])

    # A drop-table payload. If interpolated as a literal record id, the trailing
    # statements would execute and wipe the table (the reproduced blocker). The
    # `note:`-prefixed payload also passes a naive `startswith("note:")` check —
    # only the strict `_RECORD_ID_RE` rejects it.
    note_payload = "note:x; REMOVE TABLE note; --"
    source_payload = "source:x; REMOVE TABLE source; --"

    async def _count(table: str) -> int:
        rows = await execute_query(
            f"SELECT count() AS c FROM {table} GROUP ALL;", config=live_surrealdb
        )
        return rows[0]["c"] if rows else 0

    note_before = await _count("note")
    source_before = await _count("source")
    edge_before = await _count("note_about")
    assert note_before >= 1 and source_before >= 1, "setup: seeded rows must exist"

    # in-position (note slot) injection — drop-table note payload.
    assert await repo.relate_note_source(
        note_payload, src, similarity_score=0.9
    ) is False
    # out-position (source slot) injection — drop-table source payload.
    assert await repo.relate_note_source(
        note, source_payload, similarity_score=0.9
    ) is False
    # Both positions injected at once.
    assert await repo.relate_note_source(
        note_payload, source_payload, similarity_score=0.9
    ) is False

    # All three tables intact (not dropped) and no rogue edge was written.
    assert await _count("note") == note_before, (
        "SECURITY FAIL: the `note` table row count changed — injection executed"
    )
    assert await _count("source") == source_before, (
        "SECURITY FAIL: the `source` table row count changed — injection executed"
    )
    assert await _count("note_about") == edge_before, (
        "SECURITY FAIL: a rogue `note_about` edge was written from an unsafe id"
    )
