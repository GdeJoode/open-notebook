"""Container tests for note-level similarity + the idempotent relate helper (Y.1).

`NoteRepository.find_related_by_embedding` is the note-level mirror of
`SourceRepository.find_related_by_embedding`: cosine over `note.embedding`,
exclude self, exclude empty-embedding notes, rank score DESC. `relate_note` is
the idempotent clear-before-relate helper for the `related_note` edge (RELATE is
not idempotent — Track W.2/W.3) that refuses self-edges.

These tests prove the Y.1 acceptance against a real SurrealDB container:

* ranks other notes by cosine DESC, excludes self;
* a query note with an empty (`[]`) embedding → graceful `[]`, no crash;
* `relate_note` is idempotent (same pair twice → exactly ONE edge row);
* `relate_note` refuses a self-edge;
* `similarity_score` / `method` / `created_at` round-trip on the edge.

Run:

    uv run --project packages/surrealdb-service \\
        pytest packages/surrealdb-service/tests/test_note_similarity_roundtrip.py -v
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
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


# ---------------------------------------------------------------------------
# AC1: ranks other notes by cosine DESC, excludes self.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_find_related_ranks_by_cosine_and_excludes_self(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NoteRepository(config=live_surrealdb)

    # Query vector points along x. A near (mostly-x) note should outrank a
    # far (mostly-y) note; the orthogonal note ranks last.
    query = await _make_note(live_surrealdb, embedding=[1.0, 0.0, 0.0])
    near = await _make_note(live_surrealdb, embedding=[0.9, 0.1, 0.0])
    mid = await _make_note(live_surrealdb, embedding=[0.5, 0.5, 0.0])
    far = await _make_note(live_surrealdb, embedding=[0.0, 1.0, 0.0])

    results = await repo.find_related_by_embedding(query, k=10)
    ids = [r["id"] for r in results]

    # Self excluded.
    assert query not in ids, "query note must not be its own related note"

    # The three seeded notes are present and ranked near > mid > far.
    seeded = [r for r in results if r["id"] in {near, mid, far}]
    seeded_ids = [r["id"] for r in seeded]
    assert seeded_ids[:3] == [near, mid, far], (
        f"cosine ranking wrong: expected [near, mid, far], got {seeded_ids}"
    )
    # Scores are descending across the seeded notes.
    scores = [r["score"] for r in seeded]
    assert scores == sorted(scores, reverse=True)
    # Shape: {id, title, score}.
    assert set(results[0].keys()) == {"id", "title", "score"}


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_find_related_respects_k(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NoteRepository(config=live_surrealdb)
    query = await _make_note(live_surrealdb, embedding=[1.0, 0.0, 0.0])
    for _ in range(4):
        await _make_note(live_surrealdb, embedding=[0.8, 0.2, 0.0])

    results = await repo.find_related_by_embedding(query, k=2)
    assert len(results) == 2, f"k=2 should cap the result set, got {len(results)}"


# ---------------------------------------------------------------------------
# AC1: a no/empty-embedding query note is graceful (no crash).
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_find_related_empty_embedding_returns_empty(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NoteRepository(config=live_surrealdb)
    # Other notes exist with real embeddings.
    await _make_note(live_surrealdb, embedding=[1.0, 0.0, 0.0])
    # The query note has the strict-but-empty embedding (`[]`).
    unembedded = await _make_note(live_surrealdb, embedding=[])

    results = await repo.find_related_by_embedding(unembedded, k=10)
    assert results == [], "an unembedded note must yield [] (needs-embedding), not crash"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_find_related_excludes_empty_embedding_candidates(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NoteRepository(config=live_surrealdb)
    query = await _make_note(live_surrealdb, embedding=[1.0, 0.0, 0.0])
    good = await _make_note(live_surrealdb, embedding=[0.9, 0.1, 0.0])
    empty = await _make_note(live_surrealdb, embedding=[])

    results = await repo.find_related_by_embedding(query, k=50)
    ids = [r["id"] for r in results]
    assert good in ids
    assert empty not in ids, "an empty-embedding candidate must never rank"


# ---------------------------------------------------------------------------
# AC3: relate_note is idempotent + refuses self-edges + fields round-trip.
# ---------------------------------------------------------------------------


async def _edge_rows(
    config: SurrealDBConfig, from_id: str, to_id: str
) -> list[dict]:
    from surrealdb_service.connection import ensure_record_id

    return await execute_query(
        "SELECT in, out, similarity_score, method, created_at "
        "FROM related_note WHERE in = $f AND out = $t",
        {"f": ensure_record_id(from_id), "t": ensure_record_id(to_id)},
        config=config,
    )


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relate_note_idempotent_single_edge(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NoteRepository(config=live_surrealdb)
    a = await _make_note(live_surrealdb, embedding=[1.0, 0.0])
    b = await _make_note(live_surrealdb, embedding=[0.9, 0.1])

    assert await repo.relate_note(a, b, similarity_score=0.91) is True
    assert await repo.relate_note(a, b, similarity_score=0.95) is True

    rows = await _edge_rows(live_surrealdb, a, b)
    assert len(rows) == 1, (
        f"relating the same pair twice must yield exactly ONE edge, got {len(rows)}"
    )
    # The surviving row carries the latest score (clear-before-relate replaces).
    assert abs(float(rows[0]["similarity_score"]) - 0.95) < 1e-9


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relate_note_refuses_self_edge(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NoteRepository(config=live_surrealdb)
    a = await _make_note(live_surrealdb, embedding=[1.0, 0.0])
    assert await repo.relate_note(a, a, similarity_score=1.0) is False
    rows = await _edge_rows(live_surrealdb, a, a)
    assert rows == [], "a self-edge must never be written"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relate_note_fields_round_trip(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NoteRepository(config=live_surrealdb)
    a = await _make_note(live_surrealdb, embedding=[1.0, 0.0])
    b = await _make_note(live_surrealdb, embedding=[0.5, 0.5])

    assert await repo.relate_note(
        a, b, similarity_score=0.707, method="embedding"
    ) is True
    rows = await _edge_rows(live_surrealdb, a, b)
    assert len(rows) == 1
    edge = rows[0]
    assert str(edge["in"]) == a and str(edge["out"]) == b
    assert abs(float(edge["similarity_score"]) - 0.707) < 1e-9
    assert edge["method"] == "embedding"
    assert edge["created_at"] is not None, "created_at default must populate"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relate_note_rejects_non_note_id(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NoteRepository(config=live_surrealdb)
    a = await _make_note(live_surrealdb, embedding=[1.0, 0.0])
    # A source id is a valid record id but the wrong table for related_note.
    src = await execute_query(
        "CREATE source SET title = $t RETURN AFTER;",
        {"t": _unique("src")},
        config=live_surrealdb,
    )
    assert await repo.relate_note(
        a, str(src[0]["id"]), similarity_score=0.5
    ) is False


# ---------------------------------------------------------------------------
# SECURITY: a SurrealQL-injection id is REFUSED — no rogue edge, no dropped
# table. `RecordID.parse` splits on the first colon and round-trips an injection
# payload verbatim, so a `startswith("note:")` check would pass it; the strict
# `_RECORD_ID_RE` validator must reject it BEFORE interpolation.
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relate_note_refuses_sql_injection_id(
    live_surrealdb: SurrealDBConfig,
) -> None:
    repo = NoteRepository(config=live_surrealdb)
    a = await _make_note(live_surrealdb, embedding=[1.0, 0.0])
    b = await _make_note(live_surrealdb, embedding=[0.5, 0.5])

    # A drop-table payload in BOTH the to and from positions. If interpolated,
    # `RELATE note:x; REMOVE TABLE note; --->related_note->...` executes as
    # multiple statements and wipes the `note` table (the reproduced blocker).
    payload = "note:x; REMOVE TABLE note; --"

    async def _note_count() -> int:
        rows = await execute_query(
            "SELECT count() AS c FROM note GROUP ALL;", config=live_surrealdb
        )
        return rows[0]["c"] if rows else 0

    async def _related_count() -> int:
        rows = await execute_query(
            "SELECT count() AS c FROM related_note GROUP ALL;",
            config=live_surrealdb,
        )
        return rows[0]["c"] if rows else 0

    note_before = await _note_count()
    related_before = await _related_count()
    assert note_before >= 2, "setup: at least the two seeded notes must exist"

    # to-position injection
    assert await repo.relate_note(a, payload, similarity_score=0.9) is False
    # from-position injection
    assert await repo.relate_note(payload, b, similarity_score=0.9) is False

    # The `note` table is intact (not dropped) and no rogue edge was written.
    assert await _note_count() == note_before, (
        "SECURITY FAIL: the `note` table row count changed — injection executed"
    )
    assert await _related_count() == related_before, (
        "SECURITY FAIL: a rogue `related_note` edge was written from an unsafe id"
    )
