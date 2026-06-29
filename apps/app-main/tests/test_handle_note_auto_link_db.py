"""Track Y.3 — DB-backed: the NOTE_AUTO_LINK job actually links a new note.

The unit tests (``test_handle_note_auto_link``) prove the handler drives the
orchestrator with mocks. This file closes AC1 end-to-end: the JOB PATH — the
registered ``handle_note_auto_link`` running a REAL ``NoteAutoLinkService`` over
a real SurrealDB container — writes ``related_note`` edges for a freshly-created,
already-embedded note (not only the manual endpoint). And AC2: the same job
path, when the auto-link step fails, leaves the note + edges intact (no
half-write).

The note is seeded with an embedding directly (the upstream embed job's job in
prod), so the orchestrator's embed step is a no-op; we still inject an
EmbeddingService stub so the service builds without a live model.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

import app_main.handlers as handlers
from app_main.services.note_auto_link_service import NoteAutoLinkService
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories import NoteRepository

pytestmark = pytest.mark.requires_docker


def _u(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


async def _make_note(
    config: SurrealDBConfig, *, embedding: list[float], title: str | None = None
) -> str:
    rows = await execute_query(
        "CREATE note SET title = $t, content = $t, embedding = $e RETURN AFTER;",
        {"t": title or _u("note"), "e": embedding},
        config=config,
    )
    return str(rows[0]["id"])


async def _targets(config: SurrealDBConfig, from_id: str) -> set[str]:
    rows = await execute_query(
        "SELECT out FROM related_note WHERE in = $f;",
        {"f": ensure_record_id(from_id)},
        config=config,
    )
    return {str(r["out"]) for r in rows}


async def _edge_count(config: SurrealDBConfig, from_id: str) -> int:
    rows = await execute_query(
        "SELECT count() AS c FROM related_note WHERE in = $f GROUP ALL;",
        {"f": ensure_record_id(from_id)},
        config=config,
    )
    return rows[0]["c"] if rows else 0


def _real_service(config: SurrealDBConfig) -> NoteAutoLinkService:
    """A real orchestrator over the live DB; the embed step is a stub (the seeded
    notes already carry embeddings, so embed_note is never reached)."""
    embed = AsyncMock()
    embed.embed_note = AsyncMock()  # unused: notes are pre-embedded
    return NoteAutoLinkService(note_repo=NoteRepository(config), embedding_service=embed)


async def test_job_links_a_new_embedded_note(live_surrealdb: SurrealDBConfig) -> None:
    """AC1: the job path (not the endpoint) writes related_note edges."""
    cfg = live_surrealdb
    q = await _make_note(cfg, embedding=[1.0, 0.0, 0.0])
    near = await _make_note(cfg, embedding=[0.99, 0.01, 0.0])  # ~1.0, above 0.9
    far = await _make_note(cfg, embedding=[0.0, 1.0, 0.0])  # ~0.0, below 0.9

    service = _real_service(cfg)
    with patch(
        "app_main.dependencies.get_note_auto_link_service",
        new=AsyncMock(return_value=service),
    ):
        result = await handlers.handle_note_auto_link(
            {"note_id": q, "k": 10, "min_similarity": 0.9}
        )

    assert result["success"] is True
    assert result["status"] == "linked"
    targets = await _targets(cfg, q)
    assert near in targets, "the job did not link the highly-similar note"
    assert far not in targets, "below-threshold note must not be linked"
    assert q not in targets, "a note must never link to itself"


async def test_job_is_idempotent_on_rerun(live_surrealdb: SurrealDBConfig) -> None:
    """AC1: re-running the job yields the identical edge set (no dup rows)."""
    cfg = live_surrealdb
    q = await _make_note(cfg, embedding=[1.0, 0.0, 0.0])
    await _make_note(cfg, embedding=[0.97, 0.03, 0.0])
    await _make_note(cfg, embedding=[0.96, 0.04, 0.0])

    service = _real_service(cfg)
    payload = {"note_id": q, "k": 10, "min_similarity": 0.8}
    with patch(
        "app_main.dependencies.get_note_auto_link_service",
        new=AsyncMock(return_value=service),
    ):
        await handlers.handle_note_auto_link(payload)
        targets_1, count_1 = await _targets(cfg, q), await _edge_count(cfg, q)
        await handlers.handle_note_auto_link(payload)
        targets_2, count_2 = await _targets(cfg, q), await _edge_count(cfg, q)

    assert targets_1 == targets_2
    assert count_1 == count_2, "re-run duplicated edge rows (RELATE not idempotent)"


async def test_job_failure_leaves_note_and_edges_intact(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC2: an auto-link step failure is isolated — the note is untouched and no
    half-written edges remain.

    We seed + embed the note (the upstream embed job already persisted it), then
    make the orchestrator's RELATE step raise mid-run. The job fails, but:
      * the note row still exists with its embedding (not corrupted);
      * no related_note edges were written (no partial graph state).
    """
    cfg = live_surrealdb
    q = await _make_note(cfg, embedding=[1.0, 0.0, 0.0])
    await _make_note(cfg, embedding=[0.98, 0.02, 0.0])

    service = _real_service(cfg)
    # Simulate the link step blowing up (e.g. DB transport error during RELATE).
    service.note_repo.relate_note = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("RELATE backend down")
    )

    with patch(
        "app_main.dependencies.get_note_auto_link_service",
        new=AsyncMock(return_value=service),
    ):
        with pytest.raises(RuntimeError, match="RELATE backend down"):
            await handlers.handle_note_auto_link(
                {"note_id": q, "k": 10, "min_similarity": 0.5}
            )

    # The note is still there with its embedding — the failing job never touched it.
    rows = await execute_query(
        "SELECT id, array::len(embedding) AS dim FROM $id;",
        {"id": ensure_record_id(q)},
        config=cfg,
    )
    assert rows and str(rows[0]["id"]) == q, "the note must survive the failed job"
    assert rows[0]["dim"] == 3, "the note's embedding must be intact"
    # No half-written edges from the failed run.
    assert await _edge_count(cfg, q) == 0
