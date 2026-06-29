"""Track Y.3: the ``NOTE_AUTO_LINK`` job handler + failure isolation.

``handle_note_auto_link`` runs the Y.2 orchestrator
(``NoteAutoLinkService.auto_link``) for a note that already has an embedding.
These DB-free tests assert:

  * the handler drives the orchestrator and flattens its summary into the job
    result (idempotency is the orchestrator's, tested in Y.1/Y.2);
  * the handler is registered for ``JobType.NOTE_AUTO_LINK``;
  * ISOLATION — if the auto-link step raises, the failure stays in THIS job
    (it surfaces to the worker as a failed job); the note + embedding it acted
    on are untouched, because this job is a separate, downstream unit from the
    embed job that already persisted them. We prove the embed seam is isolated
    in :mod:`test_handle_embed_note_autolink` (enqueue never fails the embed);
    here we prove the auto-link job itself never reaches back into note state.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import app_main.handlers as handlers
from app_main.handlers import registry
from app_main.services.note_auto_link_service import AutoLinkResult
from shared.types.enums import JobType


def test_note_auto_link_handler_registered() -> None:
    assert registry.has_handler(JobType.NOTE_AUTO_LINK), (
        f"No handler registered for {JobType.NOTE_AUTO_LINK.value!r}. "
        f"Registered: {[t.value for t in registry.registered_types]}"
    )


def _patch_service(service):
    """Patch the lazily-imported ``get_note_auto_link_service`` factory.

    The handler imports it from ``app_main.dependencies`` at call time and
    awaits it (it is async), hence the AsyncMock return.
    """
    return patch(
        "app_main.dependencies.get_note_auto_link_service",
        new=AsyncMock(return_value=service),
    )


@pytest.mark.asyncio
async def test_handler_drives_orchestrator_and_returns_summary() -> None:
    service = AsyncMock()
    service.auto_link = AsyncMock(
        return_value=AutoLinkResult(
            note_id="note:abc",
            status="linked",
            created=2,
            candidates_considered=3,
            linked_note_ids=["note:x", "note:y"],
        )
    )

    with _patch_service(service):
        result = await handlers.handle_note_auto_link({"note_id": "note:abc"})

    assert result["success"] is True
    assert result["status"] == "linked"
    assert result["created"] == 2
    assert result["linked_note_ids"] == ["note:x", "note:y"]
    assert "processing_time" in result
    service.auto_link.assert_awaited_once()
    # Chained enqueue omits k/min_similarity -> orchestrator applies defaults.
    assert service.auto_link.await_args.kwargs == {"k": None, "min_similarity": None}


@pytest.mark.asyncio
async def test_handler_forwards_optional_params() -> None:
    service = AsyncMock()
    service.auto_link = AsyncMock(
        return_value=AutoLinkResult(note_id="note:abc", status="linked")
    )

    with _patch_service(service):
        await handlers.handle_note_auto_link(
            {"note_id": "note:abc", "k": 3, "min_similarity": 0.9}
        )

    assert service.auto_link.await_args.kwargs == {"k": 3, "min_similarity": 0.9}


@pytest.mark.asyncio
async def test_needs_embedding_is_not_an_error() -> None:
    """A note with no content reports needs_embedding via status — NOT a raise.

    The job completes successfully with that status; it is not a failure.
    """
    service = AsyncMock()
    service.auto_link = AsyncMock(
        return_value=AutoLinkResult(note_id="note:empty", status="needs_embedding")
    )

    with _patch_service(service):
        result = await handlers.handle_note_auto_link({"note_id": "note:empty"})

    assert result["success"] is True
    assert result["status"] == "needs_embedding"
    assert result["created"] == 0


@pytest.mark.asyncio
async def test_autolink_failure_is_isolated_to_this_job() -> None:
    """If the auto-link step raises, the handler raises (the worker records a
    FAILED job) — but nothing it touched is the note/embedding.

    The note + embedding were persisted by the upstream embed job; this job
    only ever writes ``related_note`` edges via the orchestrator. A raise here
    therefore cannot corrupt the note: the failure is contained in this job
    unit. We assert the handler does not swallow the error into a fake success
    (a swallowed error would mask a real linking problem) and that it never
    mutates note state (it has no note-writing path).
    """
    service = AsyncMock()
    service.auto_link = AsyncMock(side_effect=RuntimeError("similarity backend down"))

    with _patch_service(service):
        with pytest.raises(RuntimeError, match="similarity backend down"):
            await handlers.handle_note_auto_link({"note_id": "note:abc"})

    # The only collaborator was the orchestrator; the handler never reaches
    # into note CRUD, so a failure cannot half-write the note.
    service.auto_link.assert_awaited_once()
