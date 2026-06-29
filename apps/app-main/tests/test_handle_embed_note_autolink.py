"""Track Y.3: ``_handle_embed_single_item`` chains ``auto_link_note`` after a
successful NOTE embed.

The background trigger for auto-link is the embed job: a note can only be ranked
by similarity once it HAS an embedding, so the embed handler is the gate and the
auto-link job is the chained follow-up (mirroring R.0's
DOCUMENT_PARSE -> embed_source chaining).

These DB-free tests assert that seam:

  * note embedded (chunks_created > 0) -> exactly one ``auto_link_note`` job
    enqueued for the note;
  * note embedded but 0 produced       -> no auto-link job (nothing to link);
  * source embedded                    -> no auto-link job (note-only trigger);
  * enqueue failure                    -> embed result still returned
    (best-effort, never fails the already-persisted embed).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import app_main.handlers as handlers


def _embed_service(created: int):
    """An EmbeddingService stub whose embed_* returns ``embeddings_created``."""
    svc = AsyncMock()
    svc.embed_note = AsyncMock(
        return_value=SimpleNamespace(embeddings_created=created)
    )
    svc.embed_source = AsyncMock(
        return_value=SimpleNamespace(embeddings_created=created)
    )
    svc.embed_insight = AsyncMock(
        return_value=SimpleNamespace(embeddings_created=created)
    )
    return svc


def _patch_embed_service(svc):
    """Patch the lazily-imported ``get_embedding_service`` factory.

    ``_handle_embed_single_item`` imports it from ``app_main.dependencies`` at
    call time, so we patch it at the source module. It is awaited, hence the
    AsyncMock return.
    """
    return patch(
        "app_main.dependencies.get_embedding_service",
        new=AsyncMock(return_value=svc),
    )


@pytest.mark.asyncio
async def test_autolink_enqueued_after_note_embed() -> None:
    with _patch_embed_service(_embed_service(created=1)), patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(return_value="job:autolink-1"),
    ) as submit:
        result = await handlers._handle_embed_single_item(
            {"item_id": "note:abc", "item_type": "note"}
        )

    assert result["success"] is True
    assert result["auto_link_command_id"] == "job:autolink-1"
    submit.assert_awaited_once()
    args = submit.await_args.args
    # submit_command_job(module_name, command_name, command_args)
    assert args[1] == "auto_link_note"
    assert args[2] == {"note_id": "note:abc"}


@pytest.mark.asyncio
async def test_no_autolink_when_note_produced_no_embedding() -> None:
    with _patch_embed_service(_embed_service(created=0)), patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(),
    ) as submit:
        result = await handlers._handle_embed_single_item(
            {"item_id": "note:empty", "item_type": "note"}
        )

    assert result["success"] is True
    assert result["auto_link_command_id"] is None
    submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_autolink_for_source_embed() -> None:
    """Auto-link is the note trigger only; embedding a source never chains it."""
    with _patch_embed_service(_embed_service(created=4)), patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(),
    ) as submit:
        result = await handlers._handle_embed_single_item(
            {"item_id": "source:abc", "item_type": "source"}
        )

    assert result["success"] is True
    assert result["auto_link_command_id"] is None
    submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_autolink_enqueue_failure_does_not_fail_embed() -> None:
    """A queue hiccup on the auto-link enqueue must not fail the (already
    persisted) note embed — best-effort isolation at the enqueue seam."""
    with _patch_embed_service(_embed_service(created=2)), patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(side_effect=RuntimeError("queue down")),
    ):
        result = await handlers._handle_embed_single_item(
            {"item_id": "note:abc", "item_type": "note"}
        )

    # Embedding already persisted — the result must still come back, no raise.
    assert result["success"] is True
    assert result["chunks_created"] == 2
    assert result["auto_link_command_id"] is None
