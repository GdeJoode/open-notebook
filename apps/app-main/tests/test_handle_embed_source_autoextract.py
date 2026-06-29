"""Track PL.2: ``_handle_embed_source`` auto-enqueues ``run_entities`` and
advances ``source.processing_stage``.

Before PL.2 a freshly embedded source NEVER had entities until a manual
``POST /sources/{id}/run-entities`` — the KG never built automatically. The
embed handler now chains entity extraction off a successful SOURCE embed
(mirroring the DOCUMENT_PARSE -> embed chain), and writes the per-source
pipeline stage at each transition. These DB-free tests assert that seam:

  * successful embed  -> exactly one ``run_entities`` job enqueued + stage
    advanced to ``embedded``;
  * enqueue failure   -> the embed result still returned (best-effort chaining,
    never fails the already-persisted embed);
  * the chain is SOURCES-ONLY — the note path (_handle_embed_single_item) chains
    NOTE_AUTO_LINK, not extraction (regression guard).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import app_main.handlers as handlers


def _patch_orchestrator(embedded_chunks: int = 5):
    """Patch the lazily-imported ``get_source_embedding_orchestrator`` factory."""
    orch = AsyncMock()
    orch.embed_source = AsyncMock(
        return_value={"source_id": "source:abc", "embedded_chunks": embedded_chunks}
    )
    return patch(
        "app_main.dependencies.get_source_embedding_orchestrator",
        return_value=orch,
    )


def _patch_stage_repo(*, notebook_id=None):
    """Patch the lazily-imported ``get_source_repo`` so the stage write is a
    no-op spy (no DB).

    ``get_notebook_id`` (used by the PL.3 insights toggle resolution) returns
    ``notebook_id`` — ``None`` by default (unlinked source -> insights default
    ON, no toggle read)."""
    repo = AsyncMock()
    repo.set_processing_stage = AsyncMock(return_value=True)
    repo.get_notebook_id = AsyncMock(return_value=notebook_id)
    return patch("app_main.dependencies.get_source_repo", return_value=repo), repo


def _patch_notebook_repo(*, auto_insights: bool = True):
    """Patch the lazily-imported ``get_notebook_repo`` so the PL.3 toggle read is
    a no-op spy (no DB) returning ``auto_insights``."""
    repo = AsyncMock()
    repo.get_auto_insights = AsyncMock(return_value=auto_insights)
    return patch("app_main.dependencies.get_notebook_repo", return_value=repo), repo


def _enqueued_commands(submit) -> list[str]:
    """The ``command_name`` of every submit_command_job call (positional arg 1)."""
    return [call.args[1] for call in submit.await_args_list]


@pytest.mark.asyncio
async def test_autoextract_enqueued_after_embed() -> None:
    # Unlinked source (notebook_id None) -> insights default ON, no toggle read.
    stage_patch, repo = _patch_stage_repo()
    with _patch_orchestrator(), stage_patch, patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(return_value="job:1"),
    ) as submit:
        result = await handlers._handle_embed_source({"source_id": "source:abc"})

    assert result["success"] is True
    assert result["extract_command_id"] == "job:1"
    # PL.2 extract chain + PL.3 insights chain (default ON for an unlinked
    # source) both fire off a successful embed with chunks.
    assert result["insight_command_id"] == "job:1"
    commands = _enqueued_commands(submit)
    assert "run_entities" in commands
    assert "run_summaries" in commands
    # submit_command_job(module_name, command_name, command_args)
    for call in submit.await_args_list:
        assert call.args[2] == {"source_id": "source:abc"}

    # PL.2: stage advanced to "embedded" after a successful embed (the graph
    # stages graphed/complete are set in the extract handler, not here).
    repo.set_processing_stage.assert_awaited_once_with("source:abc", "embedded")


@pytest.mark.asyncio
async def test_enqueue_failure_does_not_fail_embed() -> None:
    stage_patch, repo = _patch_stage_repo()
    with _patch_orchestrator(), stage_patch, patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(side_effect=RuntimeError("queue down")),
    ):
        result = await handlers._handle_embed_source({"source_id": "source:abc"})

    # The embed already persisted vectors — the result must still come back, and
    # the stage must still have advanced to embedded (the failures are only the
    # downstream extract + insights enqueues, both best-effort).
    assert result["success"] is True
    assert result["extract_command_id"] is None
    assert result["insight_command_id"] is None
    repo.set_processing_stage.assert_awaited_once_with("source:abc", "embedded")


@pytest.mark.asyncio
async def test_embed_failure_sets_stage_failed() -> None:
    orch = AsyncMock()
    orch.embed_source = AsyncMock(side_effect=RuntimeError("embed boom"))
    stage_patch, repo = _patch_stage_repo()
    with patch(
        "app_main.dependencies.get_source_embedding_orchestrator",
        return_value=orch,
    ), stage_patch, patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(),
    ) as submit:
        with pytest.raises(RuntimeError, match="embed boom"):
            await handlers._handle_embed_source({"source_id": "source:abc"})

    # A hard embed failure -> stage = failed, and NO extract is chained.
    repo.set_processing_stage.assert_awaited_once_with("source:abc", "failed")
    submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_note_embed_does_not_chain_extraction() -> None:
    """Regression guard: the NOTE embed path chains NOTE_AUTO_LINK, never
    run_entities. PL.2's extract chain is sources-only."""
    embed_result = AsyncMock()
    embed_result.embeddings_created = 2
    service = AsyncMock()
    service.embed_note = AsyncMock(return_value=embed_result)

    with patch(
        "app_main.dependencies.get_embedding_service",
        new=AsyncMock(return_value=service),
    ), patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(return_value="job:link-1"),
    ) as submit:
        result = await handlers._handle_embed_single_item(
            {"item_id": "note:xyz", "item_type": "note"}
        )

    assert result["success"] is True
    # The only enqueued command is the auto-link, never run_entities.
    submit.assert_awaited_once()
    assert submit.await_args.args[1] == "auto_link_note"
