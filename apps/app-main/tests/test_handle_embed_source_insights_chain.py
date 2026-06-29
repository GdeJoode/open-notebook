"""Track PL.3: ``_handle_embed_source`` chains INSIGHTS (gated) + the folded
chunk-count guard.

PL.3 adds, parallel to the PL.2 extract chain, an auto-INSIGHTS enqueue off a
successful source embed — gated by the per-notebook ``auto_insights`` toggle
(default ON) and the same ``embedded_chunks > 0`` guard the parse->embed link
uses. These DB-free seam tests assert:

  * toggle ON (notebook linked, auto_insights=True)  -> run_summaries enqueued;
  * toggle OFF (auto_insights=False)                 -> run_summaries SKIPPED
    (the extract chain still fires; insights is independent);
  * unlinked source (no notebook)                    -> insights default ON;
  * zero embedded chunks                             -> NEITHER extract NOR
    insights enqueued (the folded guard);
  * an insights enqueue failure is best-effort       -> the embed still returns,
    stage still embedded.

The enqueue is asserted at the seam (mock ``submit_command_job``) — the same
boundary PL.2 uses, because the chain runs through the module-level
``CommandService`` singleton bound to the env DB.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import app_main.handlers as handlers


def _patch_orchestrator(embedded_chunks: int = 5):
    orch = AsyncMock()
    orch.embed_source = AsyncMock(
        return_value={"source_id": "source:abc", "embedded_chunks": embedded_chunks}
    )
    return patch(
        "app_main.dependencies.get_source_embedding_orchestrator",
        return_value=orch,
    )


def _patch_source_repo(*, notebook_id=None, stage="embedded"):
    # PL.4: ``advance_source`` reads the stage via ``get_processing_stage`` to
    # decide what to dispatch; without a DB the mock returns ``stage`` (the embed
    # handler writes ``embedded`` then advances off it).
    repo = AsyncMock()
    repo.set_processing_stage = AsyncMock(return_value=True)
    repo.get_processing_stage = AsyncMock(return_value=stage)
    repo.get_notebook_id = AsyncMock(return_value=notebook_id)
    return patch("app_main.dependencies.get_source_repo", return_value=repo), repo


def _patch_notebook_repo(*, auto_insights: bool = True):
    repo = AsyncMock()
    repo.get_auto_insights = AsyncMock(return_value=auto_insights)
    return patch("app_main.dependencies.get_notebook_repo", return_value=repo), repo


def _commands(submit) -> list[str]:
    return [call.args[1] for call in submit.await_args_list]


@pytest.mark.asyncio
async def test_insights_chained_when_toggle_on() -> None:
    src_patch, _ = _patch_source_repo(notebook_id="notebook:nb1")
    nb_patch, nb_repo = _patch_notebook_repo(auto_insights=True)
    with _patch_orchestrator(), src_patch, nb_patch, patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(return_value="job:1"),
    ) as submit:
        result = await handlers._handle_embed_source({"source_id": "source:abc"})

    assert result["success"] is True
    assert result["insight_command_id"] == "job:1"
    commands = _commands(submit)
    assert "run_summaries" in commands
    assert "run_entities" in commands  # the extract chain still fires
    nb_repo.get_auto_insights.assert_awaited_once_with("notebook:nb1")


@pytest.mark.asyncio
async def test_insights_skipped_when_toggle_off() -> None:
    src_patch, _ = _patch_source_repo(notebook_id="notebook:nb1")
    nb_patch, nb_repo = _patch_notebook_repo(auto_insights=False)
    with _patch_orchestrator(), src_patch, nb_patch, patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(return_value="job:1"),
    ) as submit:
        result = await handlers._handle_embed_source({"source_id": "source:abc"})

    assert result["success"] is True
    # Insights SKIPPED -> no command id, no run_summaries enqueue.
    assert result["insight_command_id"] is None
    commands = _commands(submit)
    assert "run_summaries" not in commands
    # The KG (free) chain is independent of the insights toggle -> still fires.
    assert "run_entities" in commands
    nb_repo.get_auto_insights.assert_awaited_once_with("notebook:nb1")


@pytest.mark.asyncio
async def test_insights_default_on_for_unlinked_source() -> None:
    # No owning notebook -> no toggle to read -> default ON (run insights).
    src_patch, _ = _patch_source_repo(notebook_id=None)
    nb_patch, nb_repo = _patch_notebook_repo(auto_insights=False)
    with _patch_orchestrator(), src_patch, nb_patch, patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(return_value="job:1"),
    ) as submit:
        result = await handlers._handle_embed_source({"source_id": "source:abc"})

    assert result["insight_command_id"] == "job:1"
    assert "run_summaries" in _commands(submit)
    # An unlinked source never consults the notebook toggle.
    nb_repo.get_auto_insights.assert_not_awaited()


@pytest.mark.asyncio
async def test_zero_chunks_skips_extract_and_insights() -> None:
    # The folded PL.2 minor: a zero-chunk embed must spawn NO downstream jobs.
    src_patch, _ = _patch_source_repo(notebook_id="notebook:nb1")
    nb_patch, _ = _patch_notebook_repo(auto_insights=True)
    with _patch_orchestrator(embedded_chunks=0), src_patch, nb_patch, patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(return_value="job:1"),
    ) as submit:
        result = await handlers._handle_embed_source({"source_id": "source:abc"})

    assert result["success"] is True
    assert result["extract_command_id"] is None
    assert result["insight_command_id"] is None
    submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_insights_enqueue_failure_is_best_effort() -> None:
    src_patch, repo = _patch_source_repo(notebook_id="notebook:nb1")
    nb_patch, _ = _patch_notebook_repo(auto_insights=True)

    async def _submit(module, command, args):
        if command == "run_summaries":
            raise RuntimeError("insights queue down")
        return "job:extract"

    with _patch_orchestrator(), src_patch, nb_patch, patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(side_effect=_submit),
    ):
        result = await handlers._handle_embed_source({"source_id": "source:abc"})

    # Insights failed but the embed result still returns; extract still chained.
    assert result["success"] is True
    assert result["insight_command_id"] is None
    assert result["extract_command_id"] == "job:extract"
    repo.set_processing_stage.assert_awaited_once_with("source:abc", "embedded")
