"""Track PL.4: the declarative ``SOURCE_PIPELINE`` + the ``advance_source`` driver.

PL.4 consolidates the PL.1-PL.3 auto-chain wiring into one place. These DB-free
unit tests pin the stage-transition TABLE — for each persisted
``processing_stage`` value, ``advance_source`` must dispatch exactly the action
the old scattered handlers did:

  * ``ingested``  -> enqueue EMBED (``embed_source``);
  * ``embedded``  -> enqueue EXTRACT (``run_entities``) AND the parallel
    toggle-gated INSIGHTS (``run_summaries``); the chunk-count guard suppresses
    both when ``embedded_chunks <= 0``;
  * ``extracted`` -> run GRAPH inline (mentions refresh) then advance
    ``graphed`` -> ``complete``;
  * ``awaiting_schema_review`` / ``failed`` / ``complete`` -> no auto-advance
    (parked/terminal — need an explicit resume);
  * the gates (``auto_insights`` toggle), the ``depends_on`` wiring, and the
    best-effort posture (a queue hiccup / projection failure never raises).

The pipeline DEFINITION itself (the shape of the table) is asserted separately
so a future edit to the stages is caught structurally.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app_main.services.source_pipeline import (
    SOURCE_PIPELINE,
    StageName,
    advance_source,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# The definition's shape (structural assertions on the declarative table).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope=None)
async def test_pipeline_definition_shape() -> None:
    """Structural assertions on the declarative table (async only to satisfy the
    module-level ``pytestmark`` — it does no IO)."""
    by_name = {s.name: s for s in SOURCE_PIPELINE}

    # The spine: ingest -> embed -> extract -> graph, each depends on the prior.
    assert by_name[StageName.INGEST].depends_on is None
    assert by_name[StageName.INGEST].produces == "ingested"
    assert by_name[StageName.EMBED].depends_on is StageName.INGEST
    assert by_name[StageName.EMBED].produces == "embedded"
    assert by_name[StageName.EMBED].enqueue_command == "embed_source"
    assert by_name[StageName.EXTRACT].depends_on is StageName.EMBED
    assert by_name[StageName.EXTRACT].produces == "extracted"
    assert by_name[StageName.EXTRACT].enqueue_command == "run_entities"
    assert by_name[StageName.EXTRACT].gate == "schema_review"
    assert by_name[StageName.GRAPH].depends_on is StageName.EXTRACT
    assert by_name[StageName.GRAPH].produces == "graphed"
    # GRAPH is inline (no job), every spine stage is auto.
    assert by_name[StageName.GRAPH].enqueue_command is None
    assert all(by_name[s].auto for s in by_name)

    # INSIGHTS is a PARALLEL branch off EMBED, toggle-gated, sets no stage value.
    insights = by_name[StageName.INSIGHTS]
    assert insights.depends_on is StageName.EMBED
    assert insights.parallel is True
    assert insights.produces is None
    assert insights.gate == "auto_insights"
    assert insights.enqueue_command == "run_summaries"


# ---------------------------------------------------------------------------
# advance_source dispatch table — one helper to drive it with all the seams
# mocked (no DB).
# ---------------------------------------------------------------------------


def _drive(
    *,
    stage: str,
    notebook_id=None,
    auto_insights=True,
    submit_return="job:x",
    submit_side_effect=None,
    refresh_side_effect=None,
):
    """Build the patch stack for an ``advance_source`` call.

    Returns ``(ctx, submit_mock, mentions_mock, set_stage_mock)`` — ``ctx`` is an
    ``ExitStack``-like context to ``with``. The source repo's
    ``get_processing_stage`` returns ``stage``; ``get_notebook_id`` /
    ``get_auto_insights`` drive the INSIGHTS toggle; ``refresh_source`` /
    ``submit_command_job`` are spies.
    """
    import contextlib

    src_repo = AsyncMock()
    src_repo.get_processing_stage = AsyncMock(return_value=stage)
    src_repo.set_processing_stage = AsyncMock(return_value=True)
    src_repo.get_notebook_id = AsyncMock(return_value=notebook_id)

    nb_repo = AsyncMock()
    nb_repo.get_auto_insights = AsyncMock(return_value=auto_insights)

    mentions = AsyncMock()
    if refresh_side_effect is not None:
        mentions.refresh_source = AsyncMock(side_effect=refresh_side_effect)
    else:
        mentions.refresh_source = AsyncMock(
            return_value=_Refresh(cleared=0, created=2, failed=0)
        )

    submit = AsyncMock(
        return_value=submit_return, side_effect=submit_side_effect
    )

    stack = contextlib.ExitStack()
    stack.enter_context(
        patch("app_main.dependencies.get_source_repo", return_value=src_repo)
    )
    stack.enter_context(
        patch("app_main.dependencies.get_notebook_repo", return_value=nb_repo)
    )
    stack.enter_context(
        patch(
            "app_main.dependencies.get_mentions_projection_service",
            return_value=mentions,
        )
    )
    stack.enter_context(
        patch(
            "app_main.services.command_service.CommandService.submit_command_job",
            new=submit,
        )
    )
    return stack, submit, mentions, src_repo.set_processing_stage


class _Refresh:
    def __init__(self, *, cleared, created, failed):
        self.cleared = cleared
        self.created = created
        self.failed = failed


def _commands(submit) -> list[str]:
    return [c.args[1] for c in submit.await_args_list]


# --- ingested -> EMBED -------------------------------------------------------


async def test_ingested_advances_to_embed() -> None:
    ctx, submit, _, _ = _drive(stage="ingested")
    with ctx:
        result = await advance_source("source:s1")
    assert _commands(submit) == ["embed_source"]
    assert result.enqueued["embed"] == "job:x"
    assert submit.await_args.args[2] == {"source_id": "source:s1"}


# --- embedded -> EXTRACT + parallel INSIGHTS --------------------------------


async def test_embedded_fans_out_to_extract_and_insights() -> None:
    # Linked notebook, toggle ON -> both run_entities AND run_summaries.
    ctx, submit, _, _ = _drive(
        stage="embedded",
        notebook_id="notebook:nb1",
        auto_insights=True,
    )
    with ctx:
        result = await advance_source("source:s1", embedded_chunks=5)
    commands = _commands(submit)
    assert "run_entities" in commands
    assert "run_summaries" in commands
    assert result.enqueued["extract"] == "job:x"
    assert result.enqueued["insights"] == "job:x"


async def test_embedded_insights_skipped_when_toggle_off() -> None:
    ctx, submit, _, _ = _drive(
        stage="embedded",
        notebook_id="notebook:nb1",
        auto_insights=False,
    )
    with ctx:
        result = await advance_source("source:s1", embedded_chunks=5)
    commands = _commands(submit)
    # The KG (free) chain still fires; insights is suppressed by the toggle.
    assert "run_entities" in commands
    assert "run_summaries" not in commands
    assert result.enqueued["insights"] is None


async def test_embedded_insights_default_on_for_unlinked_source() -> None:
    # No notebook -> no toggle consulted -> insights default ON.
    ctx, submit, _, _ = _drive(stage="embedded", notebook_id=None)
    with ctx:
        result = await advance_source("source:s1", embedded_chunks=5)
    assert "run_summaries" in _commands(submit)
    assert result.enqueued["insights"] == "job:x"


async def test_embedded_zero_chunks_suppresses_both() -> None:
    # The chunk-count guard: a zero-chunk embed chains nothing (no no-op jobs).
    ctx, submit, _, _ = _drive(stage="embedded", notebook_id="notebook:nb1")
    with ctx:
        result = await advance_source("source:s1", embedded_chunks=0)
    submit.assert_not_awaited()
    assert result.skipped["extract"] == "0 embedded chunks"
    assert result.skipped["insights"] == "0 embedded chunks"


# --- extracted -> GRAPH inline -> graphed -> complete -----------------------


async def test_extracted_runs_graph_inline_then_completes() -> None:
    ctx, submit, mentions, set_stage = _drive(stage="extracted")
    with ctx:
        result = await advance_source("source:s1")
    # GRAPH runs inline (no job enqueued for the spine successor).
    mentions.refresh_source.assert_awaited_once_with("source:s1")
    assert "graph" in result.inline_ran
    stages = [c.args for c in set_stage.await_args_list]
    assert stages == [("source:s1", "graphed"), ("source:s1", "complete")]
    # No EMBED/EXTRACT enqueue from the extracted stage.
    assert "embed_source" not in _commands(submit)


async def test_graph_refresh_failure_keeps_extracted() -> None:
    # Best-effort: a projection failure must not raise; the stage stays extracted
    # (no graphed/complete) and the result records the inline stage did not run.
    ctx, _, mentions, set_stage = _drive(
        stage="extracted", refresh_side_effect=RuntimeError("graph boom")
    )
    with ctx:
        result = await advance_source("source:s1")
    assert result.inline_ran == []  # GRAPH did not complete
    set_stage.assert_not_awaited()  # no graphed/complete written


# --- parked / terminal stages do not auto-advance ---------------------------


@pytest.mark.parametrize("stage", ["awaiting_schema_review", "failed", "complete"])
async def test_parked_and_terminal_stages_do_not_advance(stage: str) -> None:
    ctx, submit, mentions, set_stage = _drive(stage=stage)
    with ctx:
        result = await advance_source("source:s1")
    submit.assert_not_awaited()
    mentions.refresh_source.assert_not_awaited()
    set_stage.assert_not_awaited()
    assert result.skipped["*"].startswith(f"stage '{stage}'")


async def test_unknown_stage_does_not_advance() -> None:
    ctx, submit, _, _ = _drive(stage="bogus")
    with ctx:
        result = await advance_source("source:s1")
    submit.assert_not_awaited()
    assert "unknown stage" in result.skipped["*"]


# --- best-effort enqueue: a queue hiccup never raises -----------------------


async def test_enqueue_failure_is_best_effort() -> None:
    ctx, _, _, _ = _drive(
        stage="ingested", submit_side_effect=RuntimeError("queue down")
    )
    with ctx:
        result = await advance_source("source:s1")  # must not raise
    assert result.enqueued["embed"] is None
