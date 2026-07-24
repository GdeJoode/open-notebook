"""Track V.5: the guarded, config-gated reference post-ingest hook (DB-free).

Pins the pipeline seam that enqueues the V.5 reference pass:

  * the REFERENCES stage is a PARALLEL branch off INGEST, gated ``references``,
    enqueuing ``extract_references``, setting no ``processing_stage``;
  * FLAG OFF (the default) -> advancing off ``ingested`` enqueues ONLY the
    pre-V.5 EMBED job — the un-flagged ingest path is byte-for-byte unchanged;
  * FLAG ON -> advancing off ``ingested`` ALSO enqueues ``extract_references``
    (the job seam — asserted at the ``submit_command_job`` boundary, per the
    repo's job-queue-singleton test pattern);
  * the hook is best-effort: a raising enqueue never fails the ingest step
    (``handle_process_source`` still returns success).
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, patch

import app_main.handlers as handlers
import pytest
from app_main.services.source_pipeline import (
    SOURCE_PIPELINE,
    StageName,
    advance_source,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# The stage definition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope=None)
async def test_references_stage_definition() -> None:
    """Structural assertions on the REFERENCES parallel branch (no IO)."""
    by_name = {s.name: s for s in SOURCE_PIPELINE}
    refs = by_name[StageName.REFERENCES]
    assert refs.depends_on is StageName.INGEST
    assert refs.parallel is True
    # Enrichment: it advances no spine stage value, so it never gates complete.
    assert refs.produces is None
    assert refs.gate == "references"
    assert refs.enqueue_command == "extract_references"
    assert refs.auto is True


# ---------------------------------------------------------------------------
# advance_source dispatch — flag OFF vs ON
# ---------------------------------------------------------------------------


def _drive(*, flag_on: bool, submit_side_effect=None, submit_return="job:x"):
    """Patch stack for an ``advance_source`` off ``ingested`` with the V.5 flag
    forced ``flag_on``. Returns ``(ctx, submit_mock)``."""
    src_repo = AsyncMock()
    src_repo.get_processing_stage = AsyncMock(return_value="ingested")
    src_repo.set_processing_stage = AsyncMock(return_value=True)
    src_repo.get_notebook_id = AsyncMock(return_value=None)

    submit = AsyncMock(return_value=submit_return, side_effect=submit_side_effect)

    stack = contextlib.ExitStack()
    stack.enter_context(
        patch("app_main.dependencies.get_source_repo", return_value=src_repo)
    )
    stack.enter_context(
        patch(
            "app_main.services.command_service.CommandService.submit_command_job",
            new=submit,
        )
    )
    stack.enter_context(
        patch(
            "app_main.config.get_reference_extraction_enabled",
            return_value=flag_on,
        )
    )
    return stack, submit


def _commands(submit) -> list[str]:
    return [c.args[1] for c in submit.await_args_list]


async def test_ingested_flag_off_enqueues_only_embed() -> None:
    # DEFAULT (flag OFF): byte-for-byte the pre-V.5 path — only embed_source.
    ctx, submit = _drive(flag_on=False)
    with ctx:
        result = await advance_source("source:s1")
    assert _commands(submit) == ["embed_source"]
    # The branch is recorded as a no-op (enqueued None), not enqueued.
    assert result.enqueued.get("references") is None


async def test_ingested_flag_on_also_enqueues_references() -> None:
    ctx, submit = _drive(flag_on=True)
    with ctx:
        result = await advance_source("source:s1")
    commands = _commands(submit)
    assert "embed_source" in commands  # spine successor unchanged
    assert "extract_references" in commands  # the V.5 parallel branch
    assert result.enqueued["references"] == "job:x"
    # The reference job is submitted for the triggering source.
    ref_call = next(
        c for c in submit.await_args_list if c.args[1] == "extract_references"
    )
    assert ref_call.args[2] == {"source_id": "source:s1"}


async def test_references_enqueue_failure_is_best_effort() -> None:
    # A queue hiccup on the reference enqueue must not raise out of advance_source.
    ctx, _ = _drive(flag_on=True, submit_side_effect=RuntimeError("queue down"))
    with ctx:
        result = await advance_source("source:s1")  # must not raise
    assert result.enqueued["references"] is None


# ---------------------------------------------------------------------------
# The ingest step survives a failing reference hook
# ---------------------------------------------------------------------------


async def test_failing_reference_hook_does_not_fail_ingest() -> None:
    """The guard AC: a raising reference enqueue inside the post-ingest hook does
    NOT fail the ingest step — ``handle_process_source`` still returns success."""
    processor = AsyncMock()
    processor.process_source = AsyncMock(
        return_value={"source_id": "source:abc", "chunk_count": 4}
    )

    src_repo = AsyncMock()
    src_repo.get_processing_stage = AsyncMock(return_value="ingested")
    src_repo.set_processing_stage = AsyncMock(return_value=True)
    src_repo.get_notebook_id = AsyncMock(return_value=None)

    with patch(
        "app_main.dependencies.get_source_processor", return_value=processor
    ), patch(
        "app_main.dependencies.get_source_repo", return_value=src_repo
    ), patch(
        "app_main.config.get_reference_extraction_enabled", return_value=True
    ), patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        # Fail the reference enqueue; the embed enqueue would also route here,
        # so fail everything and assert ingest STILL succeeds (best-effort).
        new=AsyncMock(side_effect=RuntimeError("queue down")),
    ):
        result = await handlers.handle_process_source(
            {"source_id": "source:abc", "content_state": {"content": "hi"}}
        )

    # Ingest already persisted chunks — the result comes back despite the hook.
    assert result["success"] is True
    assert result["chunk_count"] == 4
