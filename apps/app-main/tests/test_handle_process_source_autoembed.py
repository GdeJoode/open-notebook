"""Track R.0 forward-fix: ``DOCUMENT_PARSE`` auto-enqueues ``embed_source``.

Before R.0 a freshly ingested source had chunks but no ``source_embedding``
rows until a manual ``run-embed``. The ``handle_process_source`` job handler
now enqueues the existing ``embed_source`` job after a successful extraction
that produced chunks. These DB-free tests assert that seam:

  * chunks produced  -> exactly one ``embed_source`` job enqueued for the source;
  * zero chunks      -> no embed job (nothing to embed);
  * enqueue failure  -> extraction result still returned (best-effort, never
    fails the already-persisted ingest).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import app_main.handlers as handlers


@pytest.mark.asyncio
async def test_autoembed_enqueued_when_chunks_created() -> None:
    processor = AsyncMock()
    processor.process_source = AsyncMock(
        return_value={"source_id": "source:abc", "chunk_count": 5}
    )

    with _patch_dep(processor), patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(return_value="job:embed-1"),
    ) as submit:
        result = await handlers.handle_process_source(
            {"source_id": "source:abc", "content_state": {"content": "hi"}}
        )

    assert result["success"] is True
    assert result["chunk_count"] == 5
    assert result["embed_command_id"] == "job:embed-1"
    submit.assert_awaited_once()
    args = submit.await_args.args
    # submit_command_job(module_name, command_name, command_args)
    assert args[1] == "embed_source"
    assert args[2] == {"source_id": "source:abc"}


@pytest.mark.asyncio
async def test_no_autoembed_when_zero_chunks() -> None:
    processor = AsyncMock()
    processor.process_source = AsyncMock(
        return_value={"source_id": "source:empty", "chunk_count": 0}
    )

    with _patch_dep(processor), patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(),
    ) as submit:
        result = await handlers.handle_process_source(
            {"source_id": "source:empty", "content_state": {"content": ""}}
        )

    assert result["success"] is True
    assert result["chunk_count"] == 0
    assert result["embed_command_id"] is None
    submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_enqueue_failure_does_not_fail_extraction() -> None:
    processor = AsyncMock()
    processor.process_source = AsyncMock(
        return_value={"source_id": "source:abc", "chunk_count": 3}
    )

    with _patch_dep(processor), patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(side_effect=RuntimeError("queue down")),
    ):
        result = await handlers.handle_process_source(
            {"source_id": "source:abc", "content_state": {"content": "hi"}}
        )

    # Extraction already persisted chunks — the result must still come back.
    assert result["success"] is True
    assert result["chunk_count"] == 3
    assert result["embed_command_id"] is None


def _patch_dep(processor):
    """Patch the lazily-imported ``get_source_processor`` factory.

    ``handle_process_source`` imports ``get_source_processor`` from
    ``app_main.dependencies`` at call time, so we patch it at the source module.
    """
    return patch(
        "app_main.dependencies.get_source_processor", return_value=processor
    )
