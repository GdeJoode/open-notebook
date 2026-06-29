"""Track PL.2: ``handle_entity_extract`` advances ``source.processing_stage``
and respects the schema-review gate.

  * success                     -> stage = extracted, result returned;
  * SchemaReviewPendingError    -> stage = awaiting_schema_review, exception
    reraised (worker parks the job PAUSED_FOR_REVIEW, no entities written);
  * hard failure                -> stage = failed, exception reraised.

DB-free: the extraction service, the notebook resolver, and the stage repo are
all mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import app_main.handlers as handlers
from app_main.services.entity_extraction_service import SchemaReviewPendingError


def _patch_all(extraction_service, *, notebook_id=None, mentions_service=None):
    """Patch the lazily-imported factories used by handle_entity_extract.

    Returns a context-manager stack plus the stage repo spy. ``mentions_service``
    (PL.3) defaults to an AsyncMock whose ``refresh_source`` returns a small
    success result so the post-extract graph refresh + graphed/complete
    transitions run without a DB.
    """
    src_repo = AsyncMock()
    src_repo.get_notebook_id = AsyncMock(return_value=notebook_id)
    stage_repo = AsyncMock()
    stage_repo.set_processing_stage = AsyncMock(return_value=True)

    if mentions_service is None:
        mentions_service = AsyncMock()
        mentions_service.refresh_source = AsyncMock(
            return_value=_RefreshResult(cleared=0, created=3, failed=0)
        )

    patches = [
        patch(
            "app_main.dependencies.get_entity_extraction_service",
            return_value=extraction_service,
        ),
        # handle_entity_extract uses get_source_repo for BOTH notebook
        # resolution and (via _set_processing_stage) the stage write.
        patch(
            "app_main.dependencies.get_source_repo",
            side_effect=lambda: _RepoRouter(src_repo, stage_repo),
        ),
        # PL.3: the post-extract source-scoped mentions refresh.
        patch(
            "app_main.dependencies.get_mentions_projection_service",
            return_value=mentions_service,
        ),
    ]
    return patches, stage_repo, mentions_service


class _RefreshResult:
    """Stand-in for RegenerateMentionsResult (only the logged fields matter)."""

    def __init__(self, *, cleared: int, created: int, failed: int):
        self.cleared = cleared
        self.created = created
        self.failed = failed


class _RepoRouter:
    """Tiny shim: notebook resolution and stage writes both go through
    ``get_source_repo()``; route each call to the matching spy."""

    def __init__(self, src_repo, stage_repo):
        self._src = src_repo
        self._stage = stage_repo

    async def get_notebook_id(self, source_id):
        return await self._src.get_notebook_id(source_id)

    async def set_processing_stage(self, source_id, stage):
        return await self._stage.set_processing_stage(source_id, stage)


@pytest.mark.asyncio
async def test_success_sets_stage_extracted_then_graphed_complete() -> None:
    svc = AsyncMock()
    svc.run_extraction = AsyncMock(
        return_value={"source_id": "source:s1", "entity_count": 4, "relation_count": 2}
    )
    patches, stage_repo, mentions = _patch_all(svc)

    with patches[0], patches[1], patches[2]:
        result = await handlers.handle_entity_extract({"source_id": "source:s1"})

    assert result["success"] is True
    assert result["entity_count"] == 4
    # PL.3: a successful extract refreshes THIS source's mentions then advances
    # the stage extracted -> graphed -> complete (the KG chain is done).
    mentions.refresh_source.assert_awaited_once_with("source:s1")
    stages = [c.args for c in stage_repo.set_processing_stage.await_args_list]
    assert stages == [
        ("source:s1", "extracted"),
        ("source:s1", "graphed"),
        ("source:s1", "complete"),
    ]


@pytest.mark.asyncio
async def test_schema_review_gate_sets_awaiting_and_reraises() -> None:
    svc = AsyncMock()
    svc.run_extraction = AsyncMock(
        side_effect=SchemaReviewPendingError(
            notebook_id="notebook:nb1", source_id="source:s1", pending_count=3
        )
    )
    patches, stage_repo, mentions = _patch_all(svc, notebook_id="notebook:nb1")

    with patches[0], patches[1], patches[2]:
        with pytest.raises(SchemaReviewPendingError):
            await handlers.handle_entity_extract({"source_id": "source:s1"})

    # The gate parked extraction -> stage reflects awaiting_schema_review and the
    # exception reraises so the worker parks the job PAUSED_FOR_REVIEW. No graph
    # refresh happens on the gate path.
    stage_repo.set_processing_stage.assert_awaited_once_with(
        "source:s1", "awaiting_schema_review"
    )
    mentions.refresh_source.assert_not_awaited()


@pytest.mark.asyncio
async def test_hard_failure_sets_stage_failed_and_reraises() -> None:
    svc = AsyncMock()
    svc.run_extraction = AsyncMock(side_effect=RuntimeError("extract boom"))
    patches, stage_repo, mentions = _patch_all(svc)

    with patches[0], patches[1], patches[2]:
        with pytest.raises(RuntimeError, match="extract boom"):
            await handlers.handle_entity_extract({"source_id": "source:s1"})

    stage_repo.set_processing_stage.assert_awaited_once_with("source:s1", "failed")
    mentions.refresh_source.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_refresh_failure_keeps_stage_extracted() -> None:
    # PL.3 best-effort: a mentions-refresh failure must NOT fail the
    # (already-persisted) extraction; the stage stays at ``extracted`` and never
    # advances to graphed/complete.
    svc = AsyncMock()
    svc.run_extraction = AsyncMock(
        return_value={"source_id": "source:s1", "entity_count": 4, "relation_count": 2}
    )
    mentions = AsyncMock()
    mentions.refresh_source = AsyncMock(side_effect=RuntimeError("graph boom"))
    patches, stage_repo, _ = _patch_all(svc, mentions_service=mentions)

    with patches[0], patches[1], patches[2]:
        result = await handlers.handle_entity_extract({"source_id": "source:s1"})

    # Extraction still succeeds (best-effort graph refresh).
    assert result["success"] is True
    stages = [c.args for c in stage_repo.set_processing_stage.await_args_list]
    assert stages == [("source:s1", "extracted")]  # no graphed/complete
