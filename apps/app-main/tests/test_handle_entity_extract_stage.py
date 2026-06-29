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


def _patch_all(extraction_service, *, notebook_id=None):
    """Patch the three lazily-imported factories used by handle_entity_extract.

    Returns a context-manager stack plus the stage repo spy.
    """
    src_repo = AsyncMock()
    src_repo.get_notebook_id = AsyncMock(return_value=notebook_id)
    stage_repo = AsyncMock()
    stage_repo.set_processing_stage = AsyncMock(return_value=True)

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
    ]
    return patches, stage_repo


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
async def test_success_sets_stage_extracted() -> None:
    svc = AsyncMock()
    svc.run_extraction = AsyncMock(
        return_value={"source_id": "source:s1", "entity_count": 4, "relation_count": 2}
    )
    patches, stage_repo = _patch_all(svc)

    with patches[0], patches[1]:
        result = await handlers.handle_entity_extract({"source_id": "source:s1"})

    assert result["success"] is True
    assert result["entity_count"] == 4
    stage_repo.set_processing_stage.assert_awaited_once_with("source:s1", "extracted")


@pytest.mark.asyncio
async def test_schema_review_gate_sets_awaiting_and_reraises() -> None:
    svc = AsyncMock()
    svc.run_extraction = AsyncMock(
        side_effect=SchemaReviewPendingError(
            notebook_id="notebook:nb1", source_id="source:s1", pending_count=3
        )
    )
    patches, stage_repo = _patch_all(svc, notebook_id="notebook:nb1")

    with patches[0], patches[1]:
        with pytest.raises(SchemaReviewPendingError):
            await handlers.handle_entity_extract({"source_id": "source:s1"})

    # The gate parked extraction -> stage reflects awaiting_schema_review and the
    # exception reraises so the worker parks the job PAUSED_FOR_REVIEW.
    stage_repo.set_processing_stage.assert_awaited_once_with(
        "source:s1", "awaiting_schema_review"
    )


@pytest.mark.asyncio
async def test_hard_failure_sets_stage_failed_and_reraises() -> None:
    svc = AsyncMock()
    svc.run_extraction = AsyncMock(side_effect=RuntimeError("extract boom"))
    patches, stage_repo = _patch_all(svc)

    with patches[0], patches[1]:
        with pytest.raises(RuntimeError, match="extract boom"):
            await handlers.handle_entity_extract({"source_id": "source:s1"})

    stage_repo.set_processing_stage.assert_awaited_once_with("source:s1", "failed")
