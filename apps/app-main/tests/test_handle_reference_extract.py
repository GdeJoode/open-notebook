"""Track V.5: the ``REFERENCE_EXTRACT`` job handler + the command routing seam.

DB-free tests of the consumer side of the reference pass:

  * the ``extract_references`` command routes to ``JobType.REFERENCE_EXTRACT``
    (the enqueue seam — module-level singletons use the env DB, so we assert the
    routing table, not a live submit);
  * the handler runs the orchestration service and returns the flattened summary;
  * ``resolve_external`` threads through the payload to the service factory + call;
  * a hard failure in the pass raises (the worker records the job FAILED) —
    isolated from the already-persisted ingest that enqueued it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import app_main.handlers as handlers
import pytest
from app_main.services.command_service import _COMMAND_TO_JOB_TYPE
from shared.types.enums import JobType

# ---------------------------------------------------------------------------
# routing seam
# ---------------------------------------------------------------------------


def test_extract_references_command_routes_to_reference_extract() -> None:
    assert _COMMAND_TO_JOB_TYPE["extract_references"] is JobType.REFERENCE_EXTRACT


@pytest.mark.asyncio
async def test_enqueue_seam_maps_command_to_job_type() -> None:
    """The job-queue-singleton seam: assert ENQUEUE routes ``extract_references``
    to ``REFERENCE_EXTRACT`` by mocking the underlying JobService.submit (the
    module-level singleton uses the env DB, so we assert at the seam)."""
    from app_main.services.command_service import CommandService

    submitted = SimpleNamespace(id="job:ref-1")
    with patch(
        "app_main.services.command_service._service.submit",
        new=AsyncMock(return_value=submitted),
    ) as submit:
        job_id = await CommandService.submit_command_job(
            "open_notebook", "extract_references", {"source_id": "source:s1"}
        )

    assert job_id == "job:ref-1"
    submission = submit.await_args.args[0]
    assert submission.job_type is JobType.REFERENCE_EXTRACT
    assert submission.source_id == "source:s1"
    assert submission.payload["command_name"] == "extract_references"


# ---------------------------------------------------------------------------
# the handler
# ---------------------------------------------------------------------------


def _summary(**over):
    base = dict(
        sources_scanned=3,
        sources_with_references=2,
        refs_extracted=7,
        edges_materialized=1,
        external_resolved=0,
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_handler_runs_pass_and_returns_summary() -> None:
    service = AsyncMock()
    service.materialize_corpus = AsyncMock(return_value=_summary())

    with patch(
        "app_main.dependencies.get_reference_extraction_service",
        return_value=service,
    ) as factory:
        result = await handlers.handle_reference_extract(
            {"source_id": "source:s1"}
        )

    # Default: resolve_external OFF (offline core path).
    factory.assert_called_once_with(resolve_external=False)
    service.materialize_corpus.assert_awaited_once_with(resolve_external=False)
    assert result["success"] is True
    assert result["sources_scanned"] == 3
    assert result["sources_with_references"] == 2
    assert result["refs_extracted"] == 7
    assert result["edges_materialized"] == 1
    assert result["external_resolved"] == 0


@pytest.mark.asyncio
async def test_handler_threads_resolve_external_flag() -> None:
    service = AsyncMock()
    service.materialize_corpus = AsyncMock(
        return_value=_summary(external_resolved=4)
    )

    with patch(
        "app_main.dependencies.get_reference_extraction_service",
        return_value=service,
    ) as factory:
        result = await handlers.handle_reference_extract(
            {"source_id": "source:s1", "resolve_external": True}
        )

    factory.assert_called_once_with(resolve_external=True)
    service.materialize_corpus.assert_awaited_once_with(resolve_external=True)
    assert result["external_resolved"] == 4


@pytest.mark.asyncio
async def test_handler_reraises_on_hard_failure() -> None:
    # A hard failure raises so the worker marks the job FAILED (isolated from the
    # already-persisted ingest that enqueued it).
    service = AsyncMock()
    service.materialize_corpus = AsyncMock(side_effect=RuntimeError("boom"))

    with patch(
        "app_main.dependencies.get_reference_extraction_service",
        return_value=service,
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await handlers.handle_reference_extract({"source_id": "source:s1"})
