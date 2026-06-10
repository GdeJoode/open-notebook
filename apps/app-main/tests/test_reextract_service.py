"""Unit tests for the Phase B.3d ReextractService.

Covers:

* Happy path — every source ends up as a submitted entity_extract job.
* Dedup — sources whose latest entity_extract job is in-flight are
  skipped silently and excluded from ``jobs_enqueued``.
* Resilience — a failure submitting one source does not abort the batch.
* Empty input — returns zero counts with no error.

The service is purposely thin (one public method); we exercise the
collaboration with the ``CommandService``-like protocol + the
``JobRepository``-like ``get_by_source`` slice via AsyncMock.
"""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from app_main.services.reextract_service import (
    ReextractResult,
    ReextractService,
)


NOTEBOOK_ID = "notebook:reextract-fixture"


def _job(job_type: str, status: str, source_id: str = "source:abc") -> MagicMock:
    """Build a Job-shaped stub the service can consume.

    The service supports both Pydantic models and dicts; we use
    MagicMock with the relevant attributes so the test reads naturally
    while still exercising the attribute-access path.
    """
    job = MagicMock()
    job.job_type = job_type
    job.status = status
    job.source_id = source_id
    return job


@pytest.fixture
def command_service() -> AsyncMock:
    svc = AsyncMock()
    # submit_command_job returns a synthetic job id.
    counter = {"n": 0}

    async def _submit(*_args, **_kwargs) -> str:
        counter["n"] += 1
        return f"job:fake-{counter['n']}"

    svc.submit_command_job = AsyncMock(side_effect=_submit)
    return svc


@pytest.fixture
def job_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.get_by_source = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def service(command_service: AsyncMock, job_repo: AsyncMock) -> ReextractService:
    return ReextractService(command_service=command_service, job_repo=job_repo)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_enqueues_one_job_per_source(
        self,
        service: ReextractService,
        command_service: AsyncMock,
        job_repo: AsyncMock,
    ) -> None:
        result = await service.enqueue_reextract_jobs(
            NOTEBOOK_ID,
            ["source:a", "source:b", "source:c"],
        )

        assert isinstance(result, ReextractResult)
        assert result.jobs_enqueued == 3
        assert result.source_ids == ["source:a", "source:b", "source:c"]
        assert result.enqueued_source_ids == ["source:a", "source:b", "source:c"]
        assert result.skipped_source_ids == []
        assert command_service.submit_command_job.await_count == 3

    @pytest.mark.asyncio
    async def test_payload_shape_matches_run_entities(
        self,
        service: ReextractService,
        command_service: AsyncMock,
    ) -> None:
        await service.enqueue_reextract_jobs(NOTEBOOK_ID, ["source:x"])
        assert command_service.submit_command_job.await_count == 1
        args, kwargs = command_service.submit_command_job.call_args
        # ``submit_command_job(module_name, command_name, command_args)``
        assert args[0] == "open_notebook"
        assert args[1] == "run_entities"
        cmd_args = args[2]
        assert cmd_args["source_id"] == "source:x"
        assert cmd_args["multi_schema_enabled"] is True

    @pytest.mark.asyncio
    async def test_preserves_source_order(
        self,
        service: ReextractService,
    ) -> None:
        ordered: List[str] = ["source:z", "source:m", "source:a"]
        result = await service.enqueue_reextract_jobs(NOTEBOOK_ID, ordered)
        assert result.source_ids == ordered
        assert result.enqueued_source_ids == ordered


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    @pytest.mark.asyncio
    async def test_empty_source_list_returns_zero_counts(
        self,
        service: ReextractService,
        command_service: AsyncMock,
    ) -> None:
        result = await service.enqueue_reextract_jobs(NOTEBOOK_ID, [])
        assert result.jobs_enqueued == 0
        assert result.source_ids == []
        assert result.enqueued_source_ids == []
        assert result.skipped_source_ids == []
        command_service.submit_command_job.assert_not_awaited()


# ---------------------------------------------------------------------------
# Idempotency / dedup
# ---------------------------------------------------------------------------


class TestDedup:
    @pytest.mark.asyncio
    async def test_skips_source_with_inflight_queued_job(
        self,
        service: ReextractService,
        command_service: AsyncMock,
        job_repo: AsyncMock,
    ) -> None:
        # source:a has a queued entity_extract job, source:b does not.
        async def _by_source(source_id: str):
            if source_id == "source:a":
                return [_job("entity_extract", "queued")]
            return []

        job_repo.get_by_source.side_effect = _by_source

        result = await service.enqueue_reextract_jobs(
            NOTEBOOK_ID, ["source:a", "source:b"]
        )

        assert result.jobs_enqueued == 1
        assert result.enqueued_source_ids == ["source:b"]
        assert result.skipped_source_ids == ["source:a"]
        # Only one actual submit call.
        assert command_service.submit_command_job.await_count == 1

    @pytest.mark.asyncio
    async def test_skips_source_with_processing_job(
        self,
        service: ReextractService,
        command_service: AsyncMock,
        job_repo: AsyncMock,
    ) -> None:
        job_repo.get_by_source.return_value = [
            _job("entity_extract", "processing")
        ]
        result = await service.enqueue_reextract_jobs(
            NOTEBOOK_ID, ["source:a"]
        )
        assert result.jobs_enqueued == 0
        assert result.skipped_source_ids == ["source:a"]
        command_service.submit_command_job.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_source_with_retrying_job(
        self,
        service: ReextractService,
        command_service: AsyncMock,
        job_repo: AsyncMock,
    ) -> None:
        job_repo.get_by_source.return_value = [
            _job("entity_extract", "retrying")
        ]
        result = await service.enqueue_reextract_jobs(NOTEBOOK_ID, ["source:a"])
        assert result.jobs_enqueued == 0
        assert result.skipped_source_ids == ["source:a"]
        command_service.submit_command_job.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_completed_job_does_not_dedup(
        self,
        service: ReextractService,
        command_service: AsyncMock,
        job_repo: AsyncMock,
    ) -> None:
        # A previously-completed job should not block a re-extract — that
        # IS the whole point of the banner.
        job_repo.get_by_source.return_value = [
            _job("entity_extract", "completed")
        ]
        result = await service.enqueue_reextract_jobs(
            NOTEBOOK_ID, ["source:a"]
        )
        assert result.jobs_enqueued == 1
        assert result.enqueued_source_ids == ["source:a"]
        command_service.submit_command_job.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failed_job_does_not_dedup(
        self,
        service: ReextractService,
        command_service: AsyncMock,
        job_repo: AsyncMock,
    ) -> None:
        job_repo.get_by_source.return_value = [
            _job("entity_extract", "failed")
        ]
        result = await service.enqueue_reextract_jobs(
            NOTEBOOK_ID, ["source:a"]
        )
        assert result.jobs_enqueued == 1

    @pytest.mark.asyncio
    async def test_inflight_job_of_different_type_does_not_dedup(
        self,
        service: ReextractService,
        command_service: AsyncMock,
        job_repo: AsyncMock,
    ) -> None:
        # Another job type (e.g. document_parse) being in-flight must not
        # block an entity_extract re-run.
        job_repo.get_by_source.return_value = [
            _job("document_parse", "processing")
        ]
        result = await service.enqueue_reextract_jobs(
            NOTEBOOK_ID, ["source:a"]
        )
        assert result.jobs_enqueued == 1
        command_service.submit_command_job.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_repeated_call_is_idempotent(
        self,
        service: ReextractService,
        command_service: AsyncMock,
        job_repo: AsyncMock,
    ) -> None:
        # Simulate the real DB: after the first submit, the source has an
        # in-flight job; the second call should dedup it.
        state = {"submitted": False}

        async def _by_source(source_id: str):
            if state["submitted"]:
                return [_job("entity_extract", "queued")]
            return []

        async def _submit(*_args, **_kwargs) -> str:
            state["submitted"] = True
            return "job:fake-1"

        job_repo.get_by_source.side_effect = _by_source
        command_service.submit_command_job.side_effect = _submit

        first = await service.enqueue_reextract_jobs(NOTEBOOK_ID, ["source:a"])
        second = await service.enqueue_reextract_jobs(NOTEBOOK_ID, ["source:a"])

        assert first.jobs_enqueued == 1
        assert second.jobs_enqueued == 0
        assert second.skipped_source_ids == ["source:a"]
        # submit_command_job called once across both calls.
        assert command_service.submit_command_job.await_count == 1


# ---------------------------------------------------------------------------
# Resilience: per-source failures do not abort the batch
# ---------------------------------------------------------------------------


class TestResilience:
    @pytest.mark.asyncio
    async def test_submit_failure_treated_as_skip_and_does_not_abort(
        self,
        service: ReextractService,
        command_service: AsyncMock,
    ) -> None:
        call_log: List[str] = []

        async def _submit(_mod, _cmd, args, *_):
            sid = args["source_id"]
            call_log.append(sid)
            if sid == "source:b":
                raise RuntimeError("transient queue hiccup")
            return f"job:fake-{sid}"

        command_service.submit_command_job.side_effect = _submit

        result = await service.enqueue_reextract_jobs(
            NOTEBOOK_ID,
            ["source:a", "source:b", "source:c"],
        )

        # All three sources were attempted; only a + c succeeded.
        assert call_log == ["source:a", "source:b", "source:c"]
        assert result.jobs_enqueued == 2
        assert result.enqueued_source_ids == ["source:a", "source:c"]
        assert result.skipped_source_ids == ["source:b"]

    @pytest.mark.asyncio
    async def test_get_by_source_failure_does_not_block_submit(
        self,
        service: ReextractService,
        command_service: AsyncMock,
        job_repo: AsyncMock,
    ) -> None:
        # If we can't read the queue we'd rather risk a duplicate than
        # refuse to re-extract.
        job_repo.get_by_source.side_effect = RuntimeError("db down")

        result = await service.enqueue_reextract_jobs(
            NOTEBOOK_ID,
            ["source:a"],
        )
        assert result.jobs_enqueued == 1
        command_service.submit_command_job.assert_awaited_once()
