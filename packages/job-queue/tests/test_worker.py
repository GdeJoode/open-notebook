"""Tests for the job worker."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.models.jobs import Job
from shared.types.enums import JobPriority, JobStatus, JobType

from job_queue.queue import JobQueue
from job_queue.registry import HandlerRegistry
from job_queue.repository import JobRepository
from job_queue.worker import JobWorker


def _make_job(
    job_id: str = "job:test1",
    job_type: JobType = JobType.DOCUMENT_PARSE,
    status: JobStatus = JobStatus.QUEUED,
    retry_count: int = 0,
    max_retries: int = 2,
    priority: JobPriority = JobPriority.NORMAL,
) -> Job:
    """Create a fake Job instance for testing."""
    return Job(
        id=job_id,
        job_type=job_type,
        status=status,
        priority=priority,
        payload={"key": "value"},
        retry_count=retry_count,
        max_retries=max_retries,
    )


class TestJobWorker:
    """Test worker job processing logic."""

    @pytest.fixture
    def queue(self):
        return JobQueue(max_size=10)

    @pytest.fixture
    def repository(self):
        repo = MagicMock(spec=JobRepository)
        repo.get = AsyncMock()
        repo.update_status = AsyncMock()
        return repo

    @pytest.fixture
    def registry(self):
        return HandlerRegistry()

    @pytest.fixture
    def worker(self, queue, repository, registry):
        return JobWorker(queue, repository, registry)

    @pytest.mark.asyncio
    async def test_start_stop(self, worker):
        await worker.start()
        assert worker.is_running
        await worker.stop()
        assert not worker.is_running

    @pytest.mark.asyncio
    async def test_start_twice_is_safe(self, worker):
        await worker.start()
        await worker.start()  # should warn but not fail
        assert worker.is_running
        await worker.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, worker):
        await worker.stop()  # should be a no-op

    @pytest.mark.asyncio
    async def test_successful_job_execution(self, queue, repository, registry):
        """Worker should process a job and mark it COMPLETED."""
        job = _make_job()
        repository.get.return_value = job
        repository.update_status.return_value = job

        @registry.register(JobType.DOCUMENT_PARSE)
        async def handler(payload):
            return {"parsed": True}

        worker = JobWorker(queue, repository, registry)

        await queue.enqueue("job:test1")
        await worker.start()
        await asyncio.sleep(0.1)
        await worker.stop()

        # Should have been marked PROCESSING then COMPLETED
        calls = repository.update_status.call_args_list
        assert len(calls) >= 2
        assert calls[0].args == ("job:test1", JobStatus.PROCESSING)
        assert calls[1].args == ("job:test1", JobStatus.COMPLETED)
        assert calls[1].kwargs.get("result") == {"parsed": True}

    @pytest.mark.asyncio
    async def test_failed_job_with_retry(self, queue, repository, registry):
        """Failed job with retries left should be re-enqueued."""
        # Provide enough mock responses for all retry attempts:
        # attempt 1 (retry_count=0) -> retry, attempt 2 (retry_count=1) -> retry,
        # attempt 3 (retry_count=2) -> exhausted -> FAILED
        repository.get.side_effect = [
            _make_job(retry_count=0, max_retries=2),
            _make_job(retry_count=1, max_retries=2),
            _make_job(retry_count=2, max_retries=2),
        ]
        repository.update_status.return_value = _make_job()

        @registry.register(JobType.DOCUMENT_PARSE)
        async def handler(payload):
            raise RuntimeError("parse failed")

        worker = JobWorker(queue, repository, registry)

        await queue.enqueue("job:test1")
        await worker.start()
        await asyncio.sleep(0.3)
        await worker.stop()

        # First attempt: PROCESSING then RETRYING (retry_count 0->1)
        calls = repository.update_status.call_args_list
        assert calls[0].args == ("job:test1", JobStatus.PROCESSING)
        assert calls[1].args == ("job:test1", JobStatus.RETRYING)
        assert calls[1].kwargs.get("retry_count") == 1

    @pytest.mark.asyncio
    async def test_failed_job_exhausted_retries(self, queue, repository, registry):
        """Failed job with no retries left should be marked FAILED."""
        job = _make_job(retry_count=2, max_retries=2)
        repository.get.return_value = job
        repository.update_status.return_value = job

        @registry.register(JobType.DOCUMENT_PARSE)
        async def handler(payload):
            raise RuntimeError("parse failed again")

        worker = JobWorker(queue, repository, registry)

        await queue.enqueue("job:test1")
        await worker.start()
        await asyncio.sleep(0.1)
        await worker.stop()

        calls = repository.update_status.call_args_list
        assert calls[0].args == ("job:test1", JobStatus.PROCESSING)
        assert calls[1].args == ("job:test1", JobStatus.FAILED)
        assert "parse failed again" in calls[1].kwargs.get("error_message", "")

    @pytest.mark.asyncio
    async def test_cancelled_job_is_skipped(self, queue, repository, registry):
        """Cancelled jobs should not be executed."""
        job = _make_job(status=JobStatus.CANCELLED)
        repository.get.return_value = job

        @registry.register(JobType.DOCUMENT_PARSE)
        async def handler(payload):
            return {"should_not": "run"}

        worker = JobWorker(queue, repository, registry)

        await queue.enqueue("job:test1")
        await worker.start()
        await asyncio.sleep(0.1)
        await worker.stop()

        # Should NOT have been marked PROCESSING
        for call in repository.update_status.call_args_list:
            assert call.args[1] != JobStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_missing_job_is_skipped(self, queue, repository, registry):
        """If the job doesn't exist in DB, skip it."""
        repository.get.return_value = None

        worker = JobWorker(queue, repository, registry)

        await queue.enqueue("job:ghost")
        await worker.start()
        await asyncio.sleep(0.1)
        await worker.stop()

        repository.update_status.assert_not_called()
