"""Tests for the job service."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.models.jobs import Job, JobStatsResponse, JobSubmitRequest
from shared.types.enums import JobPriority, JobStatus, JobType

from job_queue.queue import JobQueue
from job_queue.repository import JobRepository
from job_queue.service import JobService
from job_queue.worker import JobWorker


def _make_job(
    job_id: str = "job:svc1",
    status: JobStatus = JobStatus.QUEUED,
    **kwargs,
) -> Job:
    return Job(
        id=job_id,
        job_type=kwargs.get("job_type", JobType.DOCUMENT_PARSE),
        status=status,
        priority=kwargs.get("priority", JobPriority.NORMAL),
        payload=kwargs.get("payload", {}),
        retry_count=kwargs.get("retry_count", 0),
        max_retries=kwargs.get("max_retries", 2),
    )


class TestJobService:
    """Test service-level operations."""

    @pytest.fixture
    def repository(self):
        repo = MagicMock(spec=JobRepository)
        repo.create_from_submission = AsyncMock()
        repo.get = AsyncMock()
        repo.update_status = AsyncMock()
        repo.list_jobs = AsyncMock(return_value=[])
        repo.get_stats = AsyncMock(return_value={"total": 0})
        repo.get_retryable = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def queue(self):
        return JobQueue(max_size=10)

    @pytest.fixture
    def worker(self):
        w = MagicMock(spec=JobWorker)
        w.start = AsyncMock()
        w.stop = AsyncMock()
        return w

    @pytest.fixture
    def service(self, repository, queue, worker):
        return JobService(repository, queue, worker)

    @pytest.mark.asyncio
    async def test_submit(self, service, repository, queue):
        """Submit should create a DB record and enqueue."""
        job = _make_job()
        repository.create_from_submission.return_value = job

        submission = JobSubmitRequest(
            job_type=JobType.DOCUMENT_PARSE,
            payload={"file": "test.pdf"},
        )
        result = await service.submit(submission)

        assert result.id == "job:svc1"
        repository.create_from_submission.assert_called_once_with(submission)
        assert queue.size() == 1

    @pytest.mark.asyncio
    async def test_get_status(self, service, repository):
        job = _make_job(status=JobStatus.PROCESSING)
        repository.get.return_value = job

        result = await service.get_status("job:svc1")
        assert result.status == JobStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_get_status_not_found(self, service, repository):
        repository.get.return_value = None
        result = await service.get_status("job:missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_jobs(self, service, repository):
        jobs = [_make_job("job:1"), _make_job("job:2")]
        repository.list_jobs.return_value = jobs

        result = await service.list_jobs(status=JobStatus.QUEUED)
        assert len(result) == 2
        repository.list_jobs.assert_called_once_with(
            status=JobStatus.QUEUED, job_type=None, limit=50
        )

    @pytest.mark.asyncio
    async def test_get_stats(self, service, repository):
        repository.get_stats.return_value = {
            "total": 10,
            "queued": 3,
            "processing": 2,
            "completed": 4,
            "failed": 1,
            "cancelled": 0,
            "retrying": 0,
        }
        stats = await service.get_stats()
        assert isinstance(stats, JobStatsResponse)
        assert stats.total == 10
        assert stats.queued == 3
        assert stats.failed == 1

    @pytest.mark.asyncio
    async def test_cancel_queued_job(self, service, repository):
        job = _make_job(status=JobStatus.QUEUED)
        repository.get.return_value = job
        repository.update_status.return_value = job

        result = await service.cancel("job:svc1")
        assert result is True
        repository.update_status.assert_called_once_with("job:svc1", JobStatus.CANCELLED)

    @pytest.mark.asyncio
    async def test_cancel_processing_job_fails(self, service, repository):
        job = _make_job(status=JobStatus.PROCESSING)
        repository.get.return_value = job

        result = await service.cancel("job:svc1")
        assert result is False
        repository.update_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_missing_job(self, service, repository):
        repository.get.return_value = None
        result = await service.cancel("job:missing")
        assert result is False

    @pytest.mark.asyncio
    async def test_retry_failed(self, service, repository, queue):
        failed_jobs = [
            _make_job("job:f1", status=JobStatus.FAILED, retry_count=0),
            _make_job("job:f2", status=JobStatus.FAILED, retry_count=1),
        ]
        repository.get_retryable.return_value = failed_jobs
        repository.update_status.return_value = _make_job(status=JobStatus.RETRYING)

        count = await service.retry_failed()
        assert count == 2
        assert queue.size() == 2

    @pytest.mark.asyncio
    async def test_retry_failed_none(self, service, repository, queue):
        repository.get_retryable.return_value = []
        count = await service.retry_failed()
        assert count == 0
        assert queue.is_empty()

    @pytest.mark.asyncio
    async def test_start_stop(self, service, worker):
        await service.start()
        worker.start.assert_called_once()
        await service.stop()
        worker.stop.assert_called_once()
