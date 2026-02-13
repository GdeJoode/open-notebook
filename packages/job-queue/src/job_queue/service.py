"""
High-level job service — the main interface for consuming code.

Coordinates the repository (persistence), queue (dispatch), and worker
(execution) to provide a simple API for submitting and managing jobs.
"""

from typing import Dict, List, Optional

from loguru import logger

from shared.models import Job
from shared.models.jobs import JobStatsResponse, JobSubmitRequest
from shared.types.enums import JobStatus, JobType

from job_queue.queue import JobQueue
from job_queue.registry import HandlerRegistry
from job_queue.repository import JobRepository
from job_queue.worker import JobWorker


class JobService:
    """
    Public API for background job management.

    Usage::

        service = JobService(repository, queue, worker)
        job = await service.submit(JobSubmitRequest(
            job_type=JobType.DOCUMENT_PARSE,
            payload={"source_id": "source:abc"},
        ))
        status = await service.get_status(job.id)
    """

    def __init__(
        self,
        repository: JobRepository,
        queue: JobQueue,
        worker: JobWorker,
    ):
        self._repository = repository
        self._queue = queue
        self._worker = worker

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background worker."""
        await self._worker.start()

    async def stop(self) -> None:
        """Stop the background worker gracefully."""
        await self._worker.stop()

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    async def submit(self, submission: JobSubmitRequest) -> Job:
        """
        Create a DB record and enqueue the job. Returns immediately.

        Args:
            submission: Job submission details.

        Returns:
            The created Job record (status=QUEUED).
        """
        job = await self._repository.create_from_submission(submission)
        await self._queue.enqueue(job.id, submission.priority)
        logger.info(
            f"Submitted job {job.id} (type={submission.job_type.value}, "
            f"priority={submission.priority.value})"
        )
        return job

    # ------------------------------------------------------------------
    # Status & queries
    # ------------------------------------------------------------------

    async def get_status(self, job_id: str) -> Optional[Job]:
        """Get the current state of a job."""
        return await self._repository.get(job_id)

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        limit: int = 50,
    ) -> List[Job]:
        """List jobs with optional filters."""
        return await self._repository.list_jobs(
            status=status, job_type=job_type, limit=limit
        )

    async def get_stats(self) -> JobStatsResponse:
        """Get aggregated job statistics."""
        raw = await self._repository.get_stats()
        return JobStatsResponse(
            total=raw.get("total", 0),
            queued=raw.get(JobStatus.QUEUED.value, 0),
            processing=raw.get(JobStatus.PROCESSING.value, 0),
            completed=raw.get(JobStatus.COMPLETED.value, 0),
            failed=raw.get(JobStatus.FAILED.value, 0),
            cancelled=raw.get(JobStatus.CANCELLED.value, 0),
            retrying=raw.get(JobStatus.RETRYING.value, 0),
        )

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    async def cancel(self, job_id: str) -> bool:
        """
        Cancel a job. Only QUEUED or RETRYING jobs can be cancelled.

        Returns True if the job was cancelled, False otherwise.
        """
        job = await self._repository.get(job_id)
        if job is None:
            return False
        if job.status not in (JobStatus.QUEUED, JobStatus.RETRYING):
            logger.warning(
                f"Cannot cancel job {job_id} in status {job.status.value}"
            )
            return False
        await self._repository.update_status(job_id, JobStatus.CANCELLED)
        logger.info(f"Cancelled job {job_id}")
        return True

    async def retry_failed(self) -> int:
        """
        Re-enqueue all retryable failed jobs.

        Returns the number of jobs re-enqueued.
        """
        retryable = await self._repository.get_retryable()
        count = 0
        for job in retryable:
            await self._repository.update_status(job.id, JobStatus.RETRYING)
            await self._queue.enqueue(job.id, job.priority)
            count += 1
        if count > 0:
            logger.info(f"Re-enqueued {count} failed jobs for retry")
        return count
