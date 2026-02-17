"""
Background worker that pulls jobs from the queue and executes handlers.

The worker runs as an asyncio task and processes one job at a time.
It handles retries, status updates, and graceful shutdown.
"""

import asyncio
from datetime import datetime, timezone

from loguru import logger

from shared.types.enums import JobStatus

from job_queue.queue import JobQueue
from job_queue.registry import HandlerRegistry
from job_queue.repository import JobRepository


class JobWorker:
    """
    Pulls jobs from the queue, executes via registry, updates DB.

    Usage::

        worker = JobWorker(queue, repository, registry)
        await worker.start()   # starts background processing
        ...
        await worker.stop()    # graceful shutdown
    """

    def __init__(
        self,
        queue: JobQueue,
        repository: JobRepository,
        registry: HandlerRegistry,
    ):
        self._queue = queue
        self._repository = repository
        self._registry = registry
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start the processing loop as an asyncio task."""
        if self._running:
            logger.warning("Worker is already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("Job worker started")

    async def stop(self, timeout: float = 30.0) -> None:
        """
        Graceful shutdown: finish current job, then stop.

        Args:
            timeout: Max seconds to wait for the current job to finish.
        """
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("Worker shutdown timed out, cancelling")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None
        logger.info("Job worker stopped")

    async def _process_loop(self) -> None:
        """Main loop: dequeue → execute → update status."""
        while self._running:
            try:
                # Use wait_for so we can check _running periodically
                job_id = await asyncio.wait_for(self._queue.dequeue(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            await self._execute_job(job_id)

    async def _execute_job(self, job_id: str) -> None:
        """Execute a single job with error handling and retry logic."""
        job = await self._repository.get(job_id)
        if job is None:
            logger.error(f"Job {job_id} not found in database, skipping")
            return

        if job.status == JobStatus.CANCELLED:
            logger.info(f"Job {job_id} was cancelled, skipping")
            return

        # Mark as processing
        await self._repository.update_status(job_id, JobStatus.PROCESSING)
        logger.info(f"Processing job {job_id} (type={job.job_type.value})")

        try:
            # Safety net: reconstruct payload from top-level fields if SCHEMAFULL stripped it
            payload = dict(job.payload) if job.payload else {}
            if "source_id" not in payload and hasattr(job, "source_id") and job.source_id:
                payload["source_id"] = str(job.source_id)

            # Warn if payload looks empty (SCHEMAFULL stripping likely occurred)
            if not payload or (len(payload) <= 1 and "source_id" in payload):
                logger.warning(
                    f"Job {job_id} payload appears stripped by SCHEMAFULL schema. "
                    "Ensure migration 30 has been applied or restart the server "
                    "so the FLEXIBLE field override takes effect."
                )

            result = await self._registry.execute(job.job_type, payload)
            await self._repository.update_status(
                job_id,
                JobStatus.COMPLETED,
                result=result,
            )
            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            if job.retry_count < job.max_retries:
                new_count = job.retry_count + 1
                logger.info(
                    f"Retrying job {job_id} (attempt {new_count}/{job.max_retries})"
                )
                await self._repository.update_status(
                    job_id,
                    JobStatus.RETRYING,
                    retry_count=new_count,
                    error_message=str(e),
                )
                await self._queue.enqueue(job_id, job.priority)
            else:
                await self._repository.update_status(
                    job_id,
                    JobStatus.FAILED,
                    error_message=str(e),
                )
