"""
Background worker that pulls jobs from the queue and executes handlers.

The worker runs as an asyncio task. It handles retries, status updates, and
graceful shutdown.

Concurrency (Track PL.5)
------------------------
The worker supports *bounded* concurrency: up to ``max_concurrency`` jobs run
in parallel, governed by an ``asyncio.Semaphore``. The default is **1**
(serial — behaviour-identical to the historical one-at-a-time worker), so the
change is opt-in: an operator raises it via ``JOB_WORKER_CONCURRENCY`` (or the
constructor arg) only when they want parallelism.

Concurrency is safe for this codebase because the pipeline does NOT rely on the
worker serializing jobs for correctness:

* **Stage ordering is enforced by the job chain, not by the worker.** Each
  source stage (INGEST→EMBED→EXTRACT→GRAPH) is enqueued only *after* its
  predecessor completed (see ``app_main.services.source_pipeline.advance_source``).
  EXTRACT for a source is never even on the queue until that source's EMBED job
  finished — so two workers can never run EMBED and EXTRACT for the same source
  out of order. The only designed same-source parallelism (INSIGHTS ∥ EXTRACT
  off EMBED) operates on disjoint, idempotent data.
* **The DB layer is per-connection-pooled** (``surrealdb_service`` acquires a
  distinct connection per call, no shared session ``LET`` state), so concurrent
  queries don't collide.
* **The LLM path is already concurrency-coordinated** — the Track J rate limiter
  and the circuit breakers are ``asyncio.Lock``-guarded shared singletons built
  to throttle parallel calls.

Because the default is 1, the conservative deployment is unchanged; the
semaphore simply caps fan-out when concurrency is raised.
"""

import asyncio
import os

from loguru import logger

from shared.types.enums import JobStatus

from job_queue.exceptions import JobPausedForReviewError
from job_queue.queue import JobQueue
from job_queue.registry import HandlerRegistry
from job_queue.repository import JobRepository

# Conservative default: 1 == serial (historical behaviour preserved).
DEFAULT_MAX_CONCURRENCY = 1
_ENV_CONCURRENCY = "JOB_WORKER_CONCURRENCY"


def _resolve_max_concurrency(explicit: int | None) -> int:
    """Resolve the concurrency limit: explicit arg > env > default(1).

    Any non-positive or unparseable value falls back to the serial default so a
    misconfiguration can never *disable* the worker or remove the bound.
    """
    if explicit is not None:
        return explicit if explicit >= 1 else DEFAULT_MAX_CONCURRENCY
    raw = os.environ.get(_ENV_CONCURRENCY)
    if raw is None:
        return DEFAULT_MAX_CONCURRENCY
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            f"{_ENV_CONCURRENCY}={raw!r} is not an int; "
            f"using serial default {DEFAULT_MAX_CONCURRENCY}"
        )
        return DEFAULT_MAX_CONCURRENCY
    if value < 1:
        logger.warning(
            f"{_ENV_CONCURRENCY}={value} < 1; using serial default "
            f"{DEFAULT_MAX_CONCURRENCY}"
        )
        return DEFAULT_MAX_CONCURRENCY
    return value


class JobWorker:
    """
    Pulls jobs from the queue, executes via registry, updates DB.

    Runs up to ``max_concurrency`` jobs in parallel (default 1 == serial).

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
        max_concurrency: int | None = None,
    ):
        self._queue = queue
        self._repository = repository
        self._registry = registry
        self._running = False
        self._task: asyncio.Task | None = None
        self._max_concurrency = _resolve_max_concurrency(max_concurrency)
        # Bounds the number of in-flight jobs. With the default of 1 this gives
        # exactly the historical one-at-a-time behaviour.
        self._slots = asyncio.Semaphore(self._max_concurrency)
        # Tracks running job tasks so shutdown can await them.
        self._inflight: set[asyncio.Task] = set()
        if self._max_concurrency > 1:
            logger.info(
                f"Job worker concurrency set to {self._max_concurrency} "
                f"(parallel jobs)"
            )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    async def start(self) -> None:
        """Start the processing loop as an asyncio task.

        On startup, re-enqueues any jobs left in QUEUED, PROCESSING, or
        RETRYING state from a previous run so they aren't lost.
        """
        if self._running:
            logger.warning("Worker is already running")
            return

        await self._reload_pending_jobs()

        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("Job worker started")

    async def _reload_pending_jobs(self) -> None:
        """Re-enqueue jobs that were in-flight when the worker last stopped."""
        reloaded = 0
        for status in (JobStatus.QUEUED, JobStatus.RETRYING, JobStatus.PROCESSING):
            jobs = await self._repository.list_jobs(status=status, limit=200)
            for job in jobs:
                await self._queue.enqueue(str(job.id), job.priority)
                reloaded += 1
                # Reset PROCESSING jobs back to QUEUED so they're picked up cleanly
                if status == JobStatus.PROCESSING:
                    await self._repository.update_status(
                        str(job.id), JobStatus.QUEUED
                    )
        if reloaded:
            logger.info(f"Reloaded {reloaded} pending job(s) from database")

    async def stop(self, timeout: float = 30.0) -> None:
        """
        Graceful shutdown: stop dequeuing, finish in-flight jobs, then stop.

        Args:
            timeout: Max seconds to wait for the dispatch loop AND every
                in-flight job to finish.
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
        # Await any still-running job tasks (the dispatch loop has stopped
        # accepting new ones). With concurrency==1 there is at most one.
        if self._inflight:
            inflight = list(self._inflight)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*inflight, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"{len(self._inflight)} in-flight job(s) did not finish "
                    "within the shutdown timeout; cancelling"
                )
                for task in inflight:
                    task.cancel()
                await asyncio.gather(*inflight, return_exceptions=True)
        logger.info("Job worker stopped")

    async def _process_loop(self) -> None:
        """Main dispatch loop: acquire a slot → dequeue → execute concurrently.

        Acquiring a semaphore slot *before* dequeuing means we only pull a job
        off the queue when we have capacity to run it — so a slow batch of heavy
        jobs cannot drain the queue into memory, and the bound is honoured. The
        job runs as a tracked task that releases its slot on completion, so an
        independent light job can be dispatched into a free slot without waiting
        for a heavy one to finish (when ``max_concurrency`` > 1).
        """
        while self._running:
            # Wait for a free execution slot. wait_for lets us re-check _running.
            try:
                await asyncio.wait_for(self._slots.acquire(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                # Use wait_for so we can check _running periodically.
                job_id = await asyncio.wait_for(self._queue.dequeue(), timeout=5.0)
            except asyncio.TimeoutError:
                # No job available — release the slot and loop.
                self._slots.release()
                continue
            except asyncio.CancelledError:
                self._slots.release()
                break

            # Dispatch the job concurrently; it releases its slot when done.
            task = asyncio.create_task(self._run_job_with_slot(job_id))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)

    async def _run_job_with_slot(self, job_id: str) -> None:
        """Execute one job and always release its concurrency slot afterwards."""
        try:
            await self._execute_job(job_id)
        finally:
            self._slots.release()

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

        except JobPausedForReviewError as e:
            # Handler signalled "park, don't fail". No retry, no
            # dead-letter — the UI surfaces the paused state and the
            # user can resume the job after acting (B.3c). The message
            # carries enough context for the UI to render a prompt.
            logger.warning(
                f"Job {job_id} paused for review: {e}"
            )
            await self._repository.update_status(
                job_id,
                JobStatus.PAUSED_FOR_REVIEW,
                error_message=str(e),
            )
            return

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
                # Write to dead-letter table for post-mortem analysis
                await self._repository.add_to_dead_letter(
                    job_id=job_id,
                    job_type=job.job_type.value,
                    payload=payload,
                    error_message=str(e),
                    retry_count=job.retry_count,
                )
