"""Tests for the job worker."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.models.jobs import Job
from shared.types.enums import JobPriority, JobStatus, JobType

from job_queue.exceptions import JobPausedForReviewError
from job_queue.queue import JobQueue
from job_queue.registry import HandlerRegistry
from job_queue.repository import JobRepository
from job_queue.worker import (
    DEFAULT_MAX_CONCURRENCY,
    JobWorker,
    _resolve_max_concurrency,
)


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
        repo.add_to_dead_letter = AsyncMock()
        repo.list_jobs = AsyncMock(return_value=[])
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

    # ------------------------------------------------------------------
    # B.1f: JobPausedForReviewError → PAUSED_FOR_REVIEW translation.
    #
    # The exception is raised by handlers (e.g. extraction-service
    # ``SchemaReviewPendingError``) to signal "park this job, do not
    # retry, do not dead-letter". The worker's ``except`` clause for
    # ``JobPausedForReviewError`` MUST sit above the generic
    # ``except Exception`` because the latter would otherwise swallow
    # the pause signal and treat it as a transient failure.
    #
    # These tests pin both the happy path AND the clause-ordering
    # invariant: if a future refactor flips the except order so the
    # generic clause traps the pause first, ``test_paused_for_review_*``
    # will start asserting ``RETRYING`` / ``FAILED`` instead of
    # ``PAUSED_FOR_REVIEW`` and fail loudly.
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_paused_for_review_no_retry_no_dead_letter(
        self, queue, repository, registry
    ):
        """Handler raises ``JobPausedForReviewError`` → job goes to
        ``PAUSED_FOR_REVIEW``, NOT ``FAILED`` / ``RETRYING``.

        Ordering guard: the worker has a dedicated ``except`` clause
        for this exception above the generic catch-all (see
        ``worker.py:154-167``). Reversing the clause order would cause
        every paused review job to be retried and eventually dead-
        lettered. This test fails loudly if that regression happens
        because the assertions check the FAILED / RETRYING / dead-
        letter paths were NOT taken.
        """
        job = _make_job(retry_count=0, max_retries=2)
        repository.get.return_value = job
        repository.update_status.return_value = job

        @registry.register(JobType.DOCUMENT_PARSE)
        async def handler(payload):
            raise JobPausedForReviewError(
                "Notebook nb:abc requires review of 3 pending extensions."
            )

        worker = JobWorker(queue, repository, registry)

        await queue.enqueue("job:test1")
        await worker.start()
        await asyncio.sleep(0.1)
        await worker.stop()

        # Status calls: PROCESSING then PAUSED_FOR_REVIEW (no RETRYING,
        # no FAILED).
        calls = repository.update_status.call_args_list
        assert len(calls) == 2
        assert calls[0].args == ("job:test1", JobStatus.PROCESSING)
        assert calls[1].args == ("job:test1", JobStatus.PAUSED_FOR_REVIEW)

        # Error message carries the handler's context for the UI.
        err_msg = calls[1].kwargs.get("error_message", "")
        assert "Notebook nb:abc" in err_msg
        assert "3 pending extensions" in err_msg

        # No retry, no dead-letter.
        for call in calls:
            assert call.args[1] != JobStatus.RETRYING
            assert call.args[1] != JobStatus.FAILED
        repository.add_to_dead_letter.assert_not_called()
        # No re-enqueue: queue must be empty after the paused job.
        assert queue.size() == 0

    @pytest.mark.asyncio
    async def test_paused_for_review_subclass_routes_correctly(
        self, queue, repository, registry
    ):
        """A user-defined subclass of ``JobPausedForReviewError`` (the
        real production case via ``SchemaReviewPendingError``) is also
        caught by the ordered ``except`` clause.

        This is the *integration* of M1 with the extraction-service
        ``SchemaReviewPendingError`` — the worker must not know about
        subclasses, only about the base type.
        """

        class FancyPausedError(JobPausedForReviewError):
            def __init__(self):
                super().__init__("subclass-pause-context")

        job = _make_job(retry_count=0, max_retries=2)
        repository.get.return_value = job
        repository.update_status.return_value = job

        @registry.register(JobType.DOCUMENT_PARSE)
        async def handler(payload):
            raise FancyPausedError()

        worker = JobWorker(queue, repository, registry)

        await queue.enqueue("job:test1")
        await worker.start()
        await asyncio.sleep(0.1)
        await worker.stop()

        # The subclass is still routed to PAUSED_FOR_REVIEW because
        # the worker's ``except`` clause uses the base class.
        calls = repository.update_status.call_args_list
        assert calls[-1].args == ("job:test1", JobStatus.PAUSED_FOR_REVIEW)
        repository.add_to_dead_letter.assert_not_called()


# ----------------------------------------------------------------------
# Track PL.5: bounded, configurable worker concurrency.
#
# The default is 1 (serial — historical behaviour). Raising it lets
# independent jobs run in parallel WITHOUT changing the chain-enforced
# stage ordering (a successor stage is only ever enqueued after its
# predecessor completes, so the worker never has to serialize to keep
# the pipeline correct — see worker.py module docstring + source_pipeline).
# ----------------------------------------------------------------------


class TestConcurrencyResolution:
    """The concurrency limit: explicit arg > env > serial default."""

    def test_default_is_serial(self, monkeypatch):
        monkeypatch.delenv("JOB_WORKER_CONCURRENCY", raising=False)
        assert _resolve_max_concurrency(None) == DEFAULT_MAX_CONCURRENCY == 1

    def test_explicit_arg_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("JOB_WORKER_CONCURRENCY", "5")
        assert _resolve_max_concurrency(3) == 3

    def test_env_used_when_no_arg(self, monkeypatch):
        monkeypatch.setenv("JOB_WORKER_CONCURRENCY", "4")
        assert _resolve_max_concurrency(None) == 4

    def test_non_positive_falls_back_to_serial(self, monkeypatch):
        monkeypatch.delenv("JOB_WORKER_CONCURRENCY", raising=False)
        assert _resolve_max_concurrency(0) == 1
        assert _resolve_max_concurrency(-2) == 1

    def test_garbage_env_falls_back_to_serial(self, monkeypatch):
        monkeypatch.setenv("JOB_WORKER_CONCURRENCY", "not-an-int")
        assert _resolve_max_concurrency(None) == 1

    def test_negative_env_falls_back_to_serial(self, monkeypatch):
        monkeypatch.setenv("JOB_WORKER_CONCURRENCY", "-3")
        assert _resolve_max_concurrency(None) == 1

    def test_worker_exposes_resolved_concurrency(self):
        q = JobQueue(max_size=10)
        repo = MagicMock(spec=JobRepository)
        worker = JobWorker(q, repo, HandlerRegistry(), max_concurrency=3)
        assert worker.max_concurrency == 3


class TestWorkerConcurrency:
    """Behavioural: with concurrency >= 2, a heavy job does NOT block an
    independent light job; with concurrency == 1 they strictly serialize."""

    def _repo(self):
        repo = MagicMock(spec=JobRepository)
        repo.update_status = AsyncMock()
        repo.add_to_dead_letter = AsyncMock()
        repo.list_jobs = AsyncMock(return_value=[])
        return repo

    @pytest.mark.asyncio
    async def test_two_independent_jobs_run_in_parallel(self):
        """concurrency=2: a heavy job and a light job overlap in time.

        The heavy handler blocks on an Event the light handler sets; if the
        worker serialized, the heavy job would never see the Event set (the
        light job couldn't start) and the test would deadlock to its timeout.
        Observing overlap (both started before either finished) proves the
        two jobs were NOT serialized.
        """
        queue = JobQueue(max_size=10)
        repository = self._repo()
        registry = HandlerRegistry()

        heavy = _make_job(job_id="job:heavy", job_type=JobType.DOCUMENT_PARSE)
        light = _make_job(job_id="job:light", job_type=JobType.EMBEDDING_GENERATE)
        repository.get = AsyncMock(
            side_effect=lambda jid: {"job:heavy": heavy, "job:light": light}[jid]
        )

        started: list[str] = []
        finished: list[str] = []
        light_started = asyncio.Event()
        both_running = asyncio.Event()

        @registry.register(JobType.DOCUMENT_PARSE)
        async def heavy_handler(payload):
            started.append("heavy")
            # Wait until the light job has also started → proves overlap.
            await asyncio.wait_for(light_started.wait(), timeout=2.0)
            both_running.set()
            finished.append("heavy")
            return {"heavy": True}

        @registry.register(JobType.EMBEDDING_GENERATE)
        async def light_handler(payload):
            started.append("light")
            light_started.set()
            finished.append("light")
            return {"light": True}

        worker = JobWorker(queue, repository, registry, max_concurrency=2)

        await queue.enqueue("job:heavy")
        await queue.enqueue("job:light")
        await worker.start()
        # If they ran in parallel, both_running fires quickly.
        await asyncio.wait_for(both_running.wait(), timeout=2.0)
        await worker.stop()

        assert both_running.is_set()
        assert set(started) == {"heavy", "light"}
        # The light job finished while the heavy job was still mid-flight
        # (the heavy job only finishes AFTER light_started, which light sets
        # right before it returns) — i.e. they were not serialized.
        assert "light" in finished
        assert "heavy" in finished

    @pytest.mark.asyncio
    async def test_serial_default_does_not_overlap(self):
        """concurrency=1 (default): the second job cannot start until the
        first finishes. The first handler checks that the second hasn't run
        yet, proving strict serialization is preserved by default."""
        queue = JobQueue(max_size=10)
        repository = self._repo()
        registry = HandlerRegistry()

        a = _make_job(job_id="job:a", job_type=JobType.DOCUMENT_PARSE)
        b = _make_job(job_id="job:b", job_type=JobType.EMBEDDING_GENERATE)
        repository.get = AsyncMock(
            side_effect=lambda jid: {"job:a": a, "job:b": b}[jid]
        )

        order: list[str] = []
        concurrently_running = {"value": False}
        active = {"count": 0}

        @registry.register(JobType.DOCUMENT_PARSE)
        async def handler_a(payload):
            active["count"] += 1
            if active["count"] > 1:
                concurrently_running["value"] = True
            order.append("a")
            await asyncio.sleep(0.05)
            active["count"] -= 1
            return {}

        @registry.register(JobType.EMBEDDING_GENERATE)
        async def handler_b(payload):
            active["count"] += 1
            if active["count"] > 1:
                concurrently_running["value"] = True
            order.append("b")
            await asyncio.sleep(0.05)
            active["count"] -= 1
            return {}

        # No max_concurrency → default 1 (serial).
        worker = JobWorker(queue, repository, registry)
        assert worker.max_concurrency == 1

        await queue.enqueue("job:a")
        await queue.enqueue("job:b")
        await worker.start()
        await asyncio.sleep(0.3)
        await worker.stop()

        assert order == ["a", "b"]
        # Never two handlers active at once under the serial default.
        assert concurrently_running["value"] is False

    @pytest.mark.asyncio
    async def test_concurrency_bound_is_respected(self):
        """With concurrency=2 and 4 jobs, at most 2 run at the same time."""
        queue = JobQueue(max_size=10)
        repository = self._repo()
        registry = HandlerRegistry()

        jobs = {
            f"job:{i}": _make_job(job_id=f"job:{i}", job_type=JobType.DOCUMENT_PARSE)
            for i in range(4)
        }
        repository.get = AsyncMock(side_effect=lambda jid: jobs[jid])

        active = {"count": 0, "peak": 0}

        @registry.register(JobType.DOCUMENT_PARSE)
        async def handler(payload):
            active["count"] += 1
            active["peak"] = max(active["peak"], active["count"])
            await asyncio.sleep(0.05)
            active["count"] -= 1
            return {}

        worker = JobWorker(queue, repository, registry, max_concurrency=2)
        for i in range(4):
            await queue.enqueue(f"job:{i}")
        await worker.start()
        await asyncio.sleep(0.4)
        await worker.stop()

        # Parallelism happened (peak >= 2) but never exceeded the bound.
        assert active["peak"] == 2
