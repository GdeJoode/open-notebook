"""Definitief mislukken van een job (F2-15, ADR 0011).

- Na de laatste poging roept de worker de afhandeling van het jobtype aan, één keer.
- ``JobNotRetryableError`` mislukt meteen, zonder nieuwe poging, en roept die
  afhandeling ook aan.
- Een fout in de afhandeling verdringt de mislukte job niet.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.models.jobs import Job
from shared.types.enums import JobStatus, JobType

from job_queue.exceptions import JobNotRetryableError
from job_queue.queue import JobQueue
from job_queue.registry import HandlerRegistry
from job_queue.repository import JobRepository
from job_queue.worker import JobWorker

pytestmark = pytest.mark.asyncio


def _job(retry_count: int, max_retries: int = 2) -> Job:
    return Job(id="job:stap", job_type=JobType.RUN_STEP, payload={"step_id": "run_step:s"},
               retry_count=retry_count, max_retries=max_retries)


def _worker(fout: Exception, jobs: list, afhandeling) -> tuple[JobWorker, MagicMock, JobQueue]:
    repo = MagicMock(spec=JobRepository)
    repo.get = AsyncMock(side_effect=jobs)
    repo.update_status = AsyncMock()
    repo.add_to_dead_letter = AsyncMock()
    registry = HandlerRegistry()

    async def handler(payload):
        raise fout

    registry.register_handler(JobType.RUN_STEP, handler)
    registry.register_final_failure(JobType.RUN_STEP, afhandeling)
    queue = JobQueue()
    return JobWorker(queue, repo, registry), repo, queue


async def _draai(worker: JobWorker, queue: JobQueue) -> None:
    await queue.enqueue("job:stap")
    while not queue.is_empty():
        await worker._execute_job(await queue.dequeue())


async def test_afhandeling_na_de_laatste_poging():
    afhandeling = AsyncMock()
    fout = RuntimeError("mislukt")
    worker, repo, queue = _worker(fout, [_job(0), _job(1), _job(2)], afhandeling)

    await _draai(worker, queue)

    afhandeling.assert_awaited_once()
    job, gegeven = afhandeling.await_args.args
    assert job.retry_count == 2 and gegeven is fout
    statussen = [c.args[1] for c in repo.update_status.call_args_list]
    assert statussen.count(JobStatus.RETRYING) == 2 and statussen[-1] == JobStatus.FAILED


async def test_niet_herhaalbaar_mislukt_meteen():
    afhandeling = AsyncMock()
    worker, repo, queue = _worker(JobNotRetryableError("run afgesloten"), [_job(0)], afhandeling)

    await _draai(worker, queue)

    afhandeling.assert_awaited_once()
    statussen = [c.args[1] for c in repo.update_status.call_args_list]
    assert JobStatus.RETRYING not in statussen and statussen[-1] == JobStatus.FAILED


async def test_fout_in_de_afhandeling_verdringt_de_job_niet():
    afhandeling = AsyncMock(side_effect=RuntimeError("database weg"))
    worker, repo, queue = _worker(JobNotRetryableError("x"), [_job(0)], afhandeling)

    await _draai(worker, queue)

    repo.add_to_dead_letter.assert_awaited_once()
    assert repo.update_status.call_args_list[-1].args[1] == JobStatus.FAILED


async def test_de_handler_krijgt_het_job_id():
    from job_queue.exceptions import JOB_ID_SLEUTEL

    ontvangen = {}
    repo = MagicMock(spec=JobRepository)
    repo.get = AsyncMock(return_value=_job(0))
    repo.update_status = AsyncMock()
    registry = HandlerRegistry()

    async def handler(payload):
        ontvangen.update(payload)
        return {}

    registry.register_handler(JobType.RUN_STEP, handler)
    queue = JobQueue()
    await _draai(JobWorker(queue, repo, registry), queue)

    assert ontvangen[JOB_ID_SLEUTEL] == "job:stap"
    assert ontvangen["step_id"] == "run_step:s"
