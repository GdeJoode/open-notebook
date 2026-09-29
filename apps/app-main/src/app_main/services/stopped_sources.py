"""Sources that stopped short of ``complete`` (WB.1).

One predicate, read by the Werkbak's "Sources that stopped" group and by
``GET /api/sources/?stalled=true`` (the page the group links to), so the
count and the list are the same set.

A source is stopped when it is at a stage short of ``complete``, no job is
queued, processing or retrying for it, and its last write (``updated``, set
on every write) is older than :data:`STALLED_AFTER_MINUTES`. The pipeline
enqueues the next stage in the step that writes the stage, and a source
with a live job is excluded, so the only normal quiet gap is between those
two writes (seconds); 15 minutes leaves room for a slow worker restart.

Each row carries ``chunks``: a source with none (a parse that extracted no
text; the zero-chunk guard then chains nothing) stays here until it is
reprocessed or deleted, and the Werkbak says so.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from job_queue.repository import JobRepository
from shared.models.jobs import Job
from shared.types.enums import JobStatus, ProcessingStage
from surrealdb_service.repositories.source import SourceRepository

STALLED_AFTER_MINUTES = 15

SHORT_OF_COMPLETE = [
    ProcessingStage.INGESTED.value,
    ProcessingStage.EMBEDDED.value,
    ProcessingStage.EXTRACTED.value,
    ProcessingStage.GRAPHED.value,
]
UNFINISHED = [JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.RETRYING]


async def unfinished_jobs(job_repo: JobRepository) -> List[Job]:
    """Queued, processing and retrying jobs, newest first."""
    jobs: List[Job] = []
    for status in UNFINISHED:
        jobs += await job_repo.list_jobs(status=status, limit=10000)
    return sorted(jobs, key=lambda j: str(j.created or ""), reverse=True)


async def stopped_sources(
    source_repo: SourceRepository,
    job_repo: Optional[JobRepository] = None,
    *,
    unfinished: Optional[List[Job]] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """The stopped sources, newest change first, each with its ``chunks``.

    ``unfinished`` may be passed when the caller already read it.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=STALLED_AFTER_MINUTES)
    rows = await source_repo.list_by_stage(SHORT_OF_COMPLETE, updated_before=cutoff)
    if unfinished is None:
        unfinished = await unfinished_jobs(job_repo or JobRepository())
    busy = {j.source_id for j in unfinished if j.source_id}
    rows = [r for r in rows if str(r["id"]) not in busy]
    counts = await source_repo.chunk_counts([str(r["id"]) for r in rows])
    for r in rows:
        r["chunks"] = counts.get(str(r["id"]), 0)
    return rows
