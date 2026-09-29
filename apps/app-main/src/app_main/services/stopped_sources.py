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

A source without chunks (a parse that extracted no text; the zero-chunk
guard then chains nothing) stays here until it is reprocessed or deleted;
the Werkbak reads the chunk counts of the rows it shows
(:func:`add_chunk_counts`) and says so.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

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
    busy: Optional[Set[str]] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """The stopped sources, newest change first: the predicate only.

    ``busy`` (the sources with a queued, processing or retrying job) may be
    passed when the caller already read the unfinished jobs; otherwise it is
    one query. The reason text is :func:`add_chunk_counts`, for the rows a
    caller shows.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=STALLED_AFTER_MINUTES)
    rows = await source_repo.list_by_stage(SHORT_OF_COMPLETE, updated_before=cutoff)
    if busy is None:
        busy = await (job_repo or JobRepository()).unfinished_source_ids(
            [s.value for s in UNFINISHED]
        )
    return [r for r in rows if str(r["id"]) not in busy]


async def add_chunk_counts(
    source_repo: SourceRepository, rows: List[Dict[str, Any]]
) -> None:
    """Set ``chunks`` on each row: 0 means the parse extracted no text."""
    counts = await source_repo.chunk_counts([str(r["id"]) for r in rows])
    for r in rows:
        r["chunks"] = counts.get(str(r["id"]), 0)
