"""Extraction jobs parked by the schema-review gate, per notebook (B.3c).

Read by the notebook's ``/extraction/paused`` and ``/extraction/resume``
endpoints and by the Werkbak (WB.1), so all three see the same jobs.
"""

from typing import Any, List

from shared.models.classification import ClassificationScope
from shared.types.enums import JobStatus, JobType
from surrealdb_service.repositories.source import SourceRepository


async def list_paused_jobs_for_notebook(
    notebook_id: str,
    source_repo: SourceRepository,
) -> List[Any]:
    """Find all ``PAUSED_FOR_REVIEW`` jobs whose source belongs to this notebook.

    Done in two steps because the job table only stores ``source_id``,
    not ``notebook_id``. We walk the ``reference`` edge from the notebook
    to its sources and then query the job table for paused entries.

    :class:`JobRepository` is imported at call time, so a test can patch
    ``job_queue.repository.JobRepository``.
    """
    from job_queue.repository import JobRepository

    job_repo = JobRepository()

    rows = await source_repo.list_with_metadata(
        notebook_id=notebook_id, limit=500, scope=ClassificationScope.ALLES
    )
    source_ids = {str(row["id"]) for row in rows if row.get("id")}
    if not source_ids:
        return []

    # F2-15: ook stappen van een run die op de schemareview wachten.
    paused_entity_jobs = await job_repo.list_jobs(
        status=JobStatus.PAUSED_FOR_REVIEW,
        job_type=JobType.ENTITY_EXTRACT,
        limit=200,
    )
    paused_steps = await job_repo.list_jobs(
        status=JobStatus.PAUSED_FOR_REVIEW,
        job_type=JobType.RUN_STEP,
        limit=200,
    )
    paused_steps = [j for j in paused_steps if getattr(j, "job_type", None) == JobType.RUN_STEP]
    return [j for j in [*paused_entity_jobs, *paused_steps] if j.source_id in source_ids]
