"""
Job repository for SurrealDB persistence.

Extends BaseRepository with job-specific query methods.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models import Job
from shared.models.jobs import JobSubmitRequest
from shared.types.enums import JobPriority, JobStatus, JobType
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.base import BaseRepository


class JobRepository(BaseRepository[Job]):
    """Repository for Job CRUD and status management."""

    _schema_ensured: bool = False

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(Job, config)

    async def _ensure_flexible_payload(self) -> None:
        """Ensure payload/result fields are FLEXIBLE so SurrealDB preserves nested data.

        The job table is SCHEMAFULL, which means SurrealDB strips nested fields
        from plain ``object`` columns. This runs once per process to apply the
        FLEXIBLE flag, making it safe regardless of whether migration 30 has
        been applied.
        """
        if JobRepository._schema_ensured:
            return
        try:
            await execute_query(
                "DEFINE FIELD OVERWRITE payload ON TABLE job FLEXIBLE TYPE object DEFAULT {};"
                "DEFINE FIELD OVERWRITE result ON TABLE job FLEXIBLE TYPE option<object>;",
                config=self.config,
            )
            JobRepository._schema_ensured = True
            logger.debug("Ensured FLEXIBLE payload/result fields on job table")
        except Exception as e:
            logger.warning(f"Failed to ensure FLEXIBLE payload field: {e}")

    async def create_from_submission(self, submission: JobSubmitRequest) -> Job:
        """Create a job record from a submission request."""
        await self._ensure_flexible_payload()
        data = {
            "job_type": submission.job_type.value,
            "status": JobStatus.QUEUED.value,
            "priority": submission.priority.value,
            "payload": submission.payload,
            "max_retries": submission.max_retries,
            "source_id": submission.source_id,
            "retry_count": 0,
        }
        return await self.create(data)

    async def update_status(
        self,
        job_id: str,
        status: JobStatus,
        **fields: Any,
    ) -> Job:
        """
        Update job status and optional additional fields.

        Args:
            job_id: Job record ID.
            status: New status.
            **fields: Additional fields to update (result, error_message, etc.)
        """
        data: Dict[str, Any] = {"status": status.value}

        if status == JobStatus.PROCESSING:
            data["started_at"] = datetime.now(timezone.utc)
        elif status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            # B.1f: PAUSED_FOR_REVIEW also terminates the *current*
            # processing attempt — the job is dormant until a user
            # resumes it. ``completed_at`` doubles as "last activity"
            # so the UI can show "paused 2h ago".
            JobStatus.PAUSED_FOR_REVIEW,
        ):
            data["completed_at"] = datetime.now(timezone.utc)

        for key, value in fields.items():
            data[key] = value

        return await self.update(job_id, data)

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        limit: int = 50,
    ) -> List[Job]:
        """List jobs with optional filters."""
        conditions = []
        params: Dict[str, Any] = {}

        if status is not None:
            conditions.append("status = $status")
            params["status"] = status.value
        if job_type is not None:
            conditions.append("job_type = $job_type")
            params["job_type"] = job_type.value

        where = " AND ".join(conditions) if conditions else "true"
        return await self.query(
            where=where,
            params=params,
            order_by="created DESC",
            limit=limit,
        )

    async def get_stats(self) -> Dict[str, int]:
        """Get job counts grouped by status."""
        try:
            result = await execute_query(
                "SELECT status, count() as count FROM job GROUP BY status",
                config=self.config,
            )
            stats: Dict[str, int] = {s.value: 0 for s in JobStatus}
            stats["total"] = 0
            for row in result:
                s = row.get("status", "")
                c = row.get("count", 0)
                stats[s] = c
                stats["total"] += c
            return stats
        except Exception as e:
            logger.error(f"Failed to get job stats: {e}")
            return {"total": 0}

    async def get_retryable(self) -> List[Job]:
        """Get failed jobs that can be retried."""
        return await self.query(
            where="status = $status AND retry_count < max_retries",
            params={"status": JobStatus.FAILED.value},
            order_by="created ASC",
        )

    async def get_by_source(self, source_id: str) -> List[Job]:
        """Get all jobs for a given source."""
        return await self.query(
            where="source_id = $source_id",
            params={"source_id": source_id},
            order_by="created DESC",
        )

    async def add_to_dead_letter(
        self,
        job_id: str,
        job_type: str,
        payload: Dict[str, Any],
        error_message: str,
        retry_count: int,
    ) -> None:
        """Record a permanently failed job in the dead-letter table."""
        try:
            await execute_query(
                "CREATE dead_letter SET "
                "job_id = $job_id, "
                "job_type = $job_type, "
                "payload = $payload, "
                "error_message = $error_message, "
                "retry_count = $retry_count, "
                "failed_at = time::now()",
                {
                    "job_id": job_id,
                    "job_type": job_type,
                    "payload": payload,
                    "error_message": error_message,
                    "retry_count": retry_count,
                },
                self.config,
            )
            logger.info(f"Job {job_id} added to dead-letter queue")
        except Exception as e:
            logger.error(f"Failed to add job {job_id} to dead-letter: {e}")
