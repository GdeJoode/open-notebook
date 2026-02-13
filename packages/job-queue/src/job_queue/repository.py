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

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(Job, config)

    async def create_from_submission(self, submission: JobSubmitRequest) -> Job:
        """Create a job record from a submission request."""
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
            data["started_at"] = datetime.now(timezone.utc).isoformat()
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            data["completed_at"] = datetime.now(timezone.utc).isoformat()

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
