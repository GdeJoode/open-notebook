"""
Job management API routes.

Provides endpoints for listing, inspecting, and cancelling background jobs
via the job-queue package.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel

from shared.types.enums import JobStatus

router = APIRouter()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class JobResponse(BaseModel):
    """Single job status response."""

    job_id: str
    job_type: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created: Optional[str] = None
    updated: Optional[str] = None


class JobListResponse(BaseModel):
    """Paginated list of jobs."""

    jobs: List[JobResponse]
    total: int


class JobStatsResponse(BaseModel):
    """Aggregate job statistics."""

    stats: Dict[str, int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_service():
    """Lazy-import JobService to keep job-queue an optional dependency."""
    try:
        from job_queue import JobRepository, JobService, JobQueue, JobWorker
        from job_queue.registry import HandlerRegistry

        # Use a module-level singleton so repeated calls share state
        if not hasattr(_get_service, "_instance"):
            repo = JobRepository()
            queue = JobQueue()
            registry = HandlerRegistry()
            worker = JobWorker(queue, repo, registry)
            _get_service._instance = JobService(repo, queue, worker)
        return _get_service._instance
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="job-queue package is not installed. Install with: pip install file-manager[queue]",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status_filter: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(50, ge=1, le=200, description="Max jobs to return"),
):
    """List background jobs with optional status filter."""
    try:
        service = _get_service()
        status = None
        if status_filter:
            try:
                status = JobStatus(status_filter)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status_filter}. "
                    f"Valid values: {[s.value for s in JobStatus]}",
                )

        jobs = await service.list_jobs(status=status, limit=limit)
        items = [
            JobResponse(
                job_id=j.id,
                job_type=j.job_type.value,
                status=j.status.value,
                error_message=j.error_message,
                created=str(j.created) if j.created else None,
                updated=str(j.updated) if j.updated else None,
            )
            for j in jobs
        ]
        return JobListResponse(jobs=items, total=len(items))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=JobStatsResponse)
async def get_job_stats():
    """Get aggregate job statistics (counts by status)."""
    try:
        service = _get_service()
        stats = await service.get_stats()
        return JobStatsResponse(stats=stats)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get status of a specific job."""
    try:
        service = _get_service()
        job = await service.get_status(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return JobResponse(
            job_id=job.id,
            job_type=job.job_type.value,
            status=job.status.value,
            result=job.result,
            error_message=job.error_message,
            created=str(job.created) if job.created else None,
            updated=str(job.updated) if job.updated else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a queued job."""
    try:
        service = _get_service()
        success = await service.cancel(job_id)
        if not success:
            raise HTTPException(
                status_code=409,
                detail=f"Job '{job_id}' cannot be cancelled (may already be processing or completed)",
            )
        return {"job_id": job_id, "cancelled": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
