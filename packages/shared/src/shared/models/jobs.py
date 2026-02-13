"""
Job-related domain models.

Pure Pydantic models for background job processing.
"""

from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from shared.models.base import ObjectModel
from shared.types.enums import JobPriority, JobStatus, JobType


class Job(ObjectModel):
    """
    Job record stored in the database.

    Represents a background processing job with status tracking,
    retry support, and result storage.
    """

    table_name: ClassVar[str] = "job"

    job_type: JobType = JobType.DOCUMENT_PARSE
    status: JobStatus = JobStatus.QUEUED
    priority: JobPriority = JobPriority.NORMAL
    payload: Dict[str, Any] = {}
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    source_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class JobConfig(BaseModel):
    """Configuration for job submission."""

    model_config = ConfigDict(frozen=True)

    max_retries: int = 2
    priority: JobPriority = JobPriority.NORMAL


class JobProgress(BaseModel):
    """Progress information for a running job."""

    job_id: str
    job_type: JobType
    status: JobStatus
    progress_pct: Optional[float] = None
    message: Optional[str] = None


class JobResult(BaseModel):
    """Result of a completed job."""

    job_id: str
    job_type: JobType
    status: JobStatus
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None


class JobSubmitRequest(BaseModel):
    """API request model for submitting a new job."""

    job_type: JobType
    priority: JobPriority = JobPriority.NORMAL
    payload: Dict[str, Any] = {}
    max_retries: int = 2
    source_id: Optional[str] = None


class JobListResponse(BaseModel):
    """API response model for listing jobs."""

    jobs: List[Job]
    total: int


class JobStatsResponse(BaseModel):
    """API response model for job statistics."""

    total: int = 0
    queued: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    retrying: int = 0
