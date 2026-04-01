"""
Command service - bridges background job submission via job-queue.

Provides a backward-compatible interface so existing routers can call
submit/status/cancel/list without knowing the underlying implementation.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models.jobs import Job, JobSubmitRequest
from shared.types.enums import JobPriority, JobStatus, JobType

from job_queue import JobQueue, JobRepository, JobService, JobWorker
from job_queue.registry import HandlerRegistry

# ---------------------------------------------------------------------------
# Module-level singletons — initialised once at import time.
# The FastAPI lifespan hooks call start_worker() / stop_worker().
# ---------------------------------------------------------------------------

_registry = HandlerRegistry()
_repository = JobRepository()
_queue = JobQueue()
_worker = JobWorker(_queue, _repository, _registry)
_service = JobService(_repository, _queue, _worker)


def get_registry() -> HandlerRegistry:
    """Return the global handler registry (for handler registration)."""
    return _registry


def get_service() -> JobService:
    """Return the global JobService instance."""
    return _service


async def start_worker() -> None:
    """Start the background job worker (call from FastAPI lifespan)."""
    await _service.start()


async def stop_worker() -> None:
    """Stop the background job worker (call from FastAPI lifespan)."""
    await _service.stop()


# ---------------------------------------------------------------------------
# Job-type mapping: old (module, command_name) -> new JobType
# ---------------------------------------------------------------------------

_COMMAND_TO_JOB_TYPE: Dict[str, JobType] = {
    "process_source": JobType.DOCUMENT_PARSE,
    "generate_podcast": JobType.BATCH_PROCESS,
    "embed_single_item": JobType.EMBEDDING_GENERATE,
    "embed_source": JobType.EMBEDDING_GENERATE,
    "rebuild_embeddings": JobType.EMBEDDING_GENERATE,
    "run_summaries": JobType.INSIGHT_EXTRACT,
    "run_entities": JobType.ENTITY_EXTRACT,
    "process_text": JobType.CHUNK_EXTRACT,
    "analyze_data": JobType.INSIGHT_EXTRACT,
}


class CommandService:
    """Backward-compatible service layer wrapping JobService."""

    @staticmethod
    async def submit_command_job(
        module_name: str,
        command_name: str,
        command_args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a job for background processing. Returns the job ID."""
        job_type = _COMMAND_TO_JOB_TYPE.get(command_name)
        if job_type is None:
            raise ValueError(
                f"Unknown command: {command_name}. "
                f"Known commands: {list(_COMMAND_TO_JOB_TYPE.keys())}"
            )

        submission = JobSubmitRequest(
            job_type=job_type,
            payload={
                "module_name": module_name,
                "command_name": command_name,
                **command_args,
            },
            source_id=command_args.get("source_id"),
        )
        job = await _service.submit(submission)
        logger.info(
            f"Submitted job {job.id} for {module_name}.{command_name}"
        )
        return job.id

    @staticmethod
    async def get_command_status(job_id: str) -> Dict[str, Any]:
        """Get status of a job — returns a dict matching the old shape."""
        job = await _service.get_status(job_id)
        if job is None:
            return {
                "job_id": job_id,
                "status": "unknown",
                "result": None,
                "error_message": None,
                "created": None,
                "updated": None,
                "progress": None,
            }
        return {
            "job_id": job.id,
            "status": job.status.value,
            "result": job.result,
            "error_message": job.error_message,
            "created": str(job.created) if job.created else None,
            "updated": str(job.updated) if job.updated else None,
            "progress": None,
        }

    @staticmethod
    async def list_command_jobs(
        module_filter: Optional[str] = None,
        command_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List jobs with optional filtering."""
        status = None
        if status_filter:
            try:
                status = JobStatus(status_filter)
            except ValueError:
                pass

        jobs = await _service.list_jobs(status=status, limit=limit)
        return [
            {
                "job_id": j.id,
                "job_type": j.job_type.value,
                "status": j.status.value,
                "created": str(j.created) if j.created else None,
                "updated": str(j.updated) if j.updated else None,
                "error_message": j.error_message,
            }
            for j in jobs
        ]

    @staticmethod
    async def cancel_command_job(job_id: str) -> bool:
        """Cancel a queued job."""
        return await _service.cancel(job_id)
