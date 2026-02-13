"""Commands router - job management via job-queue service."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from app_main.services.command_service import CommandService

router = APIRouter(prefix="/commands", tags=["commands"])


class CommandExecutionRequest(BaseModel):
    command: str = Field(..., description="Command function name")
    app: str = Field(..., description="Application name")
    input: Dict[str, Any] = Field(..., description="Arguments to pass to the command")


class CommandJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class CommandJobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created: Optional[str] = None
    updated: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None


@router.post("/jobs", response_model=CommandJobResponse)
async def execute_command(request: CommandExecutionRequest):
    """Submit a command for background processing.

    Returns immediately with job ID for status tracking.
    """
    try:
        job_id = await CommandService.submit_command_job(
            module_name=request.app,
            command_name=request.command,
            command_args=request.input,
        )

        return CommandJobResponse(
            job_id=job_id,
            status="submitted",
            message=f"Command '{request.command}' submitted successfully",
        )
    except Exception as e:
        logger.error(f"Error submitting command: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit command: {str(e)}",
        )


@router.get("/jobs/{job_id}", response_model=CommandJobStatusResponse)
async def get_command_job_status(job_id: str):
    """Get the status of a specific command job."""
    try:
        status_data = await CommandService.get_command_status(job_id)
        return CommandJobStatusResponse(**status_data)
    except Exception as e:
        logger.error(f"Error fetching job status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch job status: {str(e)}",
        )


@router.get("/jobs", response_model=List[Dict[str, Any]])
async def list_command_jobs(
    command_filter: Optional[str] = Query(
        None, description="Filter by command name"
    ),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, description="Maximum number of jobs to return"),
):
    """List command jobs with optional filtering."""
    try:
        jobs = await CommandService.list_command_jobs(
            command_filter=command_filter,
            status_filter=status_filter,
            limit=limit,
        )
        return jobs
    except Exception as e:
        logger.error(f"Error listing command jobs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list command jobs: {str(e)}",
        )


@router.delete("/jobs/{job_id}")
async def cancel_command_job(job_id: str):
    """Cancel a running command job."""
    try:
        success = await CommandService.cancel_command_job(job_id)
        return {"job_id": job_id, "cancelled": success}
    except Exception as e:
        logger.error(f"Error cancelling command job: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel command job: {str(e)}",
        )
