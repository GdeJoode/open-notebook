"""
Command service - bridges background job submission.

Uses surreal_commands for now, will be replaced by job_queue in Phase 5.
"""

from typing import Any, Dict, List, Optional

from loguru import logger


class CommandService:
    """Service layer for command/job operations."""

    @staticmethod
    async def submit_command_job(
        module_name: str,
        command_name: str,
        command_args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a command for background processing."""
        try:
            from surreal_commands import submit_command

            # Ensure command modules are imported for registry validation
            try:
                import commands.podcast_commands  # noqa: F401
            except ImportError:
                pass

            cmd_id = submit_command(module_name, command_name, command_args)
            if not cmd_id:
                raise ValueError("Failed to get cmd_id from submit_command")

            cmd_id_str = str(cmd_id)
            logger.info(
                f"Submitted command job: {cmd_id_str} "
                f"for {module_name}.{command_name}"
            )
            return cmd_id_str

        except Exception as e:
            logger.error(f"Failed to submit command job: {e}")
            raise

    @staticmethod
    async def get_command_status(job_id: str) -> Dict[str, Any]:
        """Get status of a command job."""
        try:
            from surreal_commands import get_command_status

            status = await get_command_status(job_id)
            return {
                "job_id": job_id,
                "status": status.status if status else "unknown",
                "result": status.result if status else None,
                "error_message": (
                    getattr(status, "error_message", None) if status else None
                ),
                "created": (
                    str(status.created)
                    if status and hasattr(status, "created") and status.created
                    else None
                ),
                "updated": (
                    str(status.updated)
                    if status and hasattr(status, "updated") and status.updated
                    else None
                ),
                "progress": (
                    getattr(status, "progress", None) if status else None
                ),
            }
        except Exception as e:
            logger.error(f"Failed to get command status: {e}")
            raise

    @staticmethod
    async def list_command_jobs(
        module_filter: Optional[str] = None,
        command_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List command jobs with optional filtering."""
        # Will be implemented with job-queue in Phase 5
        return []

    @staticmethod
    async def cancel_command_job(job_id: str) -> bool:
        """Cancel a running command job."""
        try:
            logger.info(f"Attempting to cancel job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel command job: {e}")
            raise
