"""Embedding rebuild router - rebuild embeddings across the knowledge base."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from app_main.api.schemas import (
    RebuildProgress,
    RebuildRequest,
    RebuildResponse,
    RebuildStats,
    RebuildStatusResponse,
)
from surrealdb_service.connection import execute_query

router = APIRouter(tags=["embedding"])


async def _count_query(query: str) -> int:
    """Execute a count query and return the integer result."""
    try:
        result = await execute_query(query)
        if not result:
            return 0
        first = result[0]
        if isinstance(first, dict):
            return first.get("count", 0)
        if isinstance(first, int):
            return first
        return 0
    except Exception:
        return 0


@router.post("/rebuild", response_model=RebuildResponse)
async def start_rebuild(request: RebuildRequest):
    """Start a background job to rebuild embeddings.

    Modes:
    - "existing": re-embed items that already have embeddings
    - "all": embed everything with content
    """
    try:
        logger.info(f"Starting rebuild request: mode={request.mode}")

        total_estimate = 0

        if request.include_sources:
            if request.mode == "existing":
                total_estimate += await _count_query(
                    "SELECT VALUE count(array::distinct("
                    "SELECT VALUE source.id "
                    "FROM source_embedding "
                    "WHERE embedding != none AND array::len(embedding) > 0"
                    ")) as count FROM {}"
                )
            else:
                total_estimate += await _count_query(
                    "SELECT VALUE count() as count "
                    "FROM source WHERE full_text != none GROUP ALL"
                )

        if request.include_notes:
            if request.mode == "existing":
                total_estimate += await _count_query(
                    "SELECT VALUE count() as count "
                    "FROM note "
                    "WHERE embedding != none AND array::len(embedding) > 0 "
                    "GROUP ALL"
                )
            else:
                total_estimate += await _count_query(
                    "SELECT VALUE count() as count "
                    "FROM note WHERE content != none GROUP ALL"
                )

        if request.include_insights:
            if request.mode == "existing":
                total_estimate += await _count_query(
                    "SELECT VALUE count() as count "
                    "FROM source_insight "
                    "WHERE embedding != none AND array::len(embedding) > 0 "
                    "GROUP ALL"
                )
            else:
                total_estimate += await _count_query(
                    "SELECT VALUE count() as count "
                    "FROM source_insight GROUP ALL"
                )

        logger.info(f"Estimated {total_estimate} items to process")

        # Submit command for background processing
        from app_main.services.command_service import CommandService

        command_id = await CommandService.submit_command_job(
            "open_notebook",
            "rebuild_embeddings",
            {
                "mode": request.mode,
                "include_sources": request.include_sources,
                "include_notes": request.include_notes,
                "include_insights": request.include_insights,
            },
        )

        logger.info(f"Submitted rebuild command: {command_id}")

        return RebuildResponse(
            command_id=command_id,
            total_items=total_estimate,
            message=(
                f"Rebuild operation started. "
                f"Estimated {total_estimate} items to process."
            ),
        )

    except Exception as e:
        logger.error(f"Failed to start rebuild: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start rebuild operation: {str(e)}",
        )


@router.get("/rebuild/{command_id}/status", response_model=RebuildStatusResponse)
async def get_rebuild_status(command_id: str):
    """Get the status of a rebuild operation."""
    try:
        from app_main.services.command_service import CommandService

        status_data = await CommandService.get_command_status(command_id)

        if not status_data:
            raise HTTPException(
                status_code=404, detail="Rebuild command not found"
            )

        response = RebuildStatusResponse(
            command_id=command_id,
            status=status_data.get("status", "unknown"),
        )

        result = status_data.get("result")
        if result and isinstance(result, dict):
            # Build progress info
            if "total_items" in result and "processed_items" in result:
                total = result["total_items"]
                processed = result["processed_items"]
                response.progress = RebuildProgress(
                    processed=processed,
                    total=total,
                    percentage=round(
                        (processed / total * 100) if total > 0 else 0, 2
                    ),
                )

            # Build stats
            response.stats = RebuildStats(
                sources=result.get("sources_processed", 0),
                notes=result.get("notes_processed", 0),
                insights=result.get("insights_processed", 0),
                failed=result.get("failed_items", 0),
            )

        # Add timestamps
        response.started_at = status_data.get("created")
        response.completed_at = status_data.get("updated")

        # Add error message if failed
        if response.status == "failed" and result and isinstance(result, dict):
            response.error_message = result.get(
                "error_message", "Unknown error"
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get rebuild status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get rebuild status: {str(e)}",
        )
