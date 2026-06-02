"""Resolution log router - review and manage entity match decisions."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app_main.services.resolution_log_service import ResolutionLogService

router = APIRouter(prefix="/resolution-log", tags=["resolution-log"])


def get_resolution_log_service() -> ResolutionLogService:
    return ResolutionLogService()


class ResolutionLogEntry(BaseModel):
    id: Optional[str] = None
    entity_a_text: str = ""
    entity_b_text: str = ""
    entity_a_label: str = "UNKNOWN"
    entity_b_label: str = "UNKNOWN"
    match: bool = False
    confidence: float = 0.0
    match_method: str = ""
    match_reasoning: str = ""
    iterations: int = 1
    source_document_a: Optional[str] = None
    source_document_b: Optional[str] = None
    source_section_a: Optional[str] = None
    source_section_b: Optional[str] = None
    matched_by_model: Optional[str] = None
    match_timestamp: Optional[str] = None
    status: str = "pending"
    reviewed_at: Optional[str] = None
    source_id: Optional[str] = None


class BulkStatusUpdate(BaseModel):
    ids: List[str] = Field(..., description="Resolution log entry IDs")
    status: str = Field(..., description="New status: accepted or rejected")


@router.get("", response_model=Dict[str, Any])
async def list_resolution_log(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, description="Filter by status"),
    source_id: Optional[str] = Query(None, description="Filter by source"),
    match_method: Optional[str] = Query(None, description="Filter by match method"),
    match_only: Optional[bool] = Query(None, description="Filter matches only"),
    svc: ResolutionLogService = Depends(get_resolution_log_service),
):
    """List resolution log entries with filters and pagination."""
    items, total = await svc.list_candidates(
        limit=limit,
        offset=offset,
        status=status,
        source_id=source_id,
        match_method=match_method,
        match_only=match_only,
    )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/stats", response_model=Dict[str, Any])
async def get_stats(
    svc: ResolutionLogService = Depends(get_resolution_log_service),
):
    """Get resolution log summary statistics."""
    return await svc.get_stats()


@router.get("/methods", response_model=List[str])
async def get_methods(
    svc: ResolutionLogService = Depends(get_resolution_log_service),
):
    """Get distinct match methods."""
    return await svc.get_methods()


@router.get("/{entry_id:path}", response_model=Dict[str, Any])
async def get_entry(
    entry_id: str,
    svc: ResolutionLogService = Depends(get_resolution_log_service),
):
    """Get a single resolution log entry."""
    result = await svc.get_candidate(entry_id)
    if not result:
        raise HTTPException(status_code=404, detail="Entry not found")
    return result


@router.patch("/{entry_id:path}/status")
async def update_entry_status(
    entry_id: str,
    status: str = Query(..., description="New status: accepted or rejected"),
    svc: ResolutionLogService = Depends(get_resolution_log_service),
):
    """Accept or reject a resolution log entry."""
    if status not in ("accepted", "rejected"):
        raise HTTPException(status_code=422, detail="Status must be 'accepted' or 'rejected'")
    result = await svc.update_status(entry_id, status)
    if not result:
        raise HTTPException(status_code=404, detail="Entry not found")
    return result


@router.post("/bulk-status")
async def bulk_update_status(
    body: BulkStatusUpdate,
    svc: ResolutionLogService = Depends(get_resolution_log_service),
):
    """Bulk accept/reject resolution log entries."""
    if body.status not in ("accepted", "rejected"):
        raise HTTPException(status_code=422, detail="Status must be 'accepted' or 'rejected'")
    updated = await svc.bulk_update_status(body.ids, body.status)
    return {"updated": updated}
