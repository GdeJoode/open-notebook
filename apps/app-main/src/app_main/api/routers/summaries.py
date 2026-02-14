"""Summaries router - document summarization management."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app_main.dependencies import get_summarization_service
from app_main.services.summarization_service import SummarizationService

router = APIRouter(prefix="/summaries", tags=["summaries"])


class GenerateSummaryRequest(BaseModel):
    source_id: str
    strategy: str
    config: Optional[Dict[str, Any]] = None


@router.get("/strategies", response_model=List[Dict[str, Any]])
async def list_strategies(
    svc: SummarizationService = Depends(get_summarization_service),
):
    """List available summarization strategies with implementation status."""
    return await svc.list_strategies()


@router.get("", response_model=List[Dict[str, Any]])
async def list_summaries(
    source_id: Optional[str] = None,
    svc: SummarizationService = Depends(get_summarization_service),
):
    """List generated summaries, optionally filtered by source."""
    return await svc.list_summaries(source_id=source_id)


@router.get("/{summary_id:path}", response_model=Dict[str, Any])
async def get_summary(
    summary_id: str,
    svc: SummarizationService = Depends(get_summarization_service),
):
    """Get a single summary with full content."""
    result = await svc.get_summary(summary_id)
    if not result:
        raise HTTPException(status_code=404, detail="Summary not found")
    return result


@router.post("/generate", response_model=Dict[str, Any], status_code=201)
async def generate_summary(
    body: GenerateSummaryRequest,
    svc: SummarizationService = Depends(get_summarization_service),
):
    """Generate a new summary for a source."""
    try:
        result = await svc.generate_summary(
            source_id=body.source_id,
            strategy=body.strategy,
            config=body.config,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{summary_id:path}", status_code=204)
async def delete_summary(
    summary_id: str,
    svc: SummarizationService = Depends(get_summarization_service),
):
    """Delete a summary."""
    deleted = await svc.delete_summary(summary_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Summary not found")
