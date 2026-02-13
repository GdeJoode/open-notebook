"""Insights router - operations for source insights."""

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app_main.api.schemas import NoteResponse, SaveAsNoteRequest, SourceInsightResponse
from app_main.dependencies import get_insight_service
from app_main.services.insight_service import InsightService

router = APIRouter(prefix="/insights", tags=["insights"])


def _insight_to_response(insight) -> SourceInsightResponse:
    """Convert a SourceInsight domain model to a SourceInsightResponse schema."""
    return SourceInsightResponse(
        id=insight.id,
        source_id=insight.source or "",
        insight_type=insight.insight_type,
        content=insight.content,
        created=str(insight.created),
        updated=str(insight.updated),
    )


@router.get("/{insight_id}", response_model=SourceInsightResponse)
async def get_insight(
    insight_id: str,
    insight_service: InsightService = Depends(get_insight_service),
):
    """Get an insight by ID."""
    insight = await insight_service.get(insight_id)
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")
    return _insight_to_response(insight)


@router.delete("/{insight_id}", status_code=204)
async def delete_insight(
    insight_id: str,
    insight_service: InsightService = Depends(get_insight_service),
):
    """Delete an insight."""
    existing = await insight_service.get(insight_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Insight not found")
    await insight_service.delete(insight_id)
    logger.info("Deleted insight {}", insight_id)


@router.post("/{insight_id}/save-as-note", response_model=NoteResponse)
async def save_insight_as_note(
    insight_id: str,
    body: SaveAsNoteRequest,
    insight_service: InsightService = Depends(get_insight_service),
):
    """Convert an insight into a note."""
    insight = await insight_service.get(insight_id)
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")

    note = await insight_service.save_as_note(
        insight=insight, notebook_id=body.notebook_id,
    )
    logger.info("Saved insight {} as note {}", insight_id, note.id)
    return NoteResponse(
        id=note.id,
        title=note.title,
        content=note.content,
        note_type=note.note_type,
        created=str(note.created),
        updated=str(note.updated),
    )
