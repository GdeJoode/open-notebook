"""
Insight service - business logic for source insights.
"""

from typing import Any, Dict, Optional

from shared.models import Note, SourceInsight
from surrealdb_service.repositories import NoteRepository, SourceInsightRepository


class InsightService:
    """Service for insight business logic."""

    def __init__(
        self,
        insight_repo: SourceInsightRepository,
        note_repo: NoteRepository,
    ):
        self.insight_repo = insight_repo
        self.note_repo = note_repo

    async def get(self, insight_id: str) -> Optional[SourceInsight]:
        """Get an insight by ID."""
        return await self.insight_repo.get(insight_id)

    async def get_source_for_insight(self, insight_id: str):
        """Get the source that owns this insight."""
        return await self.insight_repo.get_source(insight_id)

    async def delete(self, insight_id: str) -> bool:
        """Delete an insight."""
        return await self.insight_repo.delete(insight_id)

    async def save_as_note(
        self,
        insight: SourceInsight,
        notebook_id: Optional[str] = None,
    ) -> Note:
        """Convert an insight to a note."""
        note = await self.note_repo.create({
            "title": f"Insight: {insight.insight_type}",
            "content": insight.content,
            "note_type": "ai",
        })
        if notebook_id and note.id:
            await self.note_repo.add_to_notebook(note.id, notebook_id)
        return note
