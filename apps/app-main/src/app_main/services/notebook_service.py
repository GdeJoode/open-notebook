"""
Notebook service - business logic for notebook operations.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models import Notebook
from surrealdb_service.repositories import (
    NoteRepository,
    NotebookRepository,
    SourceRepository,
)
from surrealdb_service.connection import execute_query, ensure_record_id


class NotebookService:
    """Service for notebook business logic."""

    def __init__(
        self,
        notebook_repo: NotebookRepository,
        source_repo: SourceRepository,
        note_repo: NoteRepository,
    ):
        self.notebook_repo = notebook_repo
        self.source_repo = source_repo
        self.note_repo = note_repo

    async def get_all_with_counts(
        self, order_by: str = "updated desc", archived: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Get all notebooks with source and note counts."""
        query = f"""
            SELECT *,
            count(<-reference.in) as source_count,
            count(<-artifact.in) as note_count
            FROM notebook
            ORDER BY {order_by}
        """
        result = await execute_query(query)
        if archived is not None:
            result = [nb for nb in result if nb.get("archived") == archived]
        return result or []

    async def get_with_counts(self, notebook_id: str) -> Optional[Dict[str, Any]]:
        """Get a single notebook with source and note counts."""
        query = """
            SELECT *,
            count(<-reference.in) as source_count,
            count(<-artifact.in) as note_count
            FROM $notebook_id
        """
        result = await execute_query(
            query, {"notebook_id": ensure_record_id(notebook_id)}
        )
        return result[0] if result else None

    async def create(self, name: str, description: str = "") -> Notebook:
        """Create a new notebook."""
        return await self.notebook_repo.create(
            {"name": name, "description": description}
        )

    async def update(self, notebook_id: str, data: Dict[str, Any]) -> Notebook:
        """Update a notebook."""
        return await self.notebook_repo.update(notebook_id, data)

    async def delete(self, notebook_id: str) -> bool:
        """Delete a notebook."""
        return await self.notebook_repo.delete(notebook_id)

    async def get(self, notebook_id: str) -> Optional[Notebook]:
        """Get a notebook by ID."""
        return await self.notebook_repo.get(notebook_id)

    async def get_sources(self, notebook_id: str) -> List[Dict[str, Any]]:
        """Get all sources for a notebook."""
        return await self.notebook_repo.get_sources(notebook_id)

    async def get_notes(self, notebook_id: str) -> List[Dict[str, Any]]:
        """Get all notes for a notebook."""
        return await self.notebook_repo.get_notes(notebook_id)

    async def get_chat_sessions(self, notebook_id: str) -> List[Dict[str, Any]]:
        """Get all chat sessions for a notebook."""
        return await self.notebook_repo.get_chat_sessions(notebook_id)

    async def add_source(self, notebook_id: str, source_id: str) -> bool:
        """Add a source to a notebook, checking for existing reference."""
        existing = await execute_query(
            "SELECT * FROM reference WHERE out = $source_id AND in = $notebook_id",
            {
                "notebook_id": ensure_record_id(notebook_id),
                "source_id": ensure_record_id(source_id),
            },
        )
        if not existing:
            await execute_query(
                "RELATE $source_id->reference->$notebook_id",
                {
                    "notebook_id": ensure_record_id(notebook_id),
                    "source_id": ensure_record_id(source_id),
                },
            )
        return True

    async def remove_source(self, notebook_id: str, source_id: str) -> bool:
        """Remove a source from a notebook."""
        await execute_query(
            "DELETE FROM reference WHERE out = $notebook_id AND in = $source_id",
            {
                "notebook_id": ensure_record_id(notebook_id),
                "source_id": ensure_record_id(source_id),
            },
        )
        return True
