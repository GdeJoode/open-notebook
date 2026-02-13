"""
Note service - business logic for note operations.
"""

from typing import Any, Dict, List, Optional

from shared.models import Note
from surrealdb_service.repositories import NoteRepository, NotebookRepository


class NoteService:
    """Service for note business logic."""

    def __init__(
        self,
        note_repo: NoteRepository,
        notebook_repo: NotebookRepository,
    ):
        self.note_repo = note_repo
        self.notebook_repo = notebook_repo

    async def get(self, note_id: str) -> Optional[Note]:
        """Get a note by ID."""
        return await self.note_repo.get(note_id)

    async def get_all(self, order_by: str = "updated desc") -> List[Note]:
        """Get all notes."""
        return await self.note_repo.get_all(order_by=order_by)

    async def create(
        self,
        title: Optional[str],
        content: str,
        note_type: Optional[str] = "human",
        notebook_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> Note:
        """Create a note and optionally add it to a notebook."""
        data: Dict[str, Any] = {"content": content}
        if title:
            data["title"] = title
        if note_type:
            data["note_type"] = note_type

        if embedding:
            note = await self.note_repo.create_with_embedding(data, embedding)
        else:
            note = await self.note_repo.create(data)

        if notebook_id and note.id:
            await self.note_repo.add_to_notebook(note.id, notebook_id)

        return note

    async def update(self, note_id: str, data: Dict[str, Any]) -> Note:
        """Update a note."""
        return await self.note_repo.update(note_id, data)

    async def delete(self, note_id: str) -> bool:
        """Delete a note."""
        return await self.note_repo.delete(note_id)

    async def add_to_notebook(self, note_id: str, notebook_id: str) -> bool:
        """Add a note to a notebook."""
        return await self.note_repo.add_to_notebook(note_id, notebook_id)
