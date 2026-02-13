"""
Notebook repository with specialized operations.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models import ChatMessage, ChatSession, Note, Notebook
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.base import BaseRepository


class NotebookRepository(BaseRepository[Notebook]):
    """Repository for Notebook operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(Notebook, config)

    async def get_sources(self, notebook_id: str) -> List[Dict[str, Any]]:
        """
        Get all sources for a notebook.

        Args:
            notebook_id: Notebook ID.

        Returns:
            List of source data (without full_text for efficiency).
        """
        try:
            result = await execute_query(
                """
                SELECT * OMIT source.full_text FROM (
                    SELECT in AS source FROM reference WHERE out=$id
                    FETCH source
                ) ORDER BY source.updated DESC
                """,
                {"id": ensure_record_id(notebook_id)},
                self.config,
            )
            return [item.get("source", {}) for item in result] if result else []
        except Exception as e:
            logger.error(f"Failed to get sources for notebook {notebook_id}: {e}")
            return []

    async def get_notes(self, notebook_id: str) -> List[Dict[str, Any]]:
        """
        Get all notes for a notebook.

        Args:
            notebook_id: Notebook ID.

        Returns:
            List of note data (without content/embedding for efficiency).
        """
        try:
            result = await execute_query(
                """
                SELECT * OMIT note.content, note.embedding FROM (
                    SELECT in AS note FROM artifact WHERE out=$id
                    FETCH note
                ) ORDER BY note.updated DESC
                """,
                {"id": ensure_record_id(notebook_id)},
                self.config,
            )
            return [item.get("note", {}) for item in result] if result else []
        except Exception as e:
            logger.error(f"Failed to get notes for notebook {notebook_id}: {e}")
            return []

    async def get_chat_sessions(self, notebook_id: str) -> List[Dict[str, Any]]:
        """
        Get all chat sessions for a notebook.

        Args:
            notebook_id: Notebook ID.

        Returns:
            List of chat session data.
        """
        try:
            result = await execute_query(
                """
                SELECT * FROM (
                    SELECT <- chat_session AS chat_session
                    FROM refers_to
                    WHERE out=$id
                    FETCH chat_session
                ) ORDER BY chat_session.updated DESC
                """,
                {"id": ensure_record_id(notebook_id)},
                self.config,
            )
            return [
                item.get("chat_session", [{}])[0]
                for item in result
                if item.get("chat_session")
            ] if result else []
        except Exception as e:
            logger.error(f"Failed to get chat sessions for notebook {notebook_id}: {e}")
            return []


class NoteRepository(BaseRepository[Note]):
    """Repository for Note operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(Note, config)

    async def add_to_notebook(self, note_id: str, notebook_id: str) -> bool:
        """
        Add a note to a notebook.

        Args:
            note_id: Note ID.
            notebook_id: Notebook ID.

        Returns:
            True if successful.
        """
        try:
            await self.relate(note_id, "artifact", notebook_id)
            return True
        except Exception as e:
            logger.error(f"Failed to add note to notebook: {e}")
            return False

    async def create_with_embedding(
        self,
        data: Dict[str, Any],
        embedding: Optional[List[float]] = None,
    ) -> Note:
        """
        Create a note with optional embedding.

        Args:
            data: Note data.
            embedding: Optional pre-computed embedding.

        Returns:
            Created note.
        """
        if embedding:
            data["embedding"] = embedding
        return await self.create(data)


class ChatSessionRepository(BaseRepository[ChatSession]):
    """Repository for ChatSession operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(ChatSession, config)

    async def relate_to_notebook(self, session_id: str, notebook_id: str) -> bool:
        """
        Relate a chat session to a notebook.

        Args:
            session_id: Chat session ID.
            notebook_id: Notebook ID.

        Returns:
            True if successful.
        """
        try:
            await self.relate(session_id, "refers_to", notebook_id)
            return True
        except Exception as e:
            logger.error(f"Failed to relate session to notebook: {e}")
            return False

    async def relate_to_source(self, session_id: str, source_id: str) -> bool:
        """
        Relate a chat session to a source.

        Args:
            session_id: Chat session ID.
            source_id: Source ID.

        Returns:
            True if successful.
        """
        try:
            await self.relate(session_id, "refers_to", source_id)
            return True
        except Exception as e:
            logger.error(f"Failed to relate session to source: {e}")
            return False


class ChatMessageRepository(BaseRepository[ChatMessage]):
    """Repository for ChatMessage operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(ChatMessage, config)

    async def get_session_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Get all messages for a chat session.

        Args:
            session_id: Chat session ID.
            limit: Optional message limit.

        Returns:
            List of messages ordered by creation time.
        """
        return await self.query(
            "session=$session_id",
            {"session_id": ensure_record_id(session_id)},
            order_by="created ASC",
            limit=limit,
        )
