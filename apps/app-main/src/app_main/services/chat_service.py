"""
Chat service - business logic for chat sessions and messages.
"""

from typing import Any, Dict, List, Optional

from shared.models import ChatMessage, ChatSession
from surrealdb_service.repositories import ChatMessageRepository, ChatSessionRepository


class ChatService:
    """Service for chat business logic."""

    def __init__(
        self,
        session_repo: ChatSessionRepository,
        message_repo: ChatMessageRepository,
    ):
        self.session_repo = session_repo
        self.message_repo = message_repo

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID."""
        return await self.session_repo.get(session_id)

    async def create_session(
        self, notebook_id: Optional[str] = None, source_id: Optional[str] = None,
    ) -> ChatSession:
        """Create a new chat session and relate to notebook/source."""
        session = await self.session_repo.create({})

        if session.id:
            if notebook_id:
                await self.session_repo.relate_to_notebook(session.id, notebook_id)
            if source_id:
                await self.session_repo.relate_to_source(session.id, source_id)

        return session

    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        return await self.session_repo.delete(session_id)

    async def get_messages(
        self, session_id: str, limit: Optional[int] = None,
    ) -> List[ChatMessage]:
        """Get all messages for a session."""
        return await self.message_repo.get_session_messages(session_id, limit)

    async def add_message(
        self, session_id: str, role: str, content: str,
    ) -> ChatMessage:
        """Add a message to a session."""
        return await self.message_repo.create({
            "session": session_id,
            "role": role,
            "content": content,
        })
