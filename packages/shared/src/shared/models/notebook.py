"""
Notebook-related domain models.

Pure Pydantic models for notebooks, notes, and chat sessions.
Database operations are handled by the surrealdb-service package.
"""

from typing import Any, ClassVar, Dict, List, Literal, Optional

from pydantic import Field, field_validator

from shared.models.base import ObjectModel
from shared.types.enums import NoteType


class Notebook(ObjectModel):
    """
    A notebook for organizing sources, notes, and chat sessions.
    """
    table_name: ClassVar[str] = "notebook"

    name: str = Field(description="Notebook name")
    description: str = Field(default="", description="Notebook description")
    archived: bool = Field(default=False, description="Whether the notebook is archived")

    # Per-notebook privacy layer (Track J.3). ``None`` inherits the global
    # ``default_privacy_mode``; ``"cloud"`` | ``"private"`` override it for this
    # notebook. A per-document ``source.private`` still wins over this (sticky).
    # Mirrors ``migrations/52.surrealql`` (``notebook.privacy_mode TYPE
    # option<string>``); legacy rows read back None.
    privacy_mode: Optional[str] = Field(
        default=None,
        description="Per-notebook privacy mode: None (inherit global) | 'cloud' | 'private'",
    )

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        """Validate that notebook name is not empty."""
        if not v or not v.strip():
            raise ValueError("Notebook name cannot be empty")
        return v.strip()


class Note(ObjectModel):
    """
    A note within a notebook.

    Can be human-created or AI-generated.
    """
    table_name: ClassVar[str] = "note"

    title: Optional[str] = None
    note_type: Optional[Literal["human", "ai"]] = None
    content: Optional[str] = None
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, v: Optional[str]) -> Optional[str]:
        """Validate that content is not empty if provided."""
        if v is not None and not v.strip():
            raise ValueError("Note content cannot be empty")
        return v

    def get_context(
        self, context_size: Literal["short", "long"] = "short"
    ) -> Dict[str, Any]:
        """Get note context for LLM consumption."""
        if context_size == "long":
            return {"id": self.id, "title": self.title, "content": self.content}
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content[:100] if self.content else None,
        }

    def needs_embedding(self) -> bool:
        """Check if this note needs embedding."""
        return True

    def get_embedding_content(self) -> Optional[str]:
        """Get the content to embed."""
        return self.content


class ChatSession(ObjectModel):
    """
    A chat session associated with a notebook.
    """
    table_name: ClassVar[str] = "chat_session"

    title: Optional[str] = None
    model_override: Optional[str] = Field(
        None, description="Override the default model for this session"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatMessage(ObjectModel):
    """
    A message within a chat session.
    """
    table_name: ClassVar[str] = "chat_message"

    session: Optional[str] = Field(None, description="Reference to the chat session")
    role: Literal["user", "assistant", "system"] = Field(description="Message role")
    content: str = Field(description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
