"""
Notebook API routes.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from shared.models import ChatMessage, ChatSession, Note, Notebook
from surrealdb_service.repositories import (
    ChatMessageRepository,
    ChatSessionRepository,
    NoteRepository,
    NotebookRepository,
)

router = APIRouter()


# Request/Response models
class NotebookCreate(BaseModel):
    name: str
    description: str = ""
    archived: bool = False


class NotebookUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    archived: Optional[bool] = None


class NoteCreate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    note_type: Optional[str] = None
    notebook_id: Optional[str] = None


class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    note_type: Optional[str] = None


class ChatSessionCreate(BaseModel):
    title: Optional[str] = None
    model_override: Optional[str] = None
    notebook_id: Optional[str] = None


class ChatMessageCreate(BaseModel):
    session: str
    role: str
    content: str


# Notebook endpoints
@router.get("", response_model=List[Dict[str, Any]])
async def list_notebooks(archived: Optional[bool] = None):
    """List all notebooks."""
    repo = NotebookRepository()
    notebooks = await repo.get_all(order_by="updated DESC")
    if archived is not None:
        notebooks = [n for n in notebooks if n.archived == archived]
    return [n.model_dump() for n in notebooks]


@router.post("", response_model=Dict[str, Any])
async def create_notebook(data: NotebookCreate):
    """Create a new notebook."""
    repo = NotebookRepository()
    notebook = await repo.create(data.model_dump())
    return notebook.model_dump()


@router.get("/{notebook_id}", response_model=Dict[str, Any])
async def get_notebook(notebook_id: str):
    """Get a notebook by ID."""
    repo = NotebookRepository()
    notebook = await repo.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return notebook.model_dump()


@router.patch("/{notebook_id}", response_model=Dict[str, Any])
async def update_notebook(notebook_id: str, data: NotebookUpdate):
    """Update a notebook."""
    repo = NotebookRepository()
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    notebook = await repo.update(notebook_id, updates)
    return notebook.model_dump()


@router.delete("/{notebook_id}")
async def delete_notebook(notebook_id: str):
    """Delete a notebook."""
    repo = NotebookRepository()
    await repo.delete(notebook_id)
    return {"status": "deleted"}


@router.get("/{notebook_id}/sources", response_model=List[Dict[str, Any]])
async def get_notebook_sources(notebook_id: str):
    """Get all sources for a notebook."""
    repo = NotebookRepository()
    return await repo.get_sources(notebook_id)


@router.get("/{notebook_id}/notes", response_model=List[Dict[str, Any]])
async def get_notebook_notes(notebook_id: str):
    """Get all notes for a notebook."""
    repo = NotebookRepository()
    return await repo.get_notes(notebook_id)


@router.get("/{notebook_id}/sessions", response_model=List[Dict[str, Any]])
async def get_notebook_sessions(notebook_id: str):
    """Get all chat sessions for a notebook."""
    repo = NotebookRepository()
    return await repo.get_chat_sessions(notebook_id)


# Note endpoints
@router.post("/notes", response_model=Dict[str, Any])
async def create_note(data: NoteCreate):
    """Create a new note."""
    repo = NoteRepository()
    note_data = {k: v for k, v in data.model_dump().items() if k != "notebook_id"}
    note = await repo.create(note_data)

    if data.notebook_id:
        await repo.add_to_notebook(note.id, data.notebook_id)

    return note.model_dump()


@router.get("/notes/{note_id}", response_model=Dict[str, Any])
async def get_note(note_id: str):
    """Get a note by ID."""
    repo = NoteRepository()
    note = await repo.get(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note.model_dump()


@router.patch("/notes/{note_id}", response_model=Dict[str, Any])
async def update_note(note_id: str, data: NoteUpdate):
    """Update a note."""
    repo = NoteRepository()
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
    note = await repo.update(note_id, updates)
    return note.model_dump()


@router.delete("/notes/{note_id}")
async def delete_note(note_id: str):
    """Delete a note."""
    repo = NoteRepository()
    await repo.delete(note_id)
    return {"status": "deleted"}


# Chat session endpoints
@router.post("/sessions", response_model=Dict[str, Any])
async def create_session(data: ChatSessionCreate):
    """Create a new chat session."""
    repo = ChatSessionRepository()
    session_data = {k: v for k, v in data.model_dump().items() if k != "notebook_id"}
    session = await repo.create(session_data)

    if data.notebook_id:
        await repo.relate_to_notebook(session.id, data.notebook_id)

    return session.model_dump()


@router.get("/sessions/{session_id}", response_model=Dict[str, Any])
async def get_session(session_id: str):
    """Get a chat session by ID."""
    repo = ChatSessionRepository()
    session = await repo.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.model_dump()


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    repo = ChatSessionRepository()
    await repo.delete(session_id)
    return {"status": "deleted"}


@router.get("/sessions/{session_id}/messages", response_model=List[Dict[str, Any]])
async def get_session_messages(session_id: str, limit: Optional[int] = None):
    """Get all messages for a chat session."""
    repo = ChatMessageRepository()
    messages = await repo.get_session_messages(session_id, limit)
    return [m.model_dump() for m in messages]


@router.post("/sessions/{session_id}/messages", response_model=Dict[str, Any])
async def create_message(session_id: str, data: ChatMessageCreate):
    """Create a new message in a chat session."""
    repo = ChatMessageRepository()
    message = await repo.create({
        "session": session_id,
        "role": data.role,
        "content": data.content,
    })
    return message.model_dump()
