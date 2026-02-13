"""Notes router - CRUD operations for notes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app_main.api.schemas import NoteCreate, NoteResponse, NoteUpdate
from app_main.dependencies import get_note_service, get_notebook_service
from app_main.services.note_service import NoteService
from app_main.services.notebook_service import NotebookService

router = APIRouter(prefix="/notes", tags=["notes"])


def _note_to_response(note) -> NoteResponse:
    """Convert a Note domain model to a NoteResponse schema."""
    return NoteResponse(
        id=note.id,
        title=note.title,
        content=note.content,
        note_type=note.note_type,
        created=str(note.created),
        updated=str(note.updated),
    )


@router.get("", response_model=list[NoteResponse])
async def list_notes(
    notebook_id: Optional[str] = None,
    note_service: NoteService = Depends(get_note_service),
    notebook_service: NotebookService = Depends(get_notebook_service),
):
    """List all notes, optionally filtered by notebook."""
    if notebook_id:
        notebook = await notebook_service.get(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        raw_notes = await notebook_service.get_notes(notebook_id)
        results = []
        for raw in raw_notes:
            results.append(
                NoteResponse(
                    id=raw.get("id", ""),
                    title=raw.get("title"),
                    content=raw.get("content"),
                    note_type=raw.get("note_type"),
                    created=str(raw.get("created", "")),
                    updated=str(raw.get("updated", "")),
                )
            )
        return results

    notes = await note_service.get_all()
    return [_note_to_response(n) for n in notes]


@router.post("", response_model=NoteResponse, status_code=201)
async def create_note(
    body: NoteCreate,
    note_service: NoteService = Depends(get_note_service),
    notebook_service: NotebookService = Depends(get_notebook_service),
):
    """Create a new note."""
    if body.notebook_id:
        notebook = await notebook_service.get(body.notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

    title = body.title
    if body.note_type == "ai" and not title:
        # Graph-based title generation will be added later
        title = "Untitled Note"

    note = await note_service.create(
        title=title,
        content=body.content,
        note_type=body.note_type,
        notebook_id=body.notebook_id,
    )
    logger.info("Created note {}", note.id)
    return _note_to_response(note)


@router.get("/{note_id}", response_model=NoteResponse)
async def get_note(
    note_id: str,
    note_service: NoteService = Depends(get_note_service),
):
    """Get a note by ID."""
    note = await note_service.get(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return _note_to_response(note)


@router.put("/{note_id}", response_model=NoteResponse)
async def update_note(
    note_id: str,
    body: NoteUpdate,
    note_service: NoteService = Depends(get_note_service),
):
    """Update a note."""
    existing = await note_service.get(note_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Note not found")

    update_data = body.model_dump(exclude_none=True)
    if not update_data:
        return _note_to_response(existing)

    note = await note_service.update(note_id, update_data)
    logger.info("Updated note {}", note_id)
    return _note_to_response(note)


@router.delete("/{note_id}", status_code=204)
async def delete_note(
    note_id: str,
    note_service: NoteService = Depends(get_note_service),
):
    """Delete a note."""
    existing = await note_service.get(note_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Note not found")
    await note_service.delete(note_id)
    logger.info("Deleted note {}", note_id)
