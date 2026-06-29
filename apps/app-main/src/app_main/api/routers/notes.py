"""Notes router - CRUD operations for notes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from app_main.api.schemas import (
    NoteAutoLinkResponse,
    NoteCreate,
    NoteResponse,
    NoteUpdate,
)
from app_main.dependencies import (
    get_note_auto_link_service,
    get_note_service,
    get_notebook_service,
)
from app_main.services.note_auto_link_service import (
    MAX_K,
    NoteAutoLinkService,
)
from app_main.services.note_service import NoteService
from app_main.services.notebook_service import NotebookService
from surrealdb_service.repositories.base import _validate_record_id

router = APIRouter(prefix="/notes", tags=["notes"])


def _validate_note_id(note_id: str) -> str:
    """Reject a malformed/unsafe note id at the route boundary (defense-in-depth).

    The repo's ``relate_note`` strict-validates ids before any interpolation
    (Y.1), but validating here too means a bad id (``;``-bearing injection
    payload, wrong table, garbage) returns a clean 422 instead of falling
    through to a service/DB layer. Mirrors the Y.1 follow-up: validate at the
    boundary, not only in the repo.
    """
    try:
        validated = _validate_record_id(str(note_id))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid note id: {exc}")
    if not validated.startswith("note:"):
        raise HTTPException(
            status_code=422, detail="id must be a note record (note:...)"
        )
    return validated


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
    limit: int = Query(50, ge=1, le=100, description="Max notes to return"),
    offset: int = Query(0, ge=0, description="Number of notes to skip"),
    note_service: NoteService = Depends(get_note_service),
    notebook_service: NotebookService = Depends(get_notebook_service),
):
    """List all notes, optionally filtered by notebook."""
    if notebook_id:
        notebook = await notebook_service.get(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")
        raw_notes = await notebook_service.get_notes(notebook_id)
        # Apply pagination to notebook-filtered notes
        paginated = raw_notes[offset : offset + limit]
        results = []
        for raw in paginated:
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

    notes = await note_service.get_all(limit=limit, offset=offset)
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


@router.post("/{note_id}/auto-link", response_model=NoteAutoLinkResponse)
async def auto_link_note(
    note_id: str,
    k: int = Query(
        5,
        ge=1,
        le=MAX_K,
        description="Max related notes to consider/link (top-k by cosine)",
    ),
    min_similarity: float = Query(
        0.75,
        ge=-1.0,
        le=1.0,
        description="Minimum cosine similarity to link",
    ),
    note_service: NoteService = Depends(get_note_service),
    auto_link_service: NoteAutoLinkService = Depends(get_note_auto_link_service),
):
    """Auto-link a note to its most-related notes AND sources by embedding
    similarity (Y.2 + NS.2).

    Ensures the note has an embedding (embeds it if not), then runs two passes
    over that one embedding: it ranks other **notes** by cosine and writes
    idempotent ``related_note`` edges, and ranks **sources** by cosine and
    writes idempotent ``note_about`` edges — both for candidates at/above
    ``min_similarity`` (top-``k`` per pass). Re-running is a no-op for
    already-linked pairs (clear-before-relate). The summary carries two distinct
    counter families (``created``/``skipped_existing``/... for notes,
    ``source_links_created``/``source_skipped_existing``/... for sources) so the
    two link types never conflate. Never a 500 for the ordinary no-content /
    not-found cases.

    Validation happens at the boundary: the note id is strict-validated, and
    ``k`` / ``min_similarity`` are bounded by ``Query`` constraints — a bad value
    is a 422, not a 500.
    """
    # Route-layer id validation (defense-in-depth; the repo also validates).
    _validate_note_id(note_id)

    existing = await note_service.get(note_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Note not found")

    result = await auto_link_service.auto_link(
        note_id, k=k, min_similarity=min_similarity
    )
    logger.info(
        "Auto-linked note {} -> {} edges (status={})",
        note_id,
        result.created,
        result.status,
    )
    return NoteAutoLinkResponse(**result.to_dict())
