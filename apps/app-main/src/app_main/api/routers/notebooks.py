"""Notebooks router - CRUD operations for notebooks."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from app_main.api.schemas import NotebookCreate, NotebookResponse, NotebookUpdate
from app_main.dependencies import get_notebook_service, get_source_service
from app_main.services.notebook_service import NotebookService
from app_main.services.source_service import SourceService

router = APIRouter(prefix="/notebooks", tags=["notebooks"])


def _notebook_dict_to_response(data: dict) -> NotebookResponse:
    """Convert a notebook dict (with counts) to a NotebookResponse schema."""
    return NotebookResponse(
        id=data.get("id", ""),
        name=data.get("name", ""),
        description=data.get("description", ""),
        archived=data.get("archived", False),
        created=str(data.get("created", "")),
        updated=str(data.get("updated", "")),
        source_count=data.get("source_count", 0),
        note_count=data.get("note_count", 0),
    )


@router.get("", response_model=list[NotebookResponse])
async def list_notebooks(
    archived: Optional[bool] = None,
    order_by: str = "updated desc",
    limit: int = Query(50, ge=1, le=100, description="Max notebooks to return"),
    offset: int = Query(0, ge=0, description="Number of notebooks to skip"),
    notebook_service: NotebookService = Depends(get_notebook_service),
):
    """List all notebooks with source and note counts."""
    notebooks = await notebook_service.get_all_with_counts(
        order_by=order_by, archived=archived, limit=limit, offset=offset,
    )
    return [_notebook_dict_to_response(nb) for nb in notebooks]


@router.post("", response_model=NotebookResponse, status_code=201)
async def create_notebook(
    body: NotebookCreate,
    notebook_service: NotebookService = Depends(get_notebook_service),
):
    """Create a new notebook."""
    notebook = await notebook_service.create(
        name=body.name, description=body.description,
    )
    logger.info("Created notebook {}", notebook.id)
    # Fetch with counts for the response
    data = await notebook_service.get_with_counts(notebook.id)
    if not data:
        raise HTTPException(status_code=500, detail="Failed to retrieve created notebook")
    return _notebook_dict_to_response(data)


@router.get("/{notebook_id}", response_model=NotebookResponse)
async def get_notebook(
    notebook_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
):
    """Get a notebook by ID with source and note counts."""
    data = await notebook_service.get_with_counts(notebook_id)
    if not data:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return _notebook_dict_to_response(data)


@router.put("/{notebook_id}", response_model=NotebookResponse)
async def update_notebook(
    notebook_id: str,
    body: NotebookUpdate,
    notebook_service: NotebookService = Depends(get_notebook_service),
):
    """Update a notebook."""
    existing = await notebook_service.get(notebook_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Notebook not found")

    update_data = body.model_dump(exclude_none=True)
    if not update_data:
        data = await notebook_service.get_with_counts(notebook_id)
        return _notebook_dict_to_response(data)

    await notebook_service.update(notebook_id, update_data)
    logger.info("Updated notebook {}", notebook_id)
    data = await notebook_service.get_with_counts(notebook_id)
    return _notebook_dict_to_response(data)


@router.post("/{notebook_id}/sources/{source_id}", status_code=204)
async def add_source_to_notebook(
    notebook_id: str,
    source_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
    source_service: SourceService = Depends(get_source_service),
):
    """Add a source to a notebook."""
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    source = await source_service.get(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    await notebook_service.add_source(notebook_id, source_id)
    logger.info("Added source {} to notebook {}", source_id, notebook_id)


@router.delete("/{notebook_id}/sources/{source_id}", status_code=204)
async def remove_source_from_notebook(
    notebook_id: str,
    source_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
):
    """Remove a source from a notebook."""
    notebook = await notebook_service.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    await notebook_service.remove_source(notebook_id, source_id)
    logger.info("Removed source {} from notebook {}", source_id, notebook_id)


@router.delete("/{notebook_id}", status_code=204)
async def delete_notebook(
    notebook_id: str,
    notebook_service: NotebookService = Depends(get_notebook_service),
):
    """Delete a notebook."""
    existing = await notebook_service.get(notebook_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Notebook not found")
    await notebook_service.delete(notebook_id)
    logger.info("Deleted notebook {}", notebook_id)
