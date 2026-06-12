"""Notebooks router - CRUD operations for notebooks."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from app_main.api.schemas import NotebookCreate, NotebookResponse, NotebookUpdate
from app_main.dependencies import (
    get_notebook_merge_service,
    get_notebook_service,
    get_source_service,
)
from app_main.services.notebook_merge_service import NotebookMergeService
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


# ---------------------------------------------------------------------------
# B.6: cross-notebook graph merge.
# ---------------------------------------------------------------------------


class NotebookMergeRequest(BaseModel):
    """Request body for the cross-notebook merge endpoint.

    ``type_match_required`` defaults to ``True`` per Q-B-5 — disjoint
    type-tag overlaps surface as conflicts instead of merging
    incompatible entities. Operators with looser semantics (e.g.
    knowingly merging an Organization and Person row) flip the flag
    to ``False``.

    ``dry_run`` powers the UI preview button; the response carries the
    same shape but the service performs no writes.
    """

    source_ids: List[str] = Field(
        ...,
        description="Source notebook record IDs (e.g. ['notebook:a','notebook:b'])",
        min_length=1,
    )
    target_id: str = Field(
        ..., description="Target notebook record ID to merge INTO"
    )
    type_match_required: bool = Field(
        default=True,
        description=(
            "When True (default), disjoint type_tags across notebooks are "
            "recorded as conflicts and skipped. When False, the labels are "
            "unioned and the entity is merged regardless."
        ),
    )
    dry_run: bool = Field(
        default=False,
        description="When True, compute counts without writing to the DB.",
    )


class NotebookMergeConflictResponse(BaseModel):
    """One conflict row in the merge report — see service docs."""

    normalized_name: str
    conflicting_type_tags: List[str]
    source_notebook_ids: List[str]


class NotebookMergeReportResponse(BaseModel):
    """API mirror of the service's ``NotebookMergeReport`` dataclass."""

    entities_merged: int
    relations_created: int
    conflicts: List[NotebookMergeConflictResponse]
    source_notebook_ids: List[str]
    target_notebook_id: str
    dry_run: bool


@router.post("/merge", response_model=NotebookMergeReportResponse)
async def merge_notebooks(
    body: NotebookMergeRequest,
    merge_service: NotebookMergeService = Depends(get_notebook_merge_service),
    notebook_service: NotebookService = Depends(get_notebook_service),
):
    """Merge entities + relations from N source notebooks into a target.

    Returns the populated merge report (entities_merged, relations_created,
    conflicts list, echoed source/target ids).

    Errors:
        404 — the target notebook doesn't exist.
        422 — empty ``source_ids`` (pydantic ``min_length=1`` already
              rejects this before we get here, but the explicit guard
              documents intent for direct service calls).
    """
    target = await notebook_service.get(body.target_id)
    if not target:
        raise HTTPException(
            status_code=404, detail="Target notebook not found"
        )
    # Pydantic min_length=1 will reject empty arrays at validation time
    # with a 422 already — this branch only fires for service-side
    # callers and keeps the contract explicit.
    if not body.source_ids:
        raise HTTPException(
            status_code=422, detail="source_ids must not be empty"
        )

    report = await merge_service.merge_notebooks(
        source_notebook_ids=body.source_ids,
        target_notebook_id=body.target_id,
        type_match_required=body.type_match_required,
        dry_run=body.dry_run,
    )

    logger.info(
        "Merged notebooks {sources} -> {target} "
        "(entities={ents}, relations={rels}, conflicts={confs}, dry_run={dr})",
        sources=body.source_ids,
        target=body.target_id,
        ents=report.entities_merged,
        rels=report.relations_created,
        confs=len(report.conflicts),
        dr=body.dry_run,
    )

    return NotebookMergeReportResponse(
        entities_merged=report.entities_merged,
        relations_created=report.relations_created,
        conflicts=[
            NotebookMergeConflictResponse(
                normalized_name=c.normalized_name,
                conflicting_type_tags=c.conflicting_type_tags,
                source_notebook_ids=c.source_notebook_ids,
            )
            for c in report.conflicts
        ],
        source_notebook_ids=report.source_notebook_ids,
        target_notebook_id=report.target_notebook_id,
        dry_run=report.dry_run,
    )
