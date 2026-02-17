"""Preprocessing router — document analysis, classification, and chunk filtering."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app_main.dependencies import (
    get_preprocessing_service,
    get_preprocessing_service_with_llm,
)
from app_main.services.preprocessing_service import PreprocessingService

router = APIRouter(prefix="/preprocessing", tags=["preprocessing"])


class RunPreprocessingRequest(BaseModel):
    source_id: str


class UpdatePreprocessingRequest(BaseModel):
    classification: Optional[Dict[str, Any]] = None
    filtered_chunk_ids: Optional[List[str]] = None
    removed_chunk_ids: Optional[List[str]] = None


@router.post("/run", response_model=Dict[str, Any], status_code=201)
async def run_preprocessing(
    body: RunPreprocessingRequest,
    svc: PreprocessingService = Depends(get_preprocessing_service_with_llm),
):
    """Run preprocessing (summary + classification + chunk filtering) for a source."""
    try:
        result = await svc.run(source_id=body.source_id)
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{source_id}", response_model=Optional[Dict[str, Any]])
async def get_preprocessing_result(
    source_id: str,
    svc: PreprocessingService = Depends(get_preprocessing_service),
):
    """Get stored preprocessing result for a source."""
    result = await svc.get_result(source_id)
    if not result:
        return None
    return result.model_dump()


@router.patch("/{source_id}", response_model=Dict[str, Any])
async def update_preprocessing_result(
    source_id: str,
    body: UpdatePreprocessingRequest,
    svc: PreprocessingService = Depends(get_preprocessing_service),
):
    """Update classification or chunk filtering for a source's preprocessing result."""
    result = await svc.get_result(source_id)
    if not result:
        raise HTTPException(status_code=404, detail="No preprocessing result found")

    updates = body.model_dump(exclude_unset=True)
    if not updates:
        return result.model_dump()

    updated = await svc.update_result(source_id, updates)
    return updated.model_dump()
