"""Transformations router - CRUD and execution for text transformations."""

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app_main.api.schemas import (
    DefaultPromptResponse,
    DefaultPromptUpdate,
    TransformationCreate,
    TransformationExecuteRequest,
    TransformationExecuteResponse,
    TransformationResponse,
    TransformationUpdate,
)
from app_main.dependencies import get_transformation_service
from app_main.services.transformation_service import TransformationService

router = APIRouter(prefix="/transformations", tags=["transformations"])


def _transformation_to_response(t) -> TransformationResponse:
    """Convert a Transformation domain model to a TransformationResponse schema."""
    return TransformationResponse(
        id=t.id,
        name=t.name,
        title=t.title,
        description=t.description,
        prompt=t.prompt,
        apply_default=t.apply_default,
        created=str(t.created),
        updated=str(t.updated),
    )


@router.get("", response_model=list[TransformationResponse])
async def list_transformations(
    transformation_service: TransformationService = Depends(
        get_transformation_service,
    ),
):
    """List all transformations."""
    transformations = await transformation_service.get_all()
    return [_transformation_to_response(t) for t in transformations]


@router.post("", response_model=TransformationResponse, status_code=201)
async def create_transformation(
    body: TransformationCreate,
    transformation_service: TransformationService = Depends(
        get_transformation_service,
    ),
):
    """Create a new transformation."""
    transformation = await transformation_service.create(body.model_dump())
    logger.info("Created transformation {}", transformation.id)
    return _transformation_to_response(transformation)


@router.get("/default-prompt", response_model=DefaultPromptResponse)
async def get_default_prompt(
    transformation_service: TransformationService = Depends(
        get_transformation_service,
    ),
):
    """Get the default transformation prompt."""
    prompts = await transformation_service.get_default_prompts()
    return DefaultPromptResponse(
        transformation_instructions=prompts.transformation_instructions or "",
    )


@router.put("/default-prompt", response_model=DefaultPromptResponse)
async def update_default_prompt(
    body: DefaultPromptUpdate,
    transformation_service: TransformationService = Depends(
        get_transformation_service,
    ),
):
    """Update the default transformation prompt."""
    prompts = await transformation_service.update_default_prompts(
        {"transformation_instructions": body.transformation_instructions},
    )
    logger.info("Updated default transformation prompt")
    return DefaultPromptResponse(
        transformation_instructions=prompts.transformation_instructions or "",
    )


@router.get("/{transformation_id}", response_model=TransformationResponse)
async def get_transformation(
    transformation_id: str,
    transformation_service: TransformationService = Depends(
        get_transformation_service,
    ),
):
    """Get a transformation by ID."""
    transformation = await transformation_service.get(transformation_id)
    if not transformation:
        raise HTTPException(status_code=404, detail="Transformation not found")
    return _transformation_to_response(transformation)


@router.put("/{transformation_id}", response_model=TransformationResponse)
async def update_transformation(
    transformation_id: str,
    body: TransformationUpdate,
    transformation_service: TransformationService = Depends(
        get_transformation_service,
    ),
):
    """Update a transformation."""
    existing = await transformation_service.get(transformation_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Transformation not found")

    update_data = body.model_dump(exclude_none=True)
    if not update_data:
        return _transformation_to_response(existing)

    transformation = await transformation_service.update(
        transformation_id, update_data,
    )
    logger.info("Updated transformation {}", transformation_id)
    return _transformation_to_response(transformation)


@router.delete("/{transformation_id}", status_code=204)
async def delete_transformation(
    transformation_id: str,
    transformation_service: TransformationService = Depends(
        get_transformation_service,
    ),
):
    """Delete a transformation."""
    existing = await transformation_service.get(transformation_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Transformation not found")
    await transformation_service.delete(transformation_id)
    logger.info("Deleted transformation {}", transformation_id)


@router.post("/execute", response_model=TransformationExecuteResponse)
async def execute_transformation(
    body: TransformationExecuteRequest,
    transformation_service: TransformationService = Depends(
        get_transformation_service,
    ),
):
    """Execute a transformation on input text.

    This is a placeholder endpoint. Full implementation requires graph-based
    execution which will be integrated later.
    """
    transformation = await transformation_service.get(body.transformation_id)
    if not transformation:
        raise HTTPException(status_code=404, detail="Transformation not found")

    # Placeholder: graph-based execution will be added later
    raise HTTPException(
        status_code=501,
        detail="Transformation execution requires graph integration (not yet implemented)",
    )
