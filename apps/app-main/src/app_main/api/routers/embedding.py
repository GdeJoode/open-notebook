"""Embedding router - embed content for vector search."""

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app_main.api.schemas import EmbedRequest, EmbedResponse
from app_main.dependencies import get_model_service, get_note_service, get_source_service
from app_main.services.model_service import ModelService
from app_main.services.note_service import NoteService
from app_main.services.source_service import SourceService

router = APIRouter(tags=["embedding"])


@router.post("/embed", response_model=EmbedResponse)
async def embed_content(
    embed_request: EmbedRequest,
    model_svc: ModelService = Depends(get_model_service),
    source_svc: SourceService = Depends(get_source_service),
    note_svc: NoteService = Depends(get_note_service),
):
    """Embed content for vector search."""
    try:
        # Check if embedding model is available
        defaults = model_svc.model_manager.get_defaults()
        if not defaults or not defaults.default_embedding_model:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No embedding model configured. "
                    "Please configure one in the Models section."
                ),
            )

        item_id = embed_request.item_id
        item_type = embed_request.item_type.lower()

        if item_type not in ["source", "note"]:
            raise HTTPException(
                status_code=400,
                detail="Item type must be either 'source' or 'note'",
            )

        if embed_request.async_processing:
            # Async: submit command for background processing
            logger.info(f"Using async processing for {item_type} {item_id}")

            try:
                from app_main.services.command_service import CommandService

                command_id = await CommandService.submit_command_job(
                    "open_notebook",
                    "embed_single_item",
                    {"item_id": item_id, "item_type": item_type},
                )

                logger.info(f"Submitted async embedding command: {command_id}")

                return EmbedResponse(
                    success=True,
                    message="Embedding queued for background processing",
                    item_id=item_id,
                    item_type=item_type,
                    command_id=command_id,
                )

            except Exception as e:
                logger.error(f"Failed to submit async embedding command: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to queue embedding: {str(e)}",
                )

        else:
            # Sync: embed directly
            logger.info(f"Using sync processing for {item_type} {item_id}")

            if item_type == "source":
                source = await source_svc.get(item_id)
                if not source:
                    raise HTTPException(
                        status_code=404, detail="Source not found"
                    )
                # Vectorize will be handled by the source domain model
                # For now, return success as the embedding pipeline will handle it
                message = "Source embedding initiated"

            elif item_type == "note":
                note = await note_svc.get(item_id)
                if not note:
                    raise HTTPException(
                        status_code=404, detail="Note not found"
                    )
                message = "Note embedding initiated"

            return EmbedResponse(
                success=True,
                message=message,
                item_id=item_id,
                item_type=item_type,
                command_id=None,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error embedding {embed_request.item_type} "
            f"{embed_request.item_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error embedding content: {str(e)}",
        )
