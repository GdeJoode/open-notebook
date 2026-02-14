"""Context router - notebook context builder for chat and analysis."""

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app_main.api.schemas import ContextRequest, ContextResponse
from app_main.dependencies import (
    get_notebook_service,
    get_note_service,
    get_source_service,
)
from app_main.exceptions import InvalidInputError
from app_main.services.note_service import NoteService
from app_main.services.notebook_service import NotebookService
from app_main.services.source_service import SourceService

router = APIRouter(tags=["context"])


def _token_count(content: str) -> int:
    """Estimate token count for a string."""
    from shared.utils import token_count

    return token_count(content)


@router.post("/notebooks/{notebook_id}/context", response_model=ContextResponse)
async def get_notebook_context(
    notebook_id: str,
    context_request: ContextRequest,
    notebook_svc: NotebookService = Depends(get_notebook_service),
    source_svc: SourceService = Depends(get_source_service),
    note_svc: NoteService = Depends(get_note_service),
):
    """Get context for a notebook based on configuration."""
    try:
        notebook = await notebook_svc.get(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        context_data: dict[str, list[dict]] = {"note": [], "source": []}
        total_content = ""

        if context_request.context_config:
            # Process sources from config
            for source_id, status in context_request.context_config.sources.items():
                if "not in" in status:
                    continue

                try:
                    full_source_id = (
                        source_id
                        if source_id.startswith("source:")
                        else f"source:{source_id}"
                    )

                    source = await source_svc.get(full_source_id)
                    if not source:
                        continue

                    if "insights" in status:
                        insights = await source_svc.get_insights(full_source_id)
                        source_context = {
                            "id": source.id,
                            "title": source.title,
                            "insights": [
                                {
                                    "type": i.insight_type,
                                    "content": i.content,
                                }
                                for i in insights
                            ],
                        }
                    elif "full content" in status:
                        source_context = {
                            "id": source.id,
                            "title": source.title,
                            "full_text": source.full_text,
                        }
                    else:
                        continue

                    context_data["source"].append(source_context)
                    total_content += str(source_context)
                except Exception as e:
                    logger.warning(f"Error processing source {source_id}: {e}")
                    continue

            # Process notes from config
            for note_id, status in context_request.context_config.notes.items():
                if "not in" in status:
                    continue

                try:
                    full_note_id = (
                        note_id
                        if note_id.startswith("note:")
                        else f"note:{note_id}"
                    )
                    note = await note_svc.get(full_note_id)
                    if not note:
                        continue

                    if "full content" in status:
                        note_context = {
                            "id": note.id,
                            "title": note.title,
                            "content": note.content,
                        }
                        context_data["note"].append(note_context)
                        total_content += str(note_context)
                except Exception as e:
                    logger.warning(f"Error processing note {note_id}: {e}")
                    continue
        else:
            # Default: include all sources (short context) and notes
            sources = await notebook_svc.get_sources(notebook_id)
            for source_data in sources:
                try:
                    sid = source_data.get("id") if isinstance(source_data, dict) else source_data.id
                    source = await source_svc.get(sid)
                    if not source:
                        continue
                    insights = await source_svc.get_insights(sid)
                    source_context = {
                        "id": sid,
                        "title": source.title,
                        "insights": [
                            {
                                "type": i.insight_type,
                                "content": i.content,
                            }
                            for i in insights
                        ],
                    }
                    context_data["source"].append(source_context)
                    total_content += str(source_context)
                except Exception as e:
                    logger.warning(f"Error processing source: {e}")
                    continue

            notes = await notebook_svc.get_notes(notebook_id)
            for note_data in notes:
                try:
                    note_context = {
                        "id": note_data.get("id") if isinstance(note_data, dict) else note_data.id,
                        "title": note_data.get("title") if isinstance(note_data, dict) else note_data.title,
                        "content": note_data.get("content") if isinstance(note_data, dict) else note_data.content,
                    }
                    context_data["note"].append(note_context)
                    total_content += str(note_context)
                except Exception as e:
                    logger.warning(f"Error processing note: {e}")
                    continue

        estimated_tokens = _token_count(total_content) if total_content else 0

        return ContextResponse(
            notebook_id=notebook_id,
            sources=context_data["source"],
            notes=context_data["note"],
            total_tokens=estimated_tokens,
        )

    except HTTPException:
        raise
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting context for notebook {notebook_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting context: {str(e)}",
        )
