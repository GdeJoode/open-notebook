"""Context router - notebook context builder for chat and analysis."""

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app_main.api.schemas import ContextRequest, ContextResponse
from app_main.dependencies import (
    get_notebook_service,
    get_source_service,
)
from app_main.exceptions import InvalidInputError
from app_main.services.notebook_service import NotebookService
from app_main.services.source_service import SourceService
from app_main.api.agent_auth import RECHT_BEHEERDER
from shared.models.classification import ClassificationScope
from app_main.api.fouten import fout_zonder_inhoud

router = APIRouter(tags=["context"], dependencies=[RECHT_BEHEERDER])


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
):
    """Get context for a notebook based on configuration."""
    try:
        notebook = await notebook_svc.get(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        # F2-05 (ADR 0004): de context voor een model bestaat alleen uit
        # fragmenten: de tekst van de huidige versie van elke gekozen bron.
        # Inzichten en notities zijn afgeleid en gaan niet mee; "note" blijft
        # leeg. Deze route is alleen voor de beheerder (bereik ALLES).
        context_data: dict[str, list[dict]] = {"source": [], "note": []}
        total_content = ""

        if context_request.context_config:
            bron_ids = [
                sid
                for sid, status in context_request.context_config.sources.items()
                if "not in" not in status
            ]
        else:
            bron_ids = [
                sd.get("id") if isinstance(sd, dict) else sd.id
                for sd in await notebook_svc.get_sources(notebook_id)
            ]
        for source_id in bron_ids:
            try:
                full_source_id = (
                    source_id if str(source_id).startswith("source:") else f"source:{source_id}"
                )
                source = await source_svc.get_scoped(
                    full_source_id, scope=ClassificationScope.ALLES
                )
                if not source:
                    continue
                source_context = {
                    "id": source.id,
                    "title": source.title,
                    "full_text": await source_svc.get_text(
                        str(source.id), scope=ClassificationScope.ALLES
                    ),
                }
                context_data["source"].append(source_context)
                total_content += str(source_context)
            except Exception as e:
                logger.warning(f"Error processing source {source_id}: {e}")
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
        raise fout_zonder_inhoud(e, "Ongeldig verzoek", status_code=400)
    except Exception as e:
        logger.error(f"Error getting context for notebook {notebook_id}: {e}")
        raise fout_zonder_inhoud(e, "Error getting context")
