"""Herclassificatie van een bron (F2-03, ADR 0001, 0002 B).

``POST /api/v1/sources/{source_id}/classification``: alleen een sleutel met
recht ``beheerder`` (met persoon), met een verplichte reden. ``pijplijn`` en de
leesrechten krijgen 403. Er is geen route om een logregel te wijzigen of te
verwijderen.
"""

from fastapi import APIRouter, Depends, HTTPException
from shared.models.agents import Right
from surrealdb_service.repositories.source import (
    InhoudEldersInternError,
    InhoudGroepTeGrootError,
)

from app_main.api.agent_auth import AgentKeyContext, require_right
from app_main.api.fouten import fout_zonder_inhoud
from app_main.api.schemas import ClassificationRequest, ClassificationResponse
from app_main.dependencies import get_classification_service
from app_main.services.classification_service import ClassificationService

router = APIRouter(prefix="/v1/sources", tags=["classification"])


@router.post("/{source_id}/classification", response_model=ClassificationResponse)
async def reclassify_source(
    source_id: str,
    body: ClassificationRequest,
    key: AgentKeyContext = Depends(require_right(Right.BEHEERDER)),
    service: ClassificationService = Depends(get_classification_service),
) -> ClassificationResponse:
    try:
        uitkomst = await service.reclassify(
            source_id,
            body.classification,
            body.reason,
            permission=key.permission,
            person=key.person,
            key_id=key.key_id,
        )
    except PermissionError as e:
        raise fout_zonder_inhoud(e, "Geen toegang", status_code=403)
    except InhoudGroepTeGrootError as e:
        raise HTTPException(status_code=409, detail=e.detail) from None
    except InhoudEldersInternError as e:
        # De id's van de records, geen inhoud (F2-14).
        raise HTTPException(status_code=409, detail=e.detail) from None
    return ClassificationResponse(
        source_id=uitkomst.source_id,
        from_classification=uitkomst.from_classification,
        to_classification=uitkomst.to_classification,
        changed=uitkomst.changed,
        log_id=uitkomst.log_id,
    )
