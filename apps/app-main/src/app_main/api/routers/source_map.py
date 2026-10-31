"""Bronkaartregels (F2-13, ADR 0020, plan §5.2).

* ``GET /api/v1/source-map-rules[/{id}]``: lezen met ``pijplijn`` of
  ``beheerder``. ``lezen_extern`` en ``lezen_intern`` krijgen 403: de bronkaart
  is configuratie, geen kennis.
* ``POST /api/v1/source-map-rules``, ``PATCH /api/v1/source-map-rules/{id}`` en
  ``POST /api/v1/source-map-rules/{id}/confirm``: alleen ``beheerder``.

Een locatie in de vault of de exportmap geeft 422; een verouderde ``rev`` 409.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from shared.models.agents import Right
from shared.models.source_map import SourceMapRule
from surrealdb_service.repositories.base import RecordNotFoundError, RevConflictError

from app_main.api.agent_auth import AgentKeyContext, require_right
from app_main.api.schemas import (
    ConfirmRequest,
    SourceMapRuleCreate,
    SourceMapRulePatch,
    SourceMapRuleResponse,
)
from app_main.dependencies import get_source_map_service
from app_main.services.intake_service import IntakeConflictError
from app_main.services.source_map_service import SourceMapService, VerbodenLocatieError

router = APIRouter(prefix="/v1/source-map-rules", tags=["source-map"])


def antwoord(regel: SourceMapRule) -> SourceMapRuleResponse:
    return SourceMapRuleResponse(
        **regel.model_dump(exclude={"created", "updated", "created_at"}),
        effective_classification=regel.effectieve_classificatie,
    )


def _opslag(body: Any, velden: Dict[str, Any]) -> Dict[str, Any]:
    """De velden voor de database; ``label_schema`` met de sleutel ``schema``."""
    if getattr(body, "label_schema", None) is not None:
        velden["label_schema"] = body.label_schema.opslag()
    if getattr(body, "schedule", None) is not None:
        # Volledig, ook bij PATCH: een DEFAULT onder `schedule` geldt onder
        # v2.7.0 niet bij een UPDATE.
        velden["schedule"] = body.schedule.model_dump()
    return velden


def fout_naar_http(e: Exception) -> HTTPException:
    """Vaste meldingen, zonder tekst van de uitzondering (ADR 0002 C)."""
    if isinstance(e, RevConflictError):
        return HTTPException(status_code=409, detail="Intussen gewijzigd; laad opnieuw")
    if isinstance(e, RecordNotFoundError):
        return HTTPException(status_code=404, detail="Niet gevonden")
    if isinstance(e, VerbodenLocatieError):
        return HTTPException(status_code=422, detail="Locatie niet toegestaan")
    if isinstance(e, PermissionError):
        return HTTPException(status_code=403, detail="Onvoldoende recht")
    if isinstance(e, IntakeConflictError):
        return HTTPException(status_code=409, detail="Status staat dit niet toe")
    return HTTPException(status_code=500, detail="Interne fout")


@router.get("", response_model=List[SourceMapRuleResponse],
            dependencies=[Depends(require_right(Right.PIJPLIJN))])
async def list_rules(
    active: Optional[bool] = None,
    service: SourceMapService = Depends(get_source_map_service),
) -> List[SourceMapRuleResponse]:
    return [antwoord(r) for r in await service.list(active=active)]


@router.get("/{rule_id}", response_model=SourceMapRuleResponse,
            dependencies=[Depends(require_right(Right.PIJPLIJN))])
async def get_rule(
    rule_id: str,
    service: SourceMapService = Depends(get_source_map_service),
) -> SourceMapRuleResponse:
    return antwoord(await service.get(rule_id))


@router.post("", status_code=201, response_model=SourceMapRuleResponse)
async def create_rule(
    body: SourceMapRuleCreate,
    key: AgentKeyContext = Depends(require_right(Right.BEHEERDER)),
    service: SourceMapService = Depends(get_source_map_service),
) -> SourceMapRuleResponse:
    velden = _opslag(body, body.model_dump(exclude={"confirm", "reason"}))
    try:
        regel = await service.create(
            velden, bevestig=body.confirm, permission=key.permission,
            person=key.person, key_id=key.key_id, reason=body.reason,
        )
    except (VerbodenLocatieError, PermissionError) as e:
        raise fout_naar_http(e) from None
    return antwoord(regel)


@router.patch("/{rule_id}", response_model=SourceMapRuleResponse)
async def update_rule(
    rule_id: str,
    body: SourceMapRulePatch,
    key: AgentKeyContext = Depends(require_right(Right.BEHEERDER)),
    service: SourceMapService = Depends(get_source_map_service),
) -> SourceMapRuleResponse:
    velden = _opslag(body, body.model_dump(exclude_unset=True, exclude={"rev", "confirm", "reason"}))
    try:
        regel = await service.update(
            rule_id, velden, rev=body.rev, bevestig=body.confirm,
            permission=key.permission, person=key.person, key_id=key.key_id,
            reason=body.reason,
        )
    except (RevConflictError, RecordNotFoundError, VerbodenLocatieError, PermissionError) as e:
        raise fout_naar_http(e) from None
    return antwoord(regel)


@router.post("/{rule_id}/confirm", response_model=SourceMapRuleResponse)
async def confirm_rule(
    rule_id: str,
    body: ConfirmRequest,
    key: AgentKeyContext = Depends(require_right(Right.BEHEERDER)),
    service: SourceMapService = Depends(get_source_map_service),
) -> SourceMapRuleResponse:
    try:
        regel = await service.confirm(
            rule_id, rev=body.rev, permission=key.permission, person=key.person,
            key_id=key.key_id, reason=body.reason,
        )
    except (RevConflictError, RecordNotFoundError, PermissionError) as e:
        raise fout_naar_http(e) from None
    return antwoord(regel)
