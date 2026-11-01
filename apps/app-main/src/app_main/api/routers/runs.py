"""Run-eindpunten voor Langflow (F2-04, ADR 0003, 0022 E, plan §5.4).

* ``POST /api/v1/runs``: maakt een run (``bezig``); 409 als de flow niet in Git
  staat en de invoer intern of onbekend is (de run is dan ``mislukt``).
* ``PATCH /api/v1/runs/{run_id}``: werkt een lopende run bij of sluit hem af;
  409 op een afgesloten run; 403 voor een andere sleutel dan de aanmaker (F2-14).
* ``GET /api/v1/runs/{run_id}``: de run.
* ``POST`` en ``GET /api/v1/runs/{run_id}/steps/{stap}``: een stap na de import
  starten en volgen (F2-15, ADR 0011); alleen de aanmaker van de run.

Recht ``pijplijn`` (en ``beheerder``).
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from shared.models.agents import Right
from shared.models.processing_run import ProcessingRun, RunStep, StapNaam

from app_main.api.agent_auth import AgentKeyContext, current_key, require_right
from app_main.api.schemas import (
    RunCreateRequest,
    RunCreateResponse,
    RunPatchRequest,
    RunResponse,
    StepResponse,
    StepStartRequest,
)
from app_main.dependencies import get_processing_run_service, get_run_step_orchestrator
from app_main.services.inname_binding import BindingError, ItemStatusError
from app_main.services.processing_run_service import (
    FlowNotRegisteredError,
    NietDeAanmakerError,
    ProcessingRunService,
    RunClosedError,
)
from app_main.services.run_step_orchestrator import (
    RunStepOrchestrator,
    StapNietGebondenError,
)

router = APIRouter(
    prefix="/v1/runs",
    tags=["runs"],
    dependencies=[Depends(require_right(Right.PIJPLIJN))],
)


def _antwoord(run: ProcessingRun) -> RunResponse:
    return RunResponse(**run.model_dump(exclude={"created", "updated"}))


@router.post("", status_code=201, response_model=RunCreateResponse)
async def create_run(
    body: RunCreateRequest,
    request: Request,
    service: ProcessingRunService = Depends(get_processing_run_service),
):
    # `handmatig` is alleen voor correcties door de beheerder (ADR 0005), met
    # de persoon van de sleutel als actor (migratie 92).
    actor = None
    key = await current_key(request)
    if body.trigger == "handmatig":
        if key.permission != Right.BEHEERDER:
            raise HTTPException(status_code=403, detail="Trigger handmatig vraagt een beheerder")
        actor = key.person
    try:
        run = await service.start(
            trigger=body.trigger,
            pipeline_id=body.pipeline_id,
            git_commit=body.pipeline_version.git_commit,
            flow_hash=body.pipeline_version.flow_hash,
            source_map_rule_id=body.source_map_rule_id,
            label_schema=body.label_schema,
            actor=actor,
            created_by_key=key.key_id,
            intake_batch_id=body.intake_batch_id,
            intake_item_id=body.intake_item_id,
        )
    except BindingError:
        raise HTTPException(status_code=409, detail={
            "code": "niet_gebonden", "melding": "Regel, batch of item horen niet bij elkaar"},
        ) from None
    except ItemStatusError:
        raise HTTPException(status_code=409, detail={
            "code": "verkeerde_status", "melding": "Status staat dit niet toe"}) from None
    except FlowNotRegisteredError as e:
        return JSONResponse(
            status_code=409,
            content={"detail": "Flow niet vastgelegd in Git", "run_id": e.run_id,
                     "status": "mislukt"},
        )
    return RunCreateResponse(run_id=str(run.id), status=run.status)


@router.patch("/{run_id}", response_model=RunResponse)
async def update_run(
    run_id: str,
    body: RunPatchRequest,
    request: Request,
    service: ProcessingRunService = Depends(get_processing_run_service),
) -> RunResponse:
    key: AgentKeyContext = await current_key(request)
    try:
        run = await service.update(
            run_id,
            status=body.status,
            ended_at=body.ended_at,
            log=[r.model_dump() for r in body.log],
            models=body.models,
            error=body.error,
            key_id=key.key_id,
        )
    except NietDeAanmakerError:
        raise HTTPException(status_code=403, detail="Alleen de aanmaker van de run") from None
    except RunClosedError:
        raise HTTPException(status_code=409, detail="Run is al afgesloten") from None
    return _antwoord(run)


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    service: ProcessingRunService = Depends(get_processing_run_service),
) -> RunResponse:
    return _antwoord(await service.get(run_id))


# --------------------------------------------------------------------------
# Stappen (F2-15, ADR 0011, plan §5.4)
# --------------------------------------------------------------------------


def _stap_antwoord(stap: RunStep) -> StepResponse:
    return StepResponse(
        step_id=str(stap.id), job_id=stap.job, source_version_id=stap.source_version,
        status=stap.status, attempts=stap.attempts, error=stap.error,
        review_item=stap.review_item,
    )


@router.post("/{run_id}/steps/{stap}", status_code=202, response_model=StepResponse)
async def start_step(
    run_id: str,
    stap: StapNaam,
    body: StepStartRequest,
    request: Request,
    orchestrator: RunStepOrchestrator = Depends(get_run_step_orchestrator),
) -> StepResponse:
    """Start een stap op een bronversie van de run; dezelfde aanroep geeft dezelfde stap."""
    key = await current_key(request)
    try:
        stap_rec = await orchestrator.start(run_id, stap, body.source_version_id,
                                            key_id=key.key_id)
    except NietDeAanmakerError:
        raise HTTPException(status_code=403, detail="Alleen de aanmaker van de run") from None
    except RunClosedError:
        raise HTTPException(status_code=409, detail={
            "code": "run_afgesloten", "melding": "Run is al afgesloten"}) from None
    except StapNietGebondenError:
        raise HTTPException(status_code=409, detail={
            "code": "niet_gebonden", "melding": "Bronversie hoort niet bij deze run"}) from None
    return _stap_antwoord(stap_rec)


@router.get("/{run_id}/steps/{stap}", response_model=StepResponse)
async def get_step(
    run_id: str,
    stap: StapNaam,
    request: Request,
    source_version_id: Optional[str] = None,
    orchestrator: RunStepOrchestrator = Depends(get_run_step_orchestrator),
) -> StepResponse:
    """De status van een stap; zonder ``source_version_id`` de laatst gestarte."""
    key = await current_key(request)
    try:
        stap_rec = await orchestrator.status(run_id, stap, key_id=key.key_id,
                                             source_version_id=source_version_id)
    except NietDeAanmakerError:
        raise HTTPException(status_code=403, detail="Alleen de aanmaker van de run") from None
    return _stap_antwoord(stap_rec)
