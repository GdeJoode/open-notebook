"""Auditregels van modelaanroepen uit Langflow (F2-07, ADR 0007, plan §5.4).

``POST /api/v1/model-calls`` schrijft één regel in ``model_call``: tijd, run,
model, aanroepwijze, taak, classificatie, tokens, duur en status; ook voor
docling-serve en WhisperX. Een onbekend veld, zoals prompt- of antwoordtekst,
geeft 422. Recht ``pijplijn`` (en ``beheerder``).
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from llm_manager.gate import ModelCallRegel
from pydantic import BaseModel
from shared.models.agents import Right

from app_main.api.agent_auth import current_key, require_right
from app_main.services.model_routing.telemetry import (
    ModelCallTelemetry,
    ModelNietInRegisterError,
    NietDeAanmakerVanDeRunError,
)

router = APIRouter(
    prefix="/v1/model-calls",
    tags=["model-calls"],
    dependencies=[Depends(require_right(Right.PIJPLIJN))],
)


class ModelCallResponse(BaseModel):
    id: str


def get_model_call_telemetry() -> ModelCallTelemetry:
    return ModelCallTelemetry()


@router.post("", status_code=201, response_model=ModelCallResponse)
async def create_model_call(
    body: ModelCallRegel,
    request: Request,
    telemetry: ModelCallTelemetry = Depends(get_model_call_telemetry),
) -> ModelCallResponse:
    key = await current_key(request)
    try:
        regel_id = await telemetry.schrijf_van_langflow(body, key_id=key.key_id)
    except ModelNietInRegisterError:
        raise HTTPException(status_code=422, detail={
            "code": "model_niet_in_register",
            "melding": "Model, aanroepwijze of locatie niet in het register"}) from None
    except (LookupError, ValueError):
        raise HTTPException(status_code=404, detail="Run niet gevonden") from None
    except NietDeAanmakerVanDeRunError:
        raise HTTPException(status_code=403, detail="Alleen de aanmaker van de run") from None
    return ModelCallResponse(id=regel_id)
