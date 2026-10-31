"""Models router: het modelregister, alleen lezen (ADR 0007).

``GET /api/models`` toont ``config/modelregister.yaml``. Er zijn geen modelrijen,
standaardmodellen of providers in de database meer (F2-09); welk model een taak
uitvoert, staat in het register.
"""

from fastapi import APIRouter
from llm_manager.register import RegisterModel, get_register

from app_main.api.agent_auth import RECHT_BEHEERDER

router = APIRouter(prefix="/models", tags=["models"], dependencies=[RECHT_BEHEERDER])


@router.get("", response_model=list[RegisterModel])
async def list_models():
    """De modellen uit het register; alleen lezen."""
    return get_register().modellen
