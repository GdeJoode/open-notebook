"""Auth router: status (zonder sleutel) en het bereik van de eigen sleutel."""

from fastapi import APIRouter, Depends
from shared.models.agents import Right

from app_main.api.agent_auth import AgentKeyContext, require_right

router = APIRouter(prefix="/auth", tags=["auth"])

#: ``/api/v1/auth``: het bereik van de aanroepende sleutel (F2-01).
scope_router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.get("/status")
async def get_auth_status():
    """Meldt alleen dat een sleutel nodig is; geen inhoud, geen sleutel nodig."""
    return {"auth_enabled": True, "message": "Authentication is required"}


@scope_router.get("/scope")
async def get_auth_scope(
    key: AgentKeyContext = Depends(require_right(Right.LEZEN_EXTERN)),
) -> dict:
    """Het recht en het ``ClassificationScope`` van de aanroepende sleutel.

    Voor de startcontrole van open-notebook-mcp (F3-10) en de gateway (F5-07).
    Geeft niets anders terug: geen id, geen persoon, geen label. Een sleutel
    ``ophalen`` krijgt 403 (ADR 0002 B: alleen de ophaalcontrole).
    """
    return {"right": key.permission.value, "scope": key.scope.value}
