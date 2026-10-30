"""Agent-key MANAGEMENT endpoints (Track G.1).

Sleutelbeheer: aanmaken, opvragen en intrekken. Alleen met recht ``beheerder``
(ADR 0002 B); ``pijplijn`` en de leesrechten krijgen 403.

The plaintext key is returned EXACTLY ONCE from the mint response and never again
(only its SHA-256 hash is stored). Listings expose ``key_prefix`` + metadata only.
"""

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from shared.models.agents import (
    AgentKeyCreate,
    AgentKeyCreated,
    AgentKeyPublic,
    Right,
)

from app_main.api.agent_auth import require_right
from app_main.services.agents.audit_service import AgentAuditService
from app_main.services.agents.key_service import AgentKeyService

router = APIRouter(
    prefix="/agent-keys",
    tags=["agent-keys"],
    dependencies=[Depends(require_right(Right.BEHEERDER))],
)


def get_agent_key_service() -> AgentKeyService:
    return AgentKeyService()


@router.post("", response_model=AgentKeyCreated, status_code=201)
async def mint_agent_key(
    body: AgentKeyCreate,
    service: AgentKeyService = Depends(get_agent_key_service),
) -> AgentKeyCreated:
    """Mint a new agent key. The plaintext is returned here ONCE — store it now."""
    try:
        public, plaintext = await service.mint(
            agent_id=body.agent_id,
            permission=body.permission,
            person=body.person,
            label=body.label,
            rate_limit_rpm=body.rate_limit_rpm,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"Minting agent key failed: {e}")
        raise HTTPException(status_code=500, detail=f"Could not mint key: {str(e)}")
    return AgentKeyCreated(key=plaintext, **public)


@router.get("", response_model=list[AgentKeyPublic])
async def list_agent_keys(
    service: AgentKeyService = Depends(get_agent_key_service),
) -> list[AgentKeyPublic]:
    """List all agent keys (secret-free: prefix + metadata, never the plaintext)."""
    rows = await service.list()
    return [AgentKeyPublic(**r) for r in rows]


@router.get("/{key_id}/audit-log")
async def get_key_audit_log(
    key_id: str,
    limit: int = 100,
    service: AgentKeyService = Depends(get_agent_key_service),
) -> dict:
    """A key's recent API calls (operator view, G.4).

    Resolves the key to its ``agent_id`` server-side and returns that agent's
    trail. Note the trail is keyed by ``agent_id`` — keys that SHARE an agent_id
    share the trail.
    """
    key = await service.get(key_id)
    if key is None:
        raise HTTPException(status_code=404, detail="key not found")
    limit = max(1, min(limit, 500))
    entries = await AgentAuditService().list_for_agent(key["agent_id"], limit=limit)
    return {"key_id": key_id, "agent_id": key["agent_id"], "entries": entries}


@router.delete("/{key_id:path}", status_code=204)
async def revoke_agent_key(
    key_id: str,
    service: AgentKeyService = Depends(get_agent_key_service),
) -> None:
    """Revoke a key by id (idempotent)."""
    await service.revoke(key_id)
