"""Model-routing config + provider-health router (Track J.5).

Surfaces the J.1-J.4 routing backend so the config is usable without DB edits:

* ``GET  /api/model-routes``         — the effective provider chain per task
  (the persisted ``model_route`` row when present, otherwise the J.1 hard-coded
  default chain, so the UI always shows what *would* run).
* ``PUT  /api/model-routes/{task}``  — upsert a task's ``provider_chain`` (+
  optional ``private_chain``) with reordering / enable-disable. Validates the
  task name and every referenced ``model_id``.
* ``GET  /api/model-routes/health``  — per-provider ``configured`` (API key
  present), ``is_local``, circuit-breaker state, and recent fallback counts
  read from the J.4 ``routing.served`` telemetry.

The endpoints are intentionally thin: resolution / privacy / failover logic all
lives in the ``model_routing`` service package. This router only reads the
effective shape and persists chain edits.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from llm_manager.providers.registry import get_provider_config, list_providers
from loguru import logger
from pydantic import BaseModel, Field
from shared.models.llm import ModelRoute

from app_main.dependencies import (
    get_circuit_breaker_registry,
    get_model_route_repo,
    get_model_service,
)
from app_main.services.model_routing.route_resolver import (
    _DEFAULT_CLOUD_PROVIDERS,
    _DEFAULT_LOCAL_PROVIDER,
    _DEFAULT_PROVIDER_MODEL_NAME,
)

router = APIRouter(prefix="/model-routes", tags=["model-routes"])

# The three routed LLM stages (mirror of ``LLMTask`` values). Embeddings and
# parsing are deliberately absent — the §1.2 guardrail keeps them local.
_VALID_TASKS = ["entity_extraction", "summarization", "chat"]


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ProviderChainEntry(BaseModel):
    """One ordered entry in a task's provider chain."""

    provider: str = Field(..., description="Provider name (nvidia, openai, ollama, ...)")
    model_id: Optional[str] = Field(
        None, description="Pinned model record id; null uses the per-task default"
    )
    enabled: bool = Field(True, description="Whether this entry participates in routing")


class ModelRouteResponse(BaseModel):
    """The effective route for one task as the UI should render it."""

    task: str
    provider_chain: List[ProviderChainEntry]
    private_chain: List[ProviderChainEntry]
    is_default: bool = Field(
        False,
        description="True when no row is persisted yet and this is the J.1 default chain",
    )


class ModelRouteUpdate(BaseModel):
    """Upsert payload for a task's chain (reorder / enable-disable)."""

    provider_chain: List[ProviderChainEntry]
    private_chain: Optional[List[ProviderChainEntry]] = None


class ProviderHealth(BaseModel):
    """Per-provider health for the status chips."""

    provider: str
    is_local: bool
    configured: bool = Field(
        ..., description="API key env var present (always True for local providers)"
    )
    circuit_state: str = Field(
        ..., description="closed | open | half_open (advisory, in-process)"
    )
    recent_fallback_count: int = Field(
        0, description="Times this provider was failed-over-FROM in recent telemetry"
    )


class ModelRouteHealthResponse(BaseModel):
    providers: List[ProviderHealth]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _default_chain_entries(local_only: bool) -> List[ProviderChainEntry]:
    """Build the hard-coded J.1 default chain as response entries.

    Mirrors ``route_resolver._entries_for(route_row=None)`` so the UI's empty
    state shows exactly what the resolver would run. CLOUD chain is the cloud
    providers followed by the local fallback; the private chain is local-only.
    """
    providers = (
        [_DEFAULT_LOCAL_PROVIDER]
        if local_only
        else _DEFAULT_CLOUD_PROVIDERS + [_DEFAULT_LOCAL_PROVIDER]
    )
    return [
        ProviderChainEntry(
            provider=p,
            model_id=_DEFAULT_PROVIDER_MODEL_NAME.get(p),
            enabled=True,
        )
        for p in providers
    ]


def _entries_from_chain(chain: List[Dict[str, Any]]) -> List[ProviderChainEntry]:
    """Coerce persisted FLEXIBLE chain dicts into response entries."""
    entries: List[ProviderChainEntry] = []
    for raw in chain:
        provider = raw.get("provider")
        if not provider:
            continue
        entries.append(
            ProviderChainEntry(
                provider=provider,
                model_id=raw.get("model_id"),
                enabled=raw.get("enabled", True),
            )
        )
    return entries


def _route_to_response(task: str, route: Optional[ModelRoute]) -> ModelRouteResponse:
    """Render a persisted row, or the default chain when no row exists."""
    if route is None:
        return ModelRouteResponse(
            task=task,
            provider_chain=_default_chain_entries(local_only=False),
            private_chain=_default_chain_entries(local_only=True),
            is_default=True,
        )
    return ModelRouteResponse(
        task=task,
        provider_chain=_entries_from_chain(route.provider_chain),
        private_chain=_entries_from_chain(route.private_chain),
        is_default=False,
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("", response_model=List[ModelRouteResponse])
async def list_model_routes(
    route_repo=Depends(get_model_route_repo),
) -> List[ModelRouteResponse]:
    """Return the effective route for all three tasks.

    A task without a persisted row falls back to the J.1 default chain
    (``is_default=True``) so the UI can show an editable read-only default with a
    "Save to customize" affordance.
    """
    persisted = {r.task: r for r in await route_repo.list_all()}
    return [_route_to_response(task, persisted.get(task)) for task in _VALID_TASKS]


@router.put("/{task}", response_model=ModelRouteResponse)
async def update_model_route(
    task: str,
    body: ModelRouteUpdate,
    route_repo=Depends(get_model_route_repo),
    model_service=Depends(get_model_service),
) -> ModelRouteResponse:
    """Upsert a task's provider chain (reorder / enable-disable).

    Validates the task name and that every pinned ``model_id`` resolves to an
    existing model row; an unknown provider or model id is a 400 so a typo in
    the editor never silently produces a dead chain.
    """
    if task not in _VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task {task!r}; expected one of {_VALID_TASKS}",
        )

    if not body.provider_chain:
        raise HTTPException(
            status_code=400,
            detail="provider_chain must contain at least one provider",
        )

    private_chain = body.private_chain or []
    # (chain entries, is private-mode chain) — the private chain carries the
    # extra LOCAL-only contract from ``ModelRoute.private_chain``.
    for chain, is_private in ((body.provider_chain, False), (private_chain, True)):
        seen: set[str] = set()
        for entry in chain:
            if entry.provider in seen:
                raise HTTPException(
                    status_code=400,
                    detail=f"Duplicate provider {entry.provider!r} in chain",
                )
            seen.add(entry.provider)

            cfg = get_provider_config(entry.provider)
            if cfg is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown provider {entry.provider!r}",
                )
            if is_private and not cfg.is_local:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Provider {entry.provider!r} is not local and cannot be "
                        "used in the private chain (PRIVATE mode is LOCAL-only)"
                    ),
                )
            if entry.model_id:
                model = await model_service.get(entry.model_id)
                if model is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown model id {entry.model_id!r}",
                    )

    route = ModelRoute(
        task=task,
        provider_chain=[e.model_dump() for e in body.provider_chain],
        private_chain=[e.model_dump() for e in (body.private_chain or [])],
    )
    await route_repo.upsert(route)
    logger.info(
        "Upserted model_route for task {} ({} provider(s))",
        task,
        len(body.provider_chain),
    )

    saved = await route_repo.get_by_task(task)
    return _route_to_response(task, saved)


@router.get("/health", response_model=ModelRouteHealthResponse)
async def model_route_health(
    registry=Depends(get_circuit_breaker_registry),
) -> ModelRouteHealthResponse:
    """Per-provider configured/health + circuit state + recent fallback counts.

    ``configured`` reuses the existing registry check (``api_key_env_var``
    presence via ``os.environ``); local providers are always configured. The
    circuit state is an advisory peek at the in-process breaker registry. Recent
    fallback counts come from the J.4 ``routing.served`` telemetry — a provider
    that appears in a served event's ``fallback_from`` was failed over from.
    """
    fallback_counts = await _recent_fallback_counts()

    providers: List[ProviderHealth] = []
    for cfg in list_providers():
        configured = cfg.is_local or bool(os.environ.get(cfg.api_key_env_var))
        breaker = registry.get_nowait(cfg.name)
        state = breaker._state.value if breaker is not None else "closed"  # noqa: SLF001
        providers.append(
            ProviderHealth(
                provider=cfg.name,
                is_local=cfg.is_local,
                configured=configured,
                circuit_state=state,
                recent_fallback_count=fallback_counts.get(cfg.name, 0),
            )
        )
    return ModelRouteHealthResponse(providers=providers)


async def _recent_fallback_counts(limit: int = 200) -> Dict[str, int]:
    """Count how often each provider was failed-over-FROM in recent telemetry.

    Reads the most recent ``routing.served`` metric rows and tallies every
    provider listed in each event's ``fallback_from``. Best-effort: a telemetry
    read failure returns empty counts rather than failing the health endpoint.
    """
    try:
        from surrealdb_service.connection import execute_query

        rows = await execute_query(
            "SELECT payload FROM metrics WHERE event_type = 'routing.served' "
            "ORDER BY created DESC LIMIT $limit;",
            {"limit": limit},
        )
    except Exception as e:  # noqa: BLE001 — health must not fail on telemetry
        logger.warning(f"model-routes health: fallback-count read failed: {e}")
        return {}

    counts: Dict[str, int] = {}
    for row in rows or []:
        payload = row.get("payload") or {}
        for provider in payload.get("fallback_from") or []:
            counts[provider] = counts.get(provider, 0) + 1
    return counts
