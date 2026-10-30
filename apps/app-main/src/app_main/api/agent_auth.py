"""Sleutels en rechten per route, standaard weigeren (F2-01, ADR 0002 A/B).

Elk verzoek draagt ``Authorization: Bearer <sleutel>``. :func:`authenticate_request`
zoekt de sleutel op (SHA-256, niet ingetrokken) en geeft een
:class:`AgentKeyContext` met het recht en de persoon. De sleutelmiddleware
(``auth.py``) doet dat voor elk verzoek behalve :data:`ZONDER_SLEUTEL`; een route
declareert haar minimale recht met :func:`require_right`. Een route zonder
declaratie faalt in ``tests/test_route_rechten.py``.

Rechten (ADR 0002 B): ``lezen_extern`` < ``lezen_intern`` < ``pijplijn`` <
``beheerder``. ``ophalen`` staat daarbuiten: het mag alleen een route die
``ophalen`` declareert (de ophaalcontrole, F4-13), en ziet geen records.
``beheerder`` mag alles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from fastapi import Depends, HTTPException, Request
from shared.models.agents import SCOPE_PER_RIGHT, ClassificationScope, Right
from starlette.middleware.base import BaseHTTPMiddleware

from app_main.services.agents.key_service import AgentKeyService

#: De enige paden zonder sleutelplicht. Geen van beide geeft inhoud terug.
#: Daarnaast beantwoordt ``CORSMiddleware`` een CORS-preflight zelf, zonder
#: sleutel en zonder inhoud (``auth.py``; ``allow_origins`` beperken: F2-20).
ZONDER_SLEUTEL = frozenset({"/health", "/api/auth/status"})

#: Per minimaal recht van een route: de rechten die de route mogen aanroepen.
_TOEGESTAAN: dict[Right, frozenset[Right]] = {
    Right.LEZEN_EXTERN: frozenset(
        {Right.LEZEN_EXTERN, Right.LEZEN_INTERN, Right.PIJPLIJN, Right.BEHEERDER}
    ),
    Right.LEZEN_INTERN: frozenset({Right.LEZEN_INTERN, Right.PIJPLIJN, Right.BEHEERDER}),
    Right.PIJPLIJN: frozenset({Right.PIJPLIJN, Right.BEHEERDER}),
    Right.OPHALEN: frozenset({Right.OPHALEN, Right.BEHEERDER}),
    Right.BEHEERDER: frozenset({Right.BEHEERDER}),
}

_WEIGER_401 = {"WWW-Authenticate": "Bearer"}


@dataclass(frozen=True)
class AgentKeyContext:
    """De geauthenticeerde sleutel van een verzoek."""

    key_id: str
    agent_id: str
    permission: Right
    person: Optional[str] = None
    rate_limit_rpm: Optional[int] = None

    @property
    def scope(self) -> ClassificationScope:
        return SCOPE_PER_RIGHT[self.permission]


def bearer_token(request: Request) -> Optional[str]:
    """De sleutel uit ``Authorization: Bearer <sleutel>``, anders ``None``."""
    header = request.headers.get("Authorization") or ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


async def authenticate_request(request: Request) -> AgentKeyContext:
    """Zoek de sleutel van het verzoek op; 401 als die ontbreekt of ongeldig is.

    Een rij met een recht buiten de vijf (bijvoorbeeld een oude ``read``-sleutel)
    of een ``beheerder`` zonder persoon geldt als ongeldig: standaard weigeren.
    """
    token = bearer_token(request)
    if token is None:
        raise HTTPException(status_code=401, detail="Sleutel ontbreekt", headers=_WEIGER_401)
    row = await AgentKeyService().authenticate(token)
    if row is None:
        raise HTTPException(status_code=401, detail="Ongeldige sleutel", headers=_WEIGER_401)
    try:
        permission = Right(str(row.get("permission")))
    except ValueError:
        raise HTTPException(
            status_code=401, detail="Ongeldige sleutel", headers=_WEIGER_401
        ) from None
    person = (row.get("person") or "").strip() or None
    if permission == Right.BEHEERDER and person is None:
        raise HTTPException(status_code=401, detail="Ongeldige sleutel", headers=_WEIGER_401)
    return AgentKeyContext(
        key_id=str(row.get("id")),
        agent_id=str(row.get("agent_id")),
        permission=permission,
        person=person,
        rate_limit_rpm=row.get("rate_limit_rpm"),
    )


async def current_key(request: Request) -> AgentKeyContext:
    """De sleutel van dit verzoek, zoals de middleware die vond.

    Ontbreekt die (een router zonder de middleware, zoals in een test), dan
    authenticeert deze functie zelf. Zo valt een route nooit open.
    """
    ctx = getattr(request.state, "agent_key", None)
    if ctx is None:
        ctx = await authenticate_request(request)
        request.state.agent_key = ctx
    return ctx


def require_right(minimum: Right | str):
    """Dependency: de route vraagt minimaal recht ``minimum``; anders 403.

    Per recht één en dezelfde functie, met ``required_right`` als kenmerk,
    zodat de routetest de declaratie kan vinden.
    """
    return _require_right(Right(minimum))


@lru_cache(maxsize=None)
def _require_right(minimum: Right):
    toegestaan = _TOEGESTAAN[minimum]

    async def _dependency(request: Request) -> AgentKeyContext:
        ctx = await current_key(request)
        if ctx.permission not in toegestaan:
            raise HTTPException(status_code=403, detail="Onvoldoende recht")
        return ctx

    _dependency.required_right = minimum  # type: ignore[attr-defined]
    _dependency.__name__ = f"require_right_{minimum.value}"
    return _dependency


#: Kant-en-klare declaraties voor ``dependencies=[...]`` van een router of route.
RECHT_LEZEN_EXTERN = Depends(require_right(Right.LEZEN_EXTERN))
RECHT_BEHEERDER = Depends(require_right(Right.BEHEERDER))


class AgentAuditMiddleware(BaseHTTPMiddleware):
    """Logboek per aanroep, voor alle routes (ADR 0002 A).

    Ook mislukte aanroepen (401/403) komen erin, zonder sleutel. Uitgezonderd:
    ``/health`` (de healthcheck van Docker, zonder inhoud) en een afgeknepen
    verzoek (429), dat anders zelf een schrijfvloed zou worden. Het logboek
    beïnvloedt het antwoord nooit.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        start = time.time()
        response = await call_next(request)
        latency_ms = int((time.time() - start) * 1000)
        if response.status_code == 429:
            return response
        ctx = getattr(request.state, "agent_key", None)
        try:
            from app_main.services.agents.audit_service import AgentAuditService

            await AgentAuditService().record(
                agent_id=ctx.agent_id if ctx else "unknown",
                agent_key_id=ctx.key_id if ctx else None,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                latency_ms=latency_ms,
                # process-url stamps the enqueued job id here.
                job_id=getattr(request.state, "job_id", None),
            )
        except Exception:  # noqa: BLE001 — audit must never affect the response
            pass
        return response


__all__ = [
    "RECHT_BEHEERDER",
    "RECHT_LEZEN_EXTERN",
    "ZONDER_SLEUTEL",
    "AgentAuditMiddleware",
    "AgentKeyContext",
    "authenticate_request",
    "bearer_token",
    "current_key",
    "require_right",
]
