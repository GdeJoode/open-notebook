"""Sleutelmiddleware: geen verzoek zonder geldige sleutel (F2-01, ADR 0002 A).

Vervangt de wachtwoordmiddleware en ``OPEN_NOTEBOOK_PASSWORD``. Elk verzoek,
ook buiten ``/api`` en ook naar een onbekend pad, moet een geldige sleutel
dragen in ``Authorization: Bearer <sleutel>``; anders HTTP 401. Uitgezonderd
zijn alleen de paden in :data:`~app_main.api.agent_auth.ZONDER_SLEUTEL`.
De middleware zet de sleutel op ``request.state.agent_key``; het recht per
route controleert :func:`~app_main.api.agent_auth.require_right`.

Drempel: per IP-adres telt de middleware de mislukte opzoekingen van een
aangeboden sleutel (in het geheugen, venster van 60 s). Boven
:data:`MAX_MISLUKT_PER_MINUUT` krijgt elk verzoek met een sleutel van dat adres
429, vóór de opzoeking in de database; een 429 komt niet in het auditlog. Een
verzoek zonder sleutel kost geen databasewerk en telt niet mee.

CORS-preflight: ``CORSMiddleware`` staat buiten deze middleware en beantwoordt
een preflight (``OPTIONS`` met ``Origin`` en ``Access-Control-Request-Method``)
zelf, zonder inhoud. Dat is de enige andere uitzondering; een gewone
``OPTIONS`` krijgt 401.
"""

import time
from collections import defaultdict
from typing import Dict, List

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from surrealdb_service.connection import DatabaseUnavailableError

from app_main.api import agent_auth


#: Mislukte sleutels per IP-adres per minuut; daarboven 429 zonder databasewerk.
MAX_MISLUKT_PER_MINUUT = 20
_VENSTER_SEC = 60.0
_OPSCHONEN_BOVEN = 10_000
_mislukt: Dict[str, List[float]] = defaultdict(list)


def _ip(request: Request) -> str:
    return request.client.host if request.client else "onbekend"


def _recent(ip: str, nu: float) -> List[float]:
    return [t for t in _mislukt.get(ip, []) if nu - t < _VENSTER_SEC]


def _boven_drempel(ip: str) -> bool:
    return len(_recent(ip, time.monotonic())) >= MAX_MISLUKT_PER_MINUUT


def _tel_mislukt(ip: str) -> None:
    nu = time.monotonic()
    _mislukt[ip] = _recent(ip, nu) + [nu]
    if len(_mislukt) > _OPSCHONEN_BOVEN:
        for k in list(_mislukt):
            if not _recent(k, nu):
                _mislukt.pop(k, None)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Authenticeert elk verzoek met een sleutel, standaard weigeren."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in agent_auth.ZONDER_SLEUTEL:
            return await call_next(request)
        ip = _ip(request)
        met_sleutel = agent_auth.bearer_token(request) is not None
        if met_sleutel and _boven_drempel(ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Te veel mislukte sleutels"},
                headers={"Retry-After": "60"},
            )
        try:
            request.state.agent_key = await agent_auth.authenticate_request(request)
        except HTTPException as exc:
            if met_sleutel and exc.status_code == 401:
                _tel_mislukt(ip)
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers,
            )
        except DatabaseUnavailableError:
            return JSONResponse(
                status_code=503,
                content={"detail": "Database unavailable"},
                headers={"Retry-After": "5"},
            )
        return await call_next(request)


__all__ = ["MAX_MISLUKT_PER_MINUUT", "ApiKeyMiddleware"]
