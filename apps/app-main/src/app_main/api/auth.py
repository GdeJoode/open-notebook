"""Sleutelmiddleware: geen verzoek zonder geldige sleutel (F2-01, ADR 0002 A).

Vervangt de wachtwoordmiddleware en ``OPEN_NOTEBOOK_PASSWORD``. Elk verzoek,
ook buiten ``/api`` en ook naar een onbekend pad, moet een geldige sleutel
dragen in ``Authorization: Bearer <sleutel>``; anders HTTP 401. Uitgezonderd
zijn alleen de paden in :data:`~app_main.api.agent_auth.ZONDER_SLEUTEL`.
De middleware zet de sleutel op ``request.state.agent_key``; het recht per
route controleert :func:`~app_main.api.agent_auth.require_right`.
"""

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from surrealdb_service.connection import DatabaseUnavailableError

from app_main.api import agent_auth


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Authenticeert elk verzoek met een sleutel, standaard weigeren."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in agent_auth.ZONDER_SLEUTEL:
            return await call_next(request)
        try:
            request.state.agent_key = await agent_auth.authenticate_request(request)
        except HTTPException as exc:
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


__all__ = ["ApiKeyMiddleware"]
