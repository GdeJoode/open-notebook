"""Health-check router for downstream services.

Currently exposes ``GET /api/health/mineru`` to back the Settings page
health chip. We keep this as a dedicated router rather than extending
``services_proxy`` because the UI wants a named, single-service
endpoint with a frozen response shape; the aggregate
``/services/health`` can grow to include mineru as an additional
follow-up if useful.

Critical contract: this endpoint **never** returns 5xx. All errors —
including transport failures and unexpected exceptions — are caught
and converted to ``{healthy: false, error: ...}`` with status 200 so
the frontend chip can render a deterministic offline state instead of
showing a generic API-error toast.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field

from app_main.services.mineru_http_client import MineruHttpClient

router = APIRouter(prefix="/health", tags=["health"])


class MineruHealthResponse(BaseModel):
    """Wire format for ``GET /api/health/mineru``.

    Mirrors the frontend ``MineruHealthResponse`` type defined in
    ``frontend/src/lib/types/api.ts``.
    """

    healthy: bool = Field(..., description="True when the mineru service responded 200")
    version: Optional[str] = Field(
        default=None, description="Service version when reported"
    )
    error: Optional[str] = Field(
        default=None, description="Failure reason when healthy is false"
    )


@router.get("/mineru", response_model=MineruHealthResponse)
async def mineru_health() -> MineruHealthResponse:
    """Probe the mineru service. Always returns HTTP 200.

    The polling client (React Query) treats 5xx as a retryable error,
    which is the wrong UX for a status indicator. We catch everything
    here so the chip can transition cleanly between green and red.
    """
    try:
        client = MineruHttpClient()
        result = await client.health_check()
        return MineruHealthResponse(
            healthy=result.healthy,
            version=result.version,
            error=result.error,
        )
    except Exception as exc:  # noqa: BLE001 — endpoint must never raise
        logger.warning(f"[health/mineru] Unexpected probe failure: {exc}")
        return MineruHealthResponse(healthy=False, error=str(exc))
