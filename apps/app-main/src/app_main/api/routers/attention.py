"""The Werkbak (WB.1): ``GET /api/attention``, what needs the owner.

A thin shell over :class:`AttentionService`; see that module for the groups,
where each is read from, and the route each links to.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app_main.dependencies import get_attention_service
from app_main.services.attention_service import AttentionReport, AttentionService

router = APIRouter(tags=["attention"])


@router.get("/attention", response_model=AttentionReport)
async def get_attention(
    since: Optional[datetime] = Query(
        None,
        description=(
            "The client's last visit (the generated_at of the report it marked "
            "as seen); adds what arrived after it"
        ),
    ),
    svc: AttentionService = Depends(get_attention_service),
) -> AttentionReport:
    """Every group that can ask for the owner, with counts, top items and links.

    A group that fails or times out comes back with ``error`` set; the rest
    still answer.
    """
    return await svc.report(since=since)
