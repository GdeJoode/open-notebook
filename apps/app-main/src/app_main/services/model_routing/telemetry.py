"""Routing telemetry write (Track J.4, J-Q7).

After a routed LLM stage completes, :func:`record_routing_event` writes one
``routing.served`` metric row carrying the served provider/model plus the
failover trail (``was_failover`` + ``fallback_from``). It reuses the always-on,
never-blocking :func:`shared.services.metrics.record_metric` sink so a telemetry
hiccup never fails the LLM stage.

This is the single served-provider write per document stage that J.5's UI reads
to show the cloud-vs-local + fallback summary, and the provenance source the
extraction path stamps onto persisted entities.
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger

from app_main.services.model_routing.route_resolver import LLMTask


async def record_routing_event(
    *,
    task: LLMTask,
    served_provider: str,
    served_model_id: Optional[str],
    was_failover: bool,
    fallback_from: List[str],
    source_id: Optional[str] = None,
    notebook_id: Optional[str] = None,
) -> None:
    """Record one ``routing.served`` telemetry event for a routed LLM stage.

    Best-effort: never raises (``record_metric`` swallows its own failures, and
    we guard the call too) so the LLM stage is never failed by telemetry.
    """
    try:
        from shared.services.metrics import record_metric

        await record_metric(
            "routing.served",
            {
                "task": task.value,
                "served_provider": served_provider,
                "served_model_id": served_model_id,
                "was_failover": was_failover,
                "fallback_from": fallback_from,
            },
            source=source_id,
            notebook=notebook_id,
        )
    except Exception as e:  # noqa: BLE001 — telemetry never fails the caller
        logger.warning(f"routing telemetry write failed (ignored): {e}")
