"""Agent audit log (Track G.1).

Append-only per-call trail for the agent API. The writer is best-effort — an
audit-write failure must NEVER fail the agent request it records (the request
already succeeded/failed on its own merits). Reads power the per-agent audit-log
surface (management UI, G.4).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query


class AgentAuditService:
    """Record + read agent API-call audit entries (G.1)."""

    def __init__(self, config: Optional[SurrealDBConfig] = None) -> None:
        self.config = config

    async def record(
        self,
        *,
        agent_id: str,
        method: str,
        path: str,
        status: int,
        agent_key_id: Optional[str] = None,
        latency_ms: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> None:
        """Append one audit row; fail-soft (never raises)."""
        try:
            from surrealdb_service.connection import ensure_record_id

            data: Dict[str, Any] = {
                "agent_id": agent_id,
                "method": method,
                "path": path,
                "status": status,
                "latency_ms": latency_ms,
                "job_id": job_id,
            }
            if agent_key_id:
                data["agent_key"] = ensure_record_id(agent_key_id)
            await execute_query(
                "CREATE agent_audit_log CONTENT $data", {"data": data}, self.config
            )
        except Exception as exc:  # noqa: BLE001 — audit must never fail a request
            logger.warning(
                "agent audit write failed ({m} {p} -> {s}): {e}",
                m=method,
                p=path,
                s=status,
                e=exc,
            )

    async def agent_owns_job(self, agent_id: str, job_id: str) -> bool:
        """True iff THIS agent enqueued ``job_id`` (per its own audit trail).

        Jobs carry no owner column, so ownership is derived from the audit row
        ``process-url`` stamps with the enqueued job id. Used to authorize
        ``GET /jobs/{id}`` so an agent can only poll a job it created (else 404),
        closing the IDOR / existence-oracle on the shared job table. Fail-soft:
        any read error → False (deny), never raises into the request.
        """
        if not agent_id or not job_id:
            return False
        try:
            rows = await execute_query(
                "SELECT VALUE id FROM agent_audit_log "
                "WHERE agent_id = $a AND job_id = $j LIMIT 1",
                {"a": agent_id, "j": job_id},
                self.config,
            )
            return bool(rows)
        except Exception as exc:  # noqa: BLE001 — deny on error, never leak
            logger.warning("agent_owns_job check failed ({j}): {e}", j=job_id, e=exc)
            return False

    async def list_for_agent(
        self, agent_id: str, *, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """The most recent audit rows for one agent id (newest first)."""
        rows = await execute_query(
            "SELECT agent_id, method, path, status, latency_ms, job_id, ts "
            "FROM agent_audit_log WHERE agent_id = $a ORDER BY ts DESC LIMIT $n",
            {"a": agent_id, "n": limit},
            self.config,
        )
        return [
            {
                "agent_id": r.get("agent_id"),
                "method": r.get("method"),
                "path": r.get("path"),
                "status": r.get("status"),
                "latency_ms": r.get("latency_ms"),
                "job_id": r.get("job_id"),
                "ts": str(r.get("ts")) if r.get("ts") else None,
            }
            for r in (rows or [])
        ]


__all__ = ["AgentAuditService"]
