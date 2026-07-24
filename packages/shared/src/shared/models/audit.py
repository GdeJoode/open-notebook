"""Audit findings models (Track F.1 — operations & quality).

The JSON contract for the per-notebook audit: a run produces a list of
:class:`AuditFinding`, grouped under one ``run_id`` and returned by the
``/api/notebooks/{id}/audit`` endpoints. Kept in ``shared`` so the service, the
repository, and the API router share one shape (mirrors the orphans-dashboard
response models).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

#: The six LLM-free check identifiers (F.1). Stable strings — the UI keys off them.
AUDIT_CHECK_IDS = (
    "citation_completeness",
    "stale_sources",
    "low_confidence_survivors",
    "long_pending_orphans",
    "community_cohesion",
    "schema_drift",
)

#: Valid severities, most→least urgent.
AUDIT_SEVERITIES = ("error", "warn", "info")


class AuditFinding(BaseModel):
    """One quality-audit finding for a notebook."""

    check_id: str = Field(description="Which check produced this (see AUDIT_CHECK_IDS).")
    severity: str = Field(description='One of "error" | "warn" | "info".')
    title: str = Field(description="Human-readable one-line summary.")
    detail: Optional[str] = Field(
        default=None,
        description="Optional JSON payload (offending ids / counts / context).",
    )
    subject: Optional[str] = Field(
        default=None,
        description="Optional offending record id (entity / source / community).",
    )
    count: Optional[int] = Field(
        default=None, description="Optional magnitude (e.g. how many entities)."
    )


class AuditRunResponse(BaseModel):
    """The result of one audit run — a snapshot of findings under a run id."""

    notebook_id: str
    run_id: str
    findings: List[AuditFinding] = Field(default_factory=list)
    created: Optional[str] = Field(
        default=None, description="ISO timestamp of the run (from the first row)."
    )

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "error")


__all__ = [
    "AUDIT_CHECK_IDS",
    "AUDIT_SEVERITIES",
    "AuditFinding",
    "AuditRunResponse",
]
