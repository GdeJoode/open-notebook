"""Agent-integration models (Track G.1).

The JSON contracts for the API-key-authed agent surface: key management (mint /
list), the show-once plaintext response, the audit-log entry, and the first
capability's request/response (`extract-entities` over raw text). Kept in
``shared`` so the services, repositories, and routers share one shape.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

#: Permission scopes, least→most privileged. A route requiring scope X accepts a
#: key whose scope is >= X (read < write < admin).
AGENT_PERMISSIONS = ("read", "write", "admin")


class AgentKeyCreate(BaseModel):
    """Request to mint a new agent key (session-auth key management)."""

    agent_id: str = Field(description="Stable id of the calling agent.")
    permission: str = Field(
        default="read", description='One of "read" | "write" | "admin".'
    )
    label: Optional[str] = Field(default=None, description="Human-readable note.")
    rate_limit_rpm: Optional[int] = Field(
        default=None,
        description="Per-key requests-per-minute cap; None → the config default.",
    )


class AgentKeyPublic(BaseModel):
    """A key as listed back — NEVER carries the plaintext or the hash."""

    id: str
    agent_id: str
    key_prefix: str
    label: Optional[str] = None
    permission: str
    rate_limit_rpm: Optional[int] = None
    revoked: bool = False
    created: Optional[str] = None
    last_used_at: Optional[str] = None


class AgentKeyCreated(AgentKeyPublic):
    """Mint response — carries the plaintext key EXACTLY ONCE."""

    key: str = Field(description="The plaintext key. Shown once; store it now.")


class AgentAuditEntry(BaseModel):
    """One append-only agent-audit-log row."""

    agent_id: str
    method: str
    path: str
    status: int
    latency_ms: Optional[int] = None
    job_id: Optional[str] = None
    ts: Optional[str] = None


class ExtractEntitiesRequest(BaseModel):
    """`POST /api/v1/agents/extract-entities` — typed entities from raw text."""

    text: str = Field(description="Raw text to extract entities from.")
    ontology_name: Optional[str] = Field(
        default=None,
        description="Ontology/schema to guide extraction; None → the default.",
    )


class ExtractedEntity(BaseModel):
    name: str
    entity_type: str
    confidence: float = 1.0


class ExtractEntitiesResponse(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)


__all__ = [
    "AGENT_PERMISSIONS",
    "AgentAuditEntry",
    "AgentKeyCreate",
    "AgentKeyCreated",
    "AgentKeyPublic",
    "ExtractEntitiesRequest",
    "ExtractEntitiesResponse",
    "ExtractedEntity",
]
