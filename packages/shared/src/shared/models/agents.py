"""Agent-integration models (Track G.1).

The JSON contracts for the API-key-authed agent surface: key management (mint /
list), the show-once plaintext response, the audit-log entry, and the first
capability's request/response (`extract-entities` over raw text). Kept in
``shared`` so the services, repositories, and routers share one shape.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class Right(str, Enum):
    """Het recht van een sleutel (ADR 0002 B, migratie 88)."""

    LEZEN_EXTERN = "lezen_extern"
    LEZEN_INTERN = "lezen_intern"
    PIJPLIJN = "pijplijn"
    OPHALEN = "ophalen"
    BEHEERDER = "beheerder"


class ClassificationScope(str, Enum):
    """Welke classificaties een sleutel mag zien (ADR 0002 C, plan §5.3)."""

    EXTERN = "EXTERN"
    ALLES = "ALLES"
    LEEG = "LEEG"


#: Het bereik volgt uit het recht. `ophalen` ziet geen records.
SCOPE_PER_RIGHT = {
    Right.LEZEN_EXTERN: ClassificationScope.EXTERN,
    Right.LEZEN_INTERN: ClassificationScope.ALLES,
    Right.PIJPLIJN: ClassificationScope.ALLES,
    Right.OPHALEN: ClassificationScope.LEEG,
    Right.BEHEERDER: ClassificationScope.ALLES,
}


class AgentKeyCreate(BaseModel):
    """Verzoek om een nieuwe sleutel (alleen met recht `beheerder`)."""

    agent_id: str = Field(description="Stable id of the calling agent.")
    permission: Right = Field(description="Het recht van de sleutel.")
    person: Optional[str] = Field(
        default=None,
        description="Naam van de persoon; verplicht bij `beheerder`.",
    )
    label: Optional[str] = Field(default=None, description="Human-readable note.")
    rate_limit_rpm: Optional[int] = Field(
        default=None,
        description="Per-key requests-per-minute cap; None → the config default.",
    )

    @model_validator(mode="after")
    def _beheerder_heeft_person(self) -> "AgentKeyCreate":
        if self.permission == Right.BEHEERDER and not (self.person or "").strip():
            raise ValueError("een sleutel met recht beheerder vraagt een person")
        return self


class AgentKeyPublic(BaseModel):
    """A key as listed back — NEVER carries the plaintext or the hash."""

    id: str
    agent_id: str
    key_prefix: str
    label: Optional[str] = None
    permission: str
    person: Optional[str] = None
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


class GenerateSummaryRequest(BaseModel):
    """`POST /api/v1/agents/generate-summary` — summarize raw text (write scope)."""

    text: str = Field(description="Raw text to summarize.")
    strategy: str = Field(
        default="naive",
        description='Summarization strategy: "naive" | "raptor" | "treekg".',
    )
    config: Optional[dict] = Field(
        default=None, description="Optional strategy config overrides."
    )


class GenerateSummaryResponse(BaseModel):
    summary: str = Field(default="")
    strategy: str = Field(default="")


class ProcessUrlRequest(BaseModel):
    """`POST /api/v1/agents/process-url` — ingest a URL headlessly (write scope)."""

    url: str = Field(description="URL to ingest.")
    notebook_id: str = Field(description="Notebook to attach the new source to.")
    transformations: Optional[List[str]] = Field(
        default=None, description="Optional transformation/insight names to run."
    )


class ProcessSourceResponse(BaseModel):
    """The deferred-ingest handle: a job id to poll via GET /agents/jobs/{id}."""

    job_id: str
    source_id: str
    status: str = "queued"


class JobStatusResponse(BaseModel):
    """`GET /api/v1/agents/jobs/{job_id}` — the CommandService status shape."""

    job_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


__all__ = [
    "SCOPE_PER_RIGHT",
    "AgentAuditEntry",
    "AgentKeyCreate",
    "AgentKeyCreated",
    "AgentKeyPublic",
    "ClassificationScope",
    "ExtractEntitiesRequest",
    "ExtractEntitiesResponse",
    "ExtractedEntity",
    "GenerateSummaryRequest",
    "GenerateSummaryResponse",
    "JobStatusResponse",
    "ProcessSourceResponse",
    "ProcessUrlRequest",
    "Right",
]
