"""
Entity and Relation domain models for the knowledge graph.

These mirror the SCHEMAFULL ``entity`` and ``relation`` tables defined in
``migrations/39.surrealql`` plus the additive ``type_tags`` / ``primary_type``
fields landed by ``migrations/44.surrealql`` (Phase B.1a).

Up to Phase B.1a, the ``entity_persistence_service`` wrote raw dicts whose
field names had drifted from the SCHEMAFULL schema (legacy ``name`` /
``weight`` / ``source_ids`` instead of canonical ``canonical_name`` / no
weight / ``source_documents``). Introducing typed models here gives B1's
merge step (Phase B.1e) and the upsert refactor a typed handle and pins
the canonical write-path to the migration-39+44 schema.
"""

from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import ConfigDict, Field, field_validator

from shared.models.base import ObjectModel


class Entity(ObjectModel):
    """Canonical entity record in the knowledge graph.

    Field set mirrors ``entity`` table in migration 39 (canonical_name,
    entity_type, description, provenance bag, graph algorithm scores,
    status) plus migration 44 (``type_tags``, ``primary_type``) for
    multi-type tagging from Phase B1's merge step.

    Timestamps: migration 39 declares ``created_at`` / ``updated_at`` on
    the ``entity`` table (not the bare ``created`` / ``updated`` carried
    by ``ObjectModel`` for other domain models). They are surfaced as
    explicit fields here so DB roundtrips preserve them — B.1e's merge
    step picks canonical winners by recency and B.3 UIs display them.
    The inherited ``created`` / ``updated`` from ``ObjectModel`` stay
    unused for this model.

    ``embedding`` defaults to an empty list for Pydantic ergonomics;
    callers with a vector should always supply it. The SCHEMAFULL column
    has no DB-side default, so the repository always sends an explicit
    value on CREATE.
    """

    table_name: ClassVar[str] = "entity"

    # Schema-side timestamps (migration 39 — entity table uses the _at
    # suffix, not the bare created/updated inherited from ObjectModel).
    created_at: Optional[datetime] = Field(
        default=None,
        description="Row creation timestamp (DB default = time::now())",
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Row last-update timestamp (DB default = time::now(), refreshed on upsert)",
    )

    # Identity
    canonical_name: str = Field(description="Canonical/preferred name of the entity")
    entity_type: str = Field(description="Primary entity type (Person, Org, ...)")
    description: Optional[str] = Field(default=None, description="Free-text description")

    # Provenance (migration 39)
    source_documents: List[str] = Field(
        default_factory=list,
        description="Source record IDs the entity was extracted from",
    )
    extracted_at: Optional[datetime] = Field(
        default=None,
        description="When the entity was first extracted (DB default = time::now())",
    )
    extraction_method: str = Field(
        default="llm", description="How the entity was extracted (llm, ner, ...)"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in the entity extraction"
    )
    provenance_chain: List[Any] = Field(
        default_factory=list,
        description="Ordered trail of (chunk_id, method, score) tuples",
    )

    # Properties bag
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Flexible domain-specific attributes",
    )

    # Embedding (FLEXIBLE TYPE array, NO DB-side default; the model defaults
    # to [] so callers without a vector yet don't need to construct one).
    embedding: List[float] = Field(
        default_factory=list,
        description="Vector embedding for semantic search; defaults to []. Callers with a real vector should supply it.",
    )

    # Graph algorithm scores (Phase 1)
    pagerank: Optional[float] = None
    betweenness: Optional[float] = None
    community_id: Optional[int] = None

    # Status (migration 39)
    status: str = Field(default="active", description="active | merged | archived")
    merged_into: Optional[str] = Field(
        default=None, description="Record ID of canonical merge target if status=merged"
    )

    # First-class denormalized alias list (migration 54 — Phase K.3).
    # On retroactive merge, each loser's surface form is recorded both as an
    # ``entity_alias`` edge (authoritative) AND appended here for cheap
    # read/display + exporter use (K-D1). Defaults to [] so pre-54 rows and
    # fresh entities read back cleanly.
    aliases: List[str] = Field(
        default_factory=list,
        description="Denormalized surface forms collapsed into this entity on merge",
    )

    # Multi-type tagging (migration 44 — Phase B.1a)
    type_tags: List[str] = Field(
        default_factory=list,
        description="All applicable entity types after the B1 merge step (e.g. ['Person', 'Researcher'])",
    )
    primary_type: Optional[str] = Field(
        default=None,
        description="The 'best' / highest-confidence type when multiple type_tags apply",
    )

    @field_validator(
        "type_tags", "source_documents", "provenance_chain", "aliases", mode="before"
    )
    @classmethod
    def ensure_list(cls, v: Any) -> List[Any]:
        """Coerce None/NONE/missing into [] for resilience against legacy rows."""
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return list(v)

    @field_validator("properties", mode="before")
    @classmethod
    def ensure_properties_dict(cls, v: Any) -> Dict[str, Any]:
        """Coerce None into {} (legacy rows may read NONE)."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        return {}

    @field_validator("created_at", "updated_at", "extracted_at", mode="before")
    @classmethod
    def parse_db_datetime(cls, value: Any) -> Optional[datetime]:
        """Parse datetime from SurrealDB string form, preserving ``None``."""
        if value is None:
            return None
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value


class Relation(ObjectModel):
    """Typed directed edge between two entities (RELATE table in migration 39).

    Note: SurrealDB RELATE tables carry ``in`` / ``out`` system fields holding
    the source/target record IDs. We surface those here as ``in_entity`` and
    ``out_entity`` to avoid shadowing the Python ``in`` keyword. Field aliases
    let callers construct from a raw DB row (with ``in`` / ``out`` keys) or
    from a Pythonic kwargs payload (with ``in_entity`` / ``out_entity``) —
    ``populate_by_name=True`` enables the latter form.

    Timestamps: migration 39 declares only ``created_at`` on the ``relation``
    table (no ``updated_at`` — relations are immutable in the current schema).
    The inherited ``created`` / ``updated`` from ``ObjectModel`` stay unused.
    """

    # Allow both `in_entity=` (Python kwarg) and `in=` (DB-row key) on
    # construction; serialization keeps both forms working as expected.
    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    table_name: ClassVar[str] = "relation"

    # Schema-side timestamp (migration 39 — relation has created_at only,
    # no updated_at).
    created_at: Optional[datetime] = Field(
        default=None,
        description="Row creation timestamp (DB default = time::now())",
    )

    in_entity: Optional[str] = Field(
        default=None,
        alias="in",
        description="Source entity record ID (DB-side 'in')",
    )
    out_entity: Optional[str] = Field(
        default=None,
        alias="out",
        description="Target entity record ID (DB-side 'out')",
    )

    relation_type: str = Field(description="Edge label, e.g. WORKS_AT, PART_OF")
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Flexible edge properties"
    )

    # Provenance (mirrors entity)
    source_documents: List[str] = Field(default_factory=list)
    extracted_at: Optional[datetime] = None
    extraction_method: str = Field(default="llm")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance_chain: List[Any] = Field(default_factory=list)

    # Status
    status: str = Field(default="active")

    @field_validator("source_documents", "provenance_chain", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return list(v)

    @field_validator("properties", mode="before")
    @classmethod
    def ensure_properties_dict(cls, v: Any) -> Dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        return {}

    @field_validator("created_at", "extracted_at", mode="before")
    @classmethod
    def parse_db_datetime(cls, value: Any) -> Optional[datetime]:
        """Parse datetime from SurrealDB string form, preserving ``None``."""
        if value is None:
            return None
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value
