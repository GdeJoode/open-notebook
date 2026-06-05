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

from pydantic import Field, field_validator

from shared.models.base import ObjectModel


class Entity(ObjectModel):
    """Canonical entity record in the knowledge graph.

    Field set mirrors ``entity`` table in migration 39 (canonical_name,
    entity_type, description, provenance bag, graph algorithm scores,
    status) plus migration 44 (``type_tags``, ``primary_type``) for
    multi-type tagging from Phase B1's merge step.

    Why ``embedding`` is required (no default): the schema declares
    ``embedding FLEXIBLE TYPE array`` with NO ``DEFAULT`` clause, so every
    ``CREATE entity`` must supply an explicit value — even an empty list.
    Callers that don't have a vector yet must pass ``embedding=[]``
    explicitly. This mirrors the production canary in
    ``test_entity_roundtrip``.
    """

    table_name: ClassVar[str] = "entity"

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

    # Embedding (FLEXIBLE TYPE array, NO default in schema → caller MUST supply)
    embedding: List[float] = Field(
        default_factory=list,
        description="Vector embedding for semantic search (empty list when not yet computed)",
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

    # Multi-type tagging (migration 44 — Phase B.1a)
    type_tags: List[str] = Field(
        default_factory=list,
        description="All applicable entity types after the B1 merge step (e.g. ['Person', 'Researcher'])",
    )
    primary_type: Optional[str] = Field(
        default=None,
        description="The 'best' / highest-confidence type when multiple type_tags apply",
    )

    @field_validator("type_tags", "source_documents", "provenance_chain", mode="before")
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


class Relation(ObjectModel):
    """Typed directed edge between two entities (RELATE table in migration 39).

    Note: SurrealDB RELATE tables carry ``in`` / ``out`` system fields holding
    the source/target record IDs. We surface those here as ``in_entity`` and
    ``out_entity`` to avoid shadowing the Python ``in`` keyword. The
    repository write-path translates these to the DB-side ``in`` / ``out``.
    """

    table_name: ClassVar[str] = "relation"

    in_entity: Optional[str] = Field(
        default=None, description="Source entity record ID (DB-side 'in')"
    )
    out_entity: Optional[str] = Field(
        default=None, description="Target entity record ID (DB-side 'out')"
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
