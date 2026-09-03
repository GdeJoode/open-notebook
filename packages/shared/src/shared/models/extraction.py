"""
Shared models for entity/relation extraction and filtering.

Used by ontology-extraction and entity-filtering pipelines for
cross-pipeline data exchange.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExtractionContext(BaseModel):
    """Document structure context for an extracted entity or relation.

    Preserved from Docling output through the chunking pipeline so that
    downstream matching can use section headings, page numbers, and
    surrounding text for disambiguation.
    """

    section_heading: Optional[str] = Field(
        default=None,
        description="Innermost heading under which the mention appears",
    )
    section_path: List[str] = Field(
        default_factory=list,
        description="Full heading breadcrumb, e.g. ['Chapter 1', 'Section 1.2']",
    )
    section_level: int = Field(
        default=0, description="Nesting depth in the document structure"
    )
    page_number: Optional[int] = Field(
        default=None, description="Physical page number (0-indexed)"
    )
    element_type: Optional[str] = Field(
        default=None, description="Source element type (paragraph, title, table, …)"
    )
    surrounding_text: Optional[str] = Field(
        default=None,
        description="~200 char window around the mention for disambiguation",
    )
    source_document: Optional[str] = Field(
        default=None, description="Source record ID the chunk belongs to"
    )


class ExtractedEntity(BaseModel):
    """An entity extracted from text.

    Multi-schema tagging (Phase B.1e)
    ================================
    When the multi-schema orchestrator merges results from several
    Pass-2 runs (one per applicable schema), the same entity may
    legitimately receive different ``label`` values from each pass.
    Rather than discarding that information:

    - ``label`` retains the highest-confidence pass's label for
      back-compat with single-schema callers.
    - ``type_tags`` accumulates *every* label this entity was assigned
      across passes — order matches the orchestrator's schema-iteration
      order (highest applicability first).
    - ``primary_type`` mirrors ``label`` and is set explicitly by the
      merger; downstream code that wants the canonical type for a
      multi-tagged entity should prefer ``primary_type`` over ``label``
      because ``label`` was historically free-form.

    Single-schema (non-merged) entities keep ``type_tags`` empty and
    ``primary_type=None`` for back-compat — the absence of these fields
    is the signal that no merge happened.
    """

    text: str = Field(description="The entity surface form as found in text")
    label: str = Field(description="Entity type label (e.g. PERSON, ORG)")
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Additional properties"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_chunk_id: Optional[str] = Field(
        default=None, description="ID of the chunk this entity was extracted from"
    )
    source_grounding: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Source grounding info (char offsets, alignment status) from LangExtract",
    )
    extraction_context: Optional[ExtractionContext] = Field(
        default=None,
        description="Document structure context from the source chunk",
    )
    type_tags: List[str] = Field(
        default_factory=list,
        description=(
            "All entity-type labels assigned to this surface form across "
            "multi-schema passes. Empty list for single-schema results."
        ),
    )
    primary_type: Optional[str] = Field(
        default=None,
        description=(
            "Canonical entity-type label chosen by the multi-schema "
            "merger — the label from the pass with highest confidence. "
            "None for single-schema results (use ``label`` instead)."
        ),
    )


class ExtractedRelation(BaseModel):
    """A relation between two entities.

    Endpoint types (K.7a)
    =====================
    ``source_type`` / ``target_type`` carry the *entity type* of each endpoint
    when it is known at extraction or persist time. They default to ``None`` for
    back-compat — every existing caller that builds an ``ExtractedRelation``
    without them is unaffected, and a ``None`` endpoint type falls back to
    name-only resolution downstream.

    They exist to disambiguate a relation endpoint when two entities share a
    canonical name across types (e.g. a ``person`` and an ``organization`` both
    named "BZK"). With the endpoint type known, the persist path can resolve the
    edge to the *type-correct* entity instead of arbitrarily picking one. See
    ``EntityPersistenceService.persist_filtered_result``.
    """

    source_entity: str = Field(description="Source entity text")
    target_entity: str = Field(description="Target entity text")
    relation_type: str = Field(description="Relation type label")
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_chunk_id: Optional[str] = Field(
        default=None, description="ID of the chunk this relation was extracted from"
    )
    source_type: Optional[str] = Field(
        default=None,
        description=(
            "Entity type of the source endpoint, when known. None → resolve "
            "the endpoint by name only (back-compat fallback)."
        ),
    )
    target_type: Optional[str] = Field(
        default=None,
        description=(
            "Entity type of the target endpoint, when known. None → resolve "
            "the endpoint by name only (back-compat fallback)."
        ),
    )
    extraction_context: Optional[ExtractionContext] = Field(
        default=None,
        description="Document structure context from the source chunk",
    )


class MatchCandidate(BaseModel):
    """A match decision between two entities, stored before merging.

    Separates the matching phase from the merging phase (stap 2C) and
    carries provenance with a reasoning trace (stap 2B).
    """

    entity_a_text: str
    entity_b_text: str
    entity_a_label: str = "UNKNOWN"
    entity_b_label: str = "UNKNOWN"
    match: bool = False
    confidence: float = 0.0
    match_method: str = Field(description="e.g. embedding_similarity, llm_match, fuzzy, abbreviation")
    match_reasoning: str = ""
    iterations: int = Field(default=1, description="Number of matching iterations (1=single-pass)")
    # Section context from extraction
    source_section_a: Optional[str] = None
    source_section_b: Optional[str] = None
    source_document_a: Optional[str] = None
    source_document_b: Optional[str] = None
    matched_by_model: Optional[str] = None
    # Review status: pending, auto_accepted, accepted, rejected
    status: str = "pending"


class ExtractionResult(BaseModel):
    """Result from an extraction pipeline run."""

    entities: List[ExtractedEntity] = Field(default_factory=list)
    relations: List[ExtractedRelation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relation_count(self) -> int:
        return len(self.relations)


class FilteredResult(ExtractionResult):
    """Result from the entity-filtering pipeline."""

    removed_entities: List[ExtractedEntity] = Field(
        default_factory=list, description="Entities removed during filtering"
    )
    merged_entity_groups: List[List[str]] = Field(
        default_factory=list,
        description="Groups of entity texts that were merged during dedup",
    )
    match_candidates: List[MatchCandidate] = Field(
        default_factory=list,
        description="All match decisions with provenance (stap 2C/2B)",
    )
    predicted_edges: List[ExtractedRelation] = Field(
        default_factory=list,
        description="New relations predicted by edge scoring",
    )
    validation_report: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Report from the ontology/graph validation stage. PC.1b: no reader "
            "yet, kept deliberately — stage 11 is inert today because neither "
            "production call site passes an ontology to FilteringWorkflow, and "
            "PC.6 owns making 'the flag is on and did nothing' visible. Listed "
            "in handoff-inventory.md with PC.6 as owner."
        ),
    )
    kg_resolution_report: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "KG entity resolution statistics (matched_count / new_count). "
            "PC.1b: no reader yet, kept deliberately — PC.3 owns turning "
            "cross-document resolution on, and its AC needs a measured figure "
            "for how many rows it collapses. Listed in the track's "
            "handoff-inventory.md with PC.3 as owner; PC.3 either reads it or "
            "deletes it."
        ),
    )
    concept_alignment_report: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Concept-alignment statistics (Track N.4): how the entities KG "
            "resolution marked new were classified — verdict/method/reason "
            "counts, the judged count, and alias review candidates. The stage "
            "emits no relations."
        ),
    )
