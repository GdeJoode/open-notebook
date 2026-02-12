"""
Shared models for entity/relation extraction and filtering.

Used by ontology-extraction and entity-filtering pipelines for
cross-pipeline data exchange.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    """An entity extracted from text."""

    text: str = Field(description="The entity surface form as found in text")
    label: str = Field(description="Entity type label (e.g. PERSON, ORG)")
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Additional properties"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_chunk_id: Optional[str] = Field(
        default=None, description="ID of the chunk this entity was extracted from"
    )


class ExtractedRelation(BaseModel):
    """A relation between two entities."""

    source_entity: str = Field(description="Source entity text")
    target_entity: str = Field(description="Target entity text")
    relation_type: str = Field(description="Relation type label")
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_chunk_id: Optional[str] = Field(
        default=None, description="ID of the chunk this relation was extracted from"
    )


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
    predicted_edges: List[ExtractedRelation] = Field(
        default_factory=list,
        description="New relations predicted by edge scoring",
    )
