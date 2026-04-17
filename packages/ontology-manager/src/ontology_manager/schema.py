"""
Pydantic models for ontology schema definitions.

Provides type-safe ontology definitions that can be loaded from YAML files
and used to generate extraction prompts and validate extracted knowledge.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DataType(str, Enum):
    """Supported data types for entity properties."""

    STRING = "string"
    TEXT = "text"  # Long text
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    URL = "url"
    EMAIL = "email"
    ENUM = "enum"  # Constrained to allowed_values
    ARRAY = "array"  # List of values
    REFERENCE = "reference"  # Reference to another entity
    OBJECT = "object"  # Complex object/dictionary
    JSON = "json"  # JSON data


class ExtractionStrategy(str, Enum):
    """How to extract property values from text."""

    DIRECT = "direct"  # Extract directly from text
    INFERRED = "inferred"  # Infer from context
    LOOKUP = "lookup"  # Look up in external database
    COMPUTED = "computed"  # Compute from other properties
    NONE = "none"  # Don't extract, set manually
    LLM = "llm"  # Extract using LLM


class Cardinality(str, Enum):
    """Cardinality constraints for relationships."""

    ONE_TO_ONE = "one-to-one"
    ONE_TO_MANY = "one-to-many"
    MANY_TO_ONE = "many-to-one"
    MANY_TO_MANY = "many-to-many"
    # Alternative underscore variants (for YAML compatibility)
    ONE_TO_ONE_ALT = "one_to_one"
    ONE_TO_MANY_ALT = "one_to_many"
    MANY_TO_ONE_ALT = "many_to_one"
    MANY_TO_MANY_ALT = "many_to_many"


class ValidationRules(BaseModel):
    """Validation constraints for a property."""

    model_config = ConfigDict(extra='ignore')

    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None  # Regex pattern
    allowed_values: Optional[List[str]] = None  # For enum types
    unique: bool = False


class PropertyDefinition(BaseModel):
    """Definition of an entity property."""

    model_config = ConfigDict(extra='ignore')

    name: str
    description: Optional[str] = None
    data_type: DataType
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.DIRECT
    validation: Optional[ValidationRules] = None
    schema_org_mapping: Optional[str] = None  # e.g., "schema:name"
    dublin_core_mapping: Optional[str] = None  # e.g., "dc:title"
    examples: Optional[List[str]] = None
    default_value: Optional[Any] = None
    reference_type: Optional[str] = None  # For REFERENCE data_type

    @field_validator("validation", mode="before")
    @classmethod
    def parse_validation(cls, v):
        if isinstance(v, dict):
            return ValidationRules(**v)
        return v


class EntityTypeDefinition(BaseModel):
    """Definition of an entity type in the ontology."""

    model_config = ConfigDict(extra='ignore')

    name: str
    description: Optional[str] = None
    properties: List[PropertyDefinition] = Field(default_factory=list)
    schema_org_type: Optional[str] = None  # e.g., "schema:Person"
    extraction_hints: Optional[List[str]] = None  # Hints for LLM extraction
    examples: Optional[List[str]] = None  # Example entity names
    aliases: Optional[List[str]] = None  # Alternative type names
    parent_type: Optional[str] = None  # For inheritance

    @field_validator("properties", mode="before")
    @classmethod
    def parse_properties(cls, v):
        if isinstance(v, list):
            return [
                PropertyDefinition(**p) if isinstance(p, dict) else p for p in v
            ]
        return v

    def get_required_properties(self) -> List[PropertyDefinition]:
        """Get all required properties for this entity type."""
        return [
            p for p in self.properties if p.validation and p.validation.required
        ]

    def get_extraction_properties(self) -> List[PropertyDefinition]:
        """Get properties that should be extracted from text."""
        return [
            p
            for p in self.properties
            if p.extraction_strategy != ExtractionStrategy.NONE
        ]


class RelationshipTypeDefinition(BaseModel):
    """Definition of a relationship type between entities."""

    model_config = ConfigDict(extra='ignore')

    name: str
    description: Optional[str] = None
    domain: List[str] = Field(default_factory=list)  # Source entity types
    range: List[str] = Field(default_factory=list)  # Target entity types
    cardinality: Cardinality = Cardinality.MANY_TO_MANY
    properties: Optional[List[PropertyDefinition]] = None
    schema_org_mapping: Optional[str] = None
    symmetric: bool = False  # If A->B implies B->A
    transitive: bool = False  # If A->B and B->C implies A->C
    inverse: Optional[str] = None  # Name of inverse relationship
    extraction_hints: Optional[List[str]] = None
    # Support source_types/target_types as aliases for domain/range
    source_types: Optional[List[str]] = None
    target_types: Optional[List[str]] = None

    def model_post_init(self, __context: Any) -> None:
        """Convert source_types/target_types to domain/range."""
        if self.source_types and not self.domain:
            object.__setattr__(self, 'domain', self.source_types)
        if self.target_types and not self.range:
            object.__setattr__(self, 'range', self.target_types)

    @field_validator("properties", mode="before")
    @classmethod
    def parse_properties(cls, v):
        if isinstance(v, list):
            return [
                PropertyDefinition(**p) if isinstance(p, dict) else p for p in v
            ]
        return v


class ConceptDefinition(BaseModel):
    """Definition of a thematic concept in the ontology."""

    model_config = ConfigDict(extra='ignore')

    name: str
    description: Optional[str] = None
    related_concepts: Optional[List[str]] = None  # Related concept names
    indicators: Optional[List[str]] = None  # Indicators/metrics for this concept
    causal_patterns: Optional[List[str]] = None  # Causal relationship patterns
    examples: Optional[List[str]] = None  # Example instances


class ClaimTypeDefinition(BaseModel):
    """Definition of a claim type for fact extraction."""

    model_config = ConfigDict(extra='ignore')

    name: str
    description: Optional[str] = None
    indicators: Optional[List[str]] = None  # Linguistic indicators


class ExtractionPatternDefinition(BaseModel):
    """Definition of an extraction pattern for text matching."""

    model_config = ConfigDict(extra='ignore')

    name: str
    description: Optional[str] = None
    pattern: Optional[str] = None  # Regex pattern
    signals: Optional[List[str]] = None  # Signal phrases to look for


class OntologyMetadata(BaseModel):
    """Metadata about the ontology."""

    model_config = ConfigDict(extra='ignore')

    name: str
    version: str
    description: Optional[str] = None
    authors: Optional[List[str]] = None
    license: Optional[str] = None
    extends: Optional[str] = None  # Parent ontology name
    includes: Optional[List[str]] = None  # Additional ontologies to merge
    created: Optional[str] = None
    updated: Optional[str] = None


class Ontology(BaseModel):
    """Complete ontology definition."""

    model_config = ConfigDict(extra='ignore')

    metadata: OntologyMetadata
    entity_types: Dict[str, EntityTypeDefinition] = Field(default_factory=dict)
    relationship_types: Dict[str, RelationshipTypeDefinition] = Field(
        default_factory=dict
    )
    prefixes: Dict[str, str] = Field(default_factory=dict)  # Namespace prefixes
    # New fields for extended ontology support
    concepts: Dict[str, ConceptDefinition] = Field(default_factory=dict)
    claim_types: Dict[str, ClaimTypeDefinition] = Field(default_factory=dict)
    extraction_patterns: Dict[str, ExtractionPatternDefinition] = Field(default_factory=dict)

    @field_validator("entity_types", mode="before")
    @classmethod
    def parse_entity_types(cls, v):
        if isinstance(v, list):
            # Convert list format to dict format using 'name' as key
            return {
                et["name"]: EntityTypeDefinition(**et) if isinstance(et, dict) else et
                for et in v
                if isinstance(et, dict) and "name" in et
            }
        if isinstance(v, dict):
            return {
                k: EntityTypeDefinition(**et) if isinstance(et, dict) else et
                for k, et in v.items()
            }
        return v

    @field_validator("relationship_types", mode="before")
    @classmethod
    def parse_relationship_types(cls, v):
        if isinstance(v, list):
            # Convert list format to dict format using 'name' as key
            return {
                rt["name"]: RelationshipTypeDefinition(**rt) if isinstance(rt, dict) else rt
                for rt in v
                if isinstance(rt, dict) and "name" in rt
            }
        if isinstance(v, dict):
            return {
                k: RelationshipTypeDefinition(**rt) if isinstance(rt, dict) else rt
                for k, rt in v.items()
            }
        return v

    @field_validator("concepts", mode="before")
    @classmethod
    def parse_concepts(cls, v):
        if isinstance(v, list):
            return {
                c["name"]: ConceptDefinition(**c) if isinstance(c, dict) else c
                for c in v
                if isinstance(c, dict) and "name" in c
            }
        if isinstance(v, dict):
            return {
                k: ConceptDefinition(**c) if isinstance(c, dict) else c
                for k, c in v.items()
            }
        return v

    @field_validator("claim_types", mode="before")
    @classmethod
    def parse_claim_types(cls, v):
        if isinstance(v, list):
            return {
                ct["name"]: ClaimTypeDefinition(**ct) if isinstance(ct, dict) else ct
                for ct in v
                if isinstance(ct, dict) and "name" in ct
            }
        if isinstance(v, dict):
            return {
                k: ClaimTypeDefinition(**ct) if isinstance(ct, dict) else ct
                for k, ct in v.items()
            }
        return v

    @field_validator("extraction_patterns", mode="before")
    @classmethod
    def parse_extraction_patterns(cls, v):
        if isinstance(v, list):
            return {
                ep["name"]: ExtractionPatternDefinition(**ep) if isinstance(ep, dict) else ep
                for ep in v
                if isinstance(ep, dict) and "name" in ep
            }
        if isinstance(v, dict):
            return {
                k: ExtractionPatternDefinition(**ep) if isinstance(ep, dict) else ep
                for k, ep in v.items()
            }
        return v

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Ontology":
        """Load an ontology from a YAML file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_yaml_string(cls, yaml_content: str) -> "Ontology":
        """Load an ontology from a YAML string."""
        data = yaml.safe_load(yaml_content)
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save the ontology to a YAML file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def get_entity_type(self, name: str) -> Optional[EntityTypeDefinition]:
        """Get an entity type by name (case-insensitive)."""
        name_lower = name.lower()
        for key, entity_type in self.entity_types.items():
            if key.lower() == name_lower:
                return entity_type
            # Check aliases
            if entity_type.aliases:
                for alias in entity_type.aliases:
                    if alias.lower() == name_lower:
                        return entity_type
        return None

    def get_relationship_type(self, name: str) -> Optional[RelationshipTypeDefinition]:
        """Get a relationship type by name (case-insensitive)."""
        name_lower = name.lower()
        for key, rel_type in self.relationship_types.items():
            if key.lower() == name_lower:
                return rel_type
        return None

    def get_valid_relationships_for_entity(
        self, entity_type: str
    ) -> List[RelationshipTypeDefinition]:
        """Get all relationships where the entity type can be the source."""
        entity_type_lower = entity_type.lower()
        valid_rels = []
        for rel in self.relationship_types.values():
            domain_lower = [d.lower() for d in rel.domain]
            if entity_type_lower in domain_lower or "entity" in domain_lower:
                valid_rels.append(rel)
        return valid_rels

    def merge_with(self, parent: "Ontology") -> "Ontology":
        """Merge this ontology with a parent ontology (for inheritance)."""
        # Start with parent's types
        merged_entity_types = dict(parent.entity_types)
        merged_relationship_types = dict(parent.relationship_types)
        merged_prefixes = dict(parent.prefixes)
        merged_concepts = dict(parent.concepts)
        merged_claim_types = dict(parent.claim_types)
        merged_extraction_patterns = dict(parent.extraction_patterns)

        # Override/extend with this ontology's types
        merged_entity_types.update(self.entity_types)
        merged_relationship_types.update(self.relationship_types)
        merged_prefixes.update(self.prefixes)
        merged_concepts.update(self.concepts)
        merged_claim_types.update(self.claim_types)
        merged_extraction_patterns.update(self.extraction_patterns)

        return Ontology(
            metadata=self.metadata,
            entity_types=merged_entity_types,
            relationship_types=merged_relationship_types,
            prefixes=merged_prefixes,
            concepts=merged_concepts,
            claim_types=merged_claim_types,
            extraction_patterns=merged_extraction_patterns,
        )

    def get_concept(self, name: str) -> Optional[ConceptDefinition]:
        """Get a concept by name (case-insensitive)."""
        name_lower = name.lower()
        for key, concept in self.concepts.items():
            if key.lower() == name_lower:
                return concept
        return None

    def get_all_concept_names(self) -> List[str]:
        """Get all concept names defined in this ontology."""
        return list(self.concepts.keys())

    def get_all_claim_type_names(self) -> List[str]:
        """Get all claim type names defined in this ontology."""
        return list(self.claim_types.keys())

    def get_all_entity_type_names(self) -> List[str]:
        """Get all entity type names defined in this ontology."""
        return list(self.entity_types.keys())

    def get_all_relationship_type_names(self) -> List[str]:
        """Get all relationship type names defined in this ontology."""
        return list(self.relationship_types.keys())
