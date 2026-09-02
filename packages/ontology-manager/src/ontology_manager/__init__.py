"""
Ontology Manager package.

Manages ontology schemas, versioning, validation, and evolution for
the open-notebook knowledge extraction system.
"""

from ontology_manager.canonical_bridge import (
    CanonicalResolution,
    resolve_ontology_type,
)
from ontology_manager.config import OntologyManagerConfig, get_config, set_config
from ontology_manager.document_mapper import (
    get_document_types_for_ontology,
    get_ontology_for_document_type,
    get_supported_document_types,
)
from ontology_manager.evolution import (
    GapStatus,
    OntologyEvolutionAgent,
    OntologyGap,
    ProposalStatus,
    ProposalType,
    SchemaProposal,
)
from ontology_manager.manager import OntologyManager, get_ontology_manager
from ontology_manager.prompts import OntologyPromptGenerator
from ontology_manager.registry import OntologyRegistry, get_ontology_registry
from ontology_manager.schema import (
    Cardinality,
    ClaimTypeDefinition,
    ConceptDefinition,
    DataType,
    EntityTypeDefinition,
    ExtractionPatternDefinition,
    ExtractionStrategy,
    Ontology,
    OntologyMetadata,
    PropertyDefinition,
    RelationshipTypeDefinition,
    ValidationRules,
)
from ontology_manager.schema_projection import (
    Projection,
    ProjectionOutcome,
    project_accepted_edits,
)
from ontology_manager.type_placement import (
    TypePlacement,
    place_proposed_type,
    sibling_types,
)
from ontology_manager.type_placement_judge import (
    JudgeSelection,
    build_judge_prompt,
    candidates_from_ontologies,
    parse_judge_response,
)
from ontology_manager.validator import (
    OntologyValidator,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)

__all__ = [
    # Config
    "OntologyManagerConfig",
    "get_config",
    "set_config",
    # Manager
    "OntologyManager",
    "get_ontology_manager",
    # Schema models
    "DataType",
    "ExtractionStrategy",
    "Cardinality",
    "ValidationRules",
    "PropertyDefinition",
    "EntityTypeDefinition",
    "RelationshipTypeDefinition",
    "ConceptDefinition",
    "ClaimTypeDefinition",
    "ExtractionPatternDefinition",
    "OntologyMetadata",
    "Ontology",
    # Registry
    "OntologyRegistry",
    "get_ontology_registry",
    # Canonical bridge (L.1)
    "CanonicalResolution",
    "resolve_ontology_type",
    # Prompts
    "OntologyPromptGenerator",
    # Schema projection (N.4d.3)
    "Projection",
    "ProjectionOutcome",
    "project_accepted_edits",
    # Validator
    "OntologyValidator",
    "ValidationIssue",
    "ValidationReport",
    "ValidationSeverity",
    # Evolution
    "OntologyEvolutionAgent",
    "OntologyGap",
    "GapStatus",
    "SchemaProposal",
    "ProposalStatus",
    "ProposalType",
    # Document mapper
    "get_ontology_for_document_type",
    "get_supported_document_types",
    "get_document_types_for_ontology",
]

__all__ += [
    "TypePlacement",
    "place_proposed_type",
    "sibling_types",
]

__all__ += [
    "JudgeSelection",
    "build_judge_prompt",
    "candidates_from_ontologies",
    "parse_judge_response",
]
