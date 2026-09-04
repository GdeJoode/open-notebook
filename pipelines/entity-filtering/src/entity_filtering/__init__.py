"""
Entity filtering pipeline.

Generic entity/relation filtering, normalization, deduplication,
semantic resolution, KG matching, and scoring. Takes ExtractionResult
from ontology-extraction and produces FilteredResult.
"""

from entity_filtering.config import (
    EdgePredictionConfig,
    EmbeddingDedupConfig,
    FilteringConfig,
    FuzzyDedupConfig,
    KGResolutionConfig,
    OntologyValidationConfig,
    SemanticConfig,
    SyntacticConfig,
)
from entity_filtering.filters.base import FilterBase
from entity_filtering.resolution.contextual_clusterer import ContextualClusterer
from entity_filtering.resolution.embedding_resolver import EmbeddingResolver
from entity_filtering.resolution.entity_linker import (
    DBpediaSpotlightLinker,
    EntityLinker,
)
from entity_filtering.resolution.kg_resolver import (
    EntityRepositoryProtocol,
    KGResolver,
)
from entity_filtering.validation.graph_analyzer import (
    GraphAnalyzer,
    MergedGraphAnalyzer,
)
from entity_filtering.validation.ontology_constraint_filter import (
    OntologyConstraintFilter,
)
from entity_filtering.workflow import FilteringWorkflow

__all__ = [
    # Config
    "FilteringConfig",
    "SyntacticConfig",
    "FuzzyDedupConfig",
    "EmbeddingDedupConfig",
    "SemanticConfig",
    "KGResolutionConfig",
    "OntologyValidationConfig",
    "EdgePredictionConfig",
    # Workflow
    "FilteringWorkflow",
    "FilterBase",
    # Resolution
    "EmbeddingResolver",
    "EntityLinker",
    "DBpediaSpotlightLinker",
    "ContextualClusterer",
    "KGResolver",
    "EntityRepositoryProtocol",
    # Validation
    "GraphAnalyzer",
    "MergedGraphAnalyzer",
    "OntologyConstraintFilter",
]
