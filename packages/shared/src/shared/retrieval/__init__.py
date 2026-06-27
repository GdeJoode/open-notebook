"""Retrieval primitives shared across services (Track R).

The KG-proximity scorer (R.2) and the hybrid fusion ranker (R.3) live here as
**pure** functions so they can be unit-tested in isolation and reused by both
the service layer and any offline ablation harness, without taking a DB
dependency.
"""

from shared.retrieval.hybrid_fusion import (
    BALANCED,
    DEFAULT_PRESET,
    DEFAULT_RRF_K,
    KG_HEAVY,
    PRESETS,
    FusedResult,
    FusionWeights,
    SignalProvenance,
    fuse_rankings,
    get_preset,
)
from shared.retrieval.kg_signal_normalizer import (
    PREDICATE_CANON,
    NormalizationStats,
    canonical_predicate,
    canonicalize_relations,
    concept_id,
    normalize_entities_for_signal,
    remap_relations_to_concepts,
)
from shared.retrieval.kg_source_scorer import (
    TYPE_SALIENCE,
    EntityRecord,
    RelationRecord,
    SharedEntityContribution,
    SourceKGScore,
    entity_weight,
    score_related_sources,
)
from shared.retrieval.mentions_projection import (
    NAMED_ONLY_MIN_WEIGHT,
    MentionEdge,
    MentionsProjectionStats,
    project_mentions_edges,
)

__all__ = [
    # KG signal (R.2)
    "EntityRecord",
    "RelationRecord",
    "SharedEntityContribution",
    "SourceKGScore",
    "TYPE_SALIENCE",
    "entity_weight",
    "score_related_sources",
    # Search-facing noise re-scope (R.6)
    "PREDICATE_CANON",
    "NormalizationStats",
    "canonical_predicate",
    "canonicalize_relations",
    "concept_id",
    "normalize_entities_for_signal",
    "remap_relations_to_concepts",
    # mentions projection (U.2)
    "MentionEdge",
    "MentionsProjectionStats",
    "NAMED_ONLY_MIN_WEIGHT",
    "project_mentions_edges",
    # Hybrid fusion (R.3)
    "BALANCED",
    "DEFAULT_PRESET",
    "DEFAULT_RRF_K",
    "KG_HEAVY",
    "PRESETS",
    "FusedResult",
    "FusionWeights",
    "SignalProvenance",
    "fuse_rankings",
    "get_preset",
]
