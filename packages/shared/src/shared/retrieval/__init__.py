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
    normalize_entities_for_signal,
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
    "normalize_entities_for_signal",
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
