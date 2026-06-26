"""Retrieval primitives shared across services (Track R).

The KG-proximity scorer (R.2) lives here as a **pure** function so it can be
unit-tested in isolation and reused by both the service layer and any offline
ablation harness, without taking a DB dependency.
"""

from shared.retrieval.kg_source_scorer import (
    EntityRecord,
    RelationRecord,
    SharedEntityContribution,
    SourceKGScore,
    TYPE_SALIENCE,
    entity_weight,
    score_related_sources,
)

__all__ = [
    "EntityRecord",
    "RelationRecord",
    "SharedEntityContribution",
    "SourceKGScore",
    "TYPE_SALIENCE",
    "entity_weight",
    "score_related_sources",
]
