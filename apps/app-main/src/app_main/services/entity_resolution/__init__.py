"""Entity-resolution services (Track K).

Layer-2 resolution over the already-persisted knowledge graph: retroactive
canonicalization/merge (K.3), vocabulary reconciliation (K.4), and
fuzzy/embedding candidate dedup (K.5). K.3 lands first.
"""

from app_main.services.entity_resolution.recanonicalization_service import (
    MergeCluster,
    MergePlan,
    MergeResult,
    RecanonicalizationService,
)

__all__ = [
    "MergeCluster",
    "MergePlan",
    "MergeResult",
    "RecanonicalizationService",
]
