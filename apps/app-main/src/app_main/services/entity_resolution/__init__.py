"""Entity-resolution services (Track K).

Layer-2 resolution over the already-persisted knowledge graph: retroactive
canonicalization/merge (K.3), vocabulary reconciliation (K.4), and
fuzzy/embedding candidate dedup + user overlay (K.5).
"""

from app_main.services.entity_resolution.candidate_dedup_service import (
    CandidateDedupService,
    CandidateReport,
    MergeCandidate,
)
from app_main.services.entity_resolution.overlay_service import (
    OverlayRule,
    OverlayService,
)
from app_main.services.entity_resolution.recanonicalization_service import (
    MergeCluster,
    MergePlan,
    MergeResult,
    RecanonicalizationService,
)
from app_main.services.entity_resolution.vocabulary_reconciler import (
    ReconcileResult,
    VocabularyReconciler,
)

__all__ = [
    "CandidateDedupService",
    "CandidateReport",
    "MergeCandidate",
    "MergeCluster",
    "MergePlan",
    "MergeResult",
    "OverlayRule",
    "OverlayService",
    "RecanonicalizationService",
    "ReconcileResult",
    "VocabularyReconciler",
]
