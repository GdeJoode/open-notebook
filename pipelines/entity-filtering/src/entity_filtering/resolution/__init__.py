"""Semantic resolution, entity linking, and KG matching."""

from entity_filtering.resolution.contextual_clusterer import ContextualClusterer
from entity_filtering.resolution.embedding_resolver import EmbeddingResolver
from entity_filtering.resolution.entity_linker import (
    DBpediaSpotlightLinker,
    EntityLinker,
)
from entity_filtering.resolution.kg_resolver import KGResolver
from entity_filtering.resolution.orphan_connector import (
    OrphanEntityRepoProtocol,
    OrphanProposal,
    OrphanTokenBudgetExceeded,
    confirm_connections,
    find_orphans,
    propose_connections,
)
from entity_filtering.resolution.orphan_prune import (
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_MAX_ATTEMPTS,
    STATUS_ARCHIVED,
    STATUS_PENDING_RECONNECT,
    OrphanPruneRepoProtocol,
    RetryOutcome,
    archive_stale_orphans,
    mark_pending_reconnect,
    retry_pending_reconnects,
)

__all__ = [
    "ContextualClusterer",
    "EmbeddingResolver",
    "EntityLinker",
    "DBpediaSpotlightLinker",
    "KGResolver",
    "OrphanEntityRepoProtocol",
    "OrphanProposal",
    "OrphanTokenBudgetExceeded",
    "find_orphans",
    "propose_connections",
    "confirm_connections",
    "OrphanPruneRepoProtocol",
    "RetryOutcome",
    "STATUS_ARCHIVED",
    "STATUS_PENDING_RECONNECT",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_AGE_DAYS",
    "mark_pending_reconnect",
    "retry_pending_reconnects",
    "archive_stale_orphans",
]
