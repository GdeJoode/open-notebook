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
]
