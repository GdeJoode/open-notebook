"""Semantic resolution, entity linking, and KG matching."""

from entity_filtering.resolution.contextual_clusterer import ContextualClusterer
from entity_filtering.resolution.embedding_resolver import EmbeddingResolver
from entity_filtering.resolution.entity_linker import (
    DBpediaSpotlightLinker,
    EntityLinker,
)
from entity_filtering.resolution.kg_resolver import KGResolver

__all__ = [
    "ContextualClusterer",
    "EmbeddingResolver",
    "EntityLinker",
    "DBpediaSpotlightLinker",
    "KGResolver",
]
