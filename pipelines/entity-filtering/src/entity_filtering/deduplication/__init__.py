"""
Entity deduplication.

Provides string-based deduplication. Embedding-based (FAISS)
deduplication will be added as a separate module when migrated
from the monolith.
"""

from entity_filtering.deduplication.entity_deduplicator import (
    EntityDeduplicator,
)

__all__ = ["EntityDeduplicator"]
