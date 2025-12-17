"""
spaCy KnowledgeBase Builder

Builds spaCy KnowledgeBase and EntityRuler patterns from SurrealDB entities.
This enables efficient entity resolution during document processing.

Key components:
1. KnowledgeBase: Entity embeddings for EntityLinker disambiguation
2. EntityRuler patterns: Fast pattern matching for known entities
3. Incremental updates: Add new entities without full rebuild
4. Caching: Avoid rebuilding on every document

See docs/INTEGRATED_IMPLEMENTATION_PLAN.md 1.7.3 for documentation.
"""

import asyncio
import hashlib
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from open_notebook.database.repository import repo_query


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class KnowledgeBaseConfig:
    """Configuration for KnowledgeBase building."""

    # Cache directory for KB and patterns
    cache_dir: str = field(
        default_factory=lambda: os.getenv(
            "KB_CACHE_DIR", "./data/kb_cache"
        )
    )

    # Entity vector dimension (must match embedding model)
    entity_vector_length: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    )

    # Cache TTL in seconds (0 = never expire)
    cache_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("KB_CACHE_TTL", "3600"))
    )

    # Minimum entity mention count to include in KB
    min_mention_count: int = 1

    # Entity types to include in KB (None = all)
    include_entity_types: Optional[List[str]] = None

    # Whether to include aliases in EntityRuler patterns
    include_aliases: bool = True

    # Maximum number of entities to load (for memory management)
    max_entities: int = 100000


# =============================================================================
# KNOWLEDGE BASE BUILDER
# =============================================================================


class SpacyKnowledgeBaseBuilder:
    """
    Builds spaCy KnowledgeBase and EntityRuler patterns from SurrealDB entities.

    This enables the 3-tier entity resolution flow:
    1. EntityRuler: Fast O(1) pattern matching for known entity names
    2. EntityLinker: Context-based disambiguation using KB
    3. Semantic fallback: Embedding KNN for unknown entities

    Usage:
        builder = SpacyKnowledgeBaseBuilder()
        kb = await builder.build_knowledge_base(nlp.vocab)
        patterns = await builder.build_entity_ruler_patterns()
    """

    def __init__(self, config: Optional[KnowledgeBaseConfig] = None):
        self.config = config or KnowledgeBaseConfig()
        self._kb_cache: Optional[Any] = None  # spaCy KnowledgeBase
        self._patterns_cache: Optional[List[Dict]] = None
        self._cache_timestamp: float = 0
        self._entity_count: int = 0

        # Ensure cache directory exists
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, name: str) -> Path:
        """Get path for a cache file."""
        return Path(self.config.cache_dir) / name

    def _is_cache_valid(self) -> bool:
        """Check if the cached data is still valid."""
        if self._cache_timestamp == 0:
            return False

        if self.config.cache_ttl_seconds == 0:
            return True  # Never expire

        elapsed = time.time() - self._cache_timestamp
        return elapsed < self.config.cache_ttl_seconds

    async def _get_entities_from_db(self) -> List[Dict[str, Any]]:
        """
        Fetch entities with embeddings from SurrealDB.

        Returns entities with: id, name, entity_type, aliases, description, embedding
        """
        try:
            # Build type filter if configured
            type_filter = ""
            params: Dict[str, Any] = {"limit": self.config.max_entities}

            if self.config.include_entity_types:
                type_filter = "AND entity_type IN $types"
                params["types"] = self.config.include_entity_types

            query = f"""
                SELECT
                    id,
                    name,
                    entity_type,
                    aliases,
                    description,
                    embedding,
                    hash_id
                FROM entity
                WHERE embedding != NONE
                    {type_filter}
                LIMIT $limit
            """

            results = await repo_query(query, params)
            logger.info(f"Fetched {len(results)} entities from database for KB")
            return results

        except Exception as e:
            logger.error(f"Error fetching entities from database: {e}")
            return []

    async def _get_entity_mention_counts(self) -> Dict[str, int]:
        """Get mention counts for entities to use as frequency in KB."""
        try:
            results = await repo_query("""
                SELECT
                    out AS entity_id,
                    count() AS mention_count
                FROM mentions
                GROUP BY out
            """)
            return {r["entity_id"]: r["mention_count"] for r in results}
        except Exception as e:
            logger.debug(f"Could not get mention counts: {e}")
            return {}

    def _entity_type_to_label(self, entity_type: str) -> str:
        """Map entity type to spaCy NER label."""
        # Standard spaCy labels
        mapping = {
            "person": "PERSON",
            "organization": "ORG",
            "location": "GPE",
            "event": "EVENT",
            "topic": "CONCEPT",
            "concept": "CONCEPT",
            "product": "PRODUCT",
            "scholarly_article": "WORK_OF_ART",
            "creative_work": "WORK_OF_ART",
            "grant": "ORG",
            "research_project": "PROJECT",
            "policy_document": "LAW",
            "legislation": "LAW",
            "government_organization": "ORG",
            "administrative_area": "GPE",
            "periodical": "ORG",
            "dataset": "PRODUCT",
        }
        return mapping.get(entity_type.lower(), entity_type.upper())

    async def build_knowledge_base(
        self,
        vocab,
        force_rebuild: bool = False,
    ):
        """
        Build spaCy KnowledgeBase from SurrealDB entities.

        Args:
            vocab: spaCy Vocab object from the nlp pipeline
            force_rebuild: Force rebuild even if cache is valid

        Returns:
            spaCy KnowledgeBase populated with entities
        """
        try:
            from spacy.kb import KnowledgeBase
        except ImportError:
            logger.error("spaCy not installed. Cannot build KnowledgeBase.")
            return None

        # Check cache
        if not force_rebuild and self._is_cache_valid() and self._kb_cache is not None:
            logger.debug("Using cached KnowledgeBase")
            return self._kb_cache

        # Try to load from disk cache
        kb_path = self._get_cache_path("knowledge_base")
        if not force_rebuild and kb_path.exists():
            try:
                kb = KnowledgeBase(
                    vocab=vocab,
                    entity_vector_length=self.config.entity_vector_length
                )
                kb.from_disk(str(kb_path))
                self._kb_cache = kb
                self._cache_timestamp = time.time()
                logger.info(f"Loaded KnowledgeBase from disk cache ({kb.get_size_entities()} entities)")
                return kb
            except Exception as e:
                logger.warning(f"Failed to load KB from cache: {e}")

        # Build fresh KB
        logger.info("Building KnowledgeBase from database...")

        # Create KB
        kb = KnowledgeBase(
            vocab=vocab,
            entity_vector_length=self.config.entity_vector_length
        )

        # Fetch entities and mention counts
        entities = await self._get_entities_from_db()
        mention_counts = await self._get_entity_mention_counts()

        if not entities:
            logger.warning("No entities found in database")
            return kb

        # Add entities to KB
        added_count = 0
        for entity in entities:
            entity_id = entity.get("id")
            name = entity.get("name")
            embedding = entity.get("embedding")

            if not entity_id or not name or not embedding:
                continue

            # Get frequency from mention count
            freq = mention_counts.get(entity_id, 1)
            if freq < self.config.min_mention_count:
                continue

            # Ensure embedding has correct dimension
            if len(embedding) != self.config.entity_vector_length:
                logger.warning(
                    f"Entity {name} has embedding dimension {len(embedding)}, "
                    f"expected {self.config.entity_vector_length}"
                )
                continue

            try:
                # Add entity to KB
                kb.add_entity(
                    entity=entity_id,
                    freq=freq,
                    entity_vector=embedding
                )

                # Add primary name as alias
                kb.add_alias(
                    alias=name,
                    entities=[entity_id],
                    probabilities=[1.0]
                )

                # Add aliases
                if self.config.include_aliases:
                    aliases = entity.get("aliases", []) or []
                    for alias in aliases:
                        if alias and alias.strip():
                            try:
                                kb.add_alias(
                                    alias=alias.strip(),
                                    entities=[entity_id],
                                    probabilities=[1.0]
                                )
                            except Exception:
                                # Alias may already exist for another entity
                                pass

                added_count += 1

            except Exception as e:
                logger.debug(f"Failed to add entity {name}: {e}")

        # Save to disk cache
        try:
            kb.to_disk(str(kb_path))
            logger.debug(f"Saved KnowledgeBase to {kb_path}")
        except Exception as e:
            logger.warning(f"Failed to save KB to cache: {e}")

        self._kb_cache = kb
        self._cache_timestamp = time.time()
        self._entity_count = added_count

        logger.info(f"Built KnowledgeBase with {added_count} entities")
        return kb

    async def build_entity_ruler_patterns(
        self,
        force_rebuild: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Build EntityRuler patterns from SurrealDB entities.

        Returns patterns in the format:
        [
            {"label": "ORG", "pattern": "Microsoft", "id": "entity:xxx"},
            {"label": "ORG", "pattern": "MSFT", "id": "entity:xxx"},
            ...
        ]

        These patterns enable O(1) hash-based lookups in spaCy's EntityRuler.
        """
        # Check cache
        if not force_rebuild and self._is_cache_valid() and self._patterns_cache is not None:
            logger.debug("Using cached EntityRuler patterns")
            return self._patterns_cache

        # Try to load from disk cache
        patterns_path = self._get_cache_path("entity_ruler_patterns.pkl")
        if not force_rebuild and patterns_path.exists():
            try:
                with open(patterns_path, "rb") as f:
                    patterns = pickle.load(f)
                self._patterns_cache = patterns
                self._cache_timestamp = time.time()
                logger.info(f"Loaded {len(patterns)} EntityRuler patterns from cache")
                return patterns
            except Exception as e:
                logger.warning(f"Failed to load patterns from cache: {e}")

        # Build fresh patterns
        logger.info("Building EntityRuler patterns from database...")

        entities = await self._get_entities_from_db()
        patterns = []
        seen_patterns = set()  # Avoid duplicate patterns

        for entity in entities:
            entity_id = entity.get("id")
            name = entity.get("name")
            entity_type = entity.get("entity_type", "other")

            if not entity_id or not name:
                continue

            label = self._entity_type_to_label(entity_type)

            # Add pattern for primary name
            pattern_key = (label, name.lower())
            if pattern_key not in seen_patterns:
                patterns.append({
                    "label": label,
                    "pattern": name,
                    "id": entity_id
                })
                seen_patterns.add(pattern_key)

            # Add patterns for aliases
            if self.config.include_aliases:
                aliases = entity.get("aliases", []) or []
                for alias in aliases:
                    if alias and alias.strip():
                        alias = alias.strip()
                        pattern_key = (label, alias.lower())
                        if pattern_key not in seen_patterns:
                            patterns.append({
                                "label": label,
                                "pattern": alias,
                                "id": entity_id
                            })
                            seen_patterns.add(pattern_key)

        # Save to disk cache
        try:
            with open(patterns_path, "wb") as f:
                pickle.dump(patterns, f)
            logger.debug(f"Saved {len(patterns)} patterns to {patterns_path}")
        except Exception as e:
            logger.warning(f"Failed to save patterns to cache: {e}")

        self._patterns_cache = patterns
        self._cache_timestamp = time.time()

        logger.info(f"Built {len(patterns)} EntityRuler patterns")
        return patterns

    async def add_entity_to_cache(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        embedding: List[float],
        aliases: Optional[List[str]] = None,
    ) -> bool:
        """
        Add a new entity to the cached KB and patterns without full rebuild.

        This enables incremental updates after processing new documents.

        Args:
            entity_id: Entity ID from database
            name: Entity name
            entity_type: Entity type string
            embedding: Entity embedding vector
            aliases: Optional list of aliases

        Returns:
            True if successfully added
        """
        label = self._entity_type_to_label(entity_type)

        # Add to KB cache
        if self._kb_cache is not None:
            try:
                if len(embedding) == self.config.entity_vector_length:
                    self._kb_cache.add_entity(
                        entity=entity_id,
                        freq=1,
                        entity_vector=embedding
                    )
                    self._kb_cache.add_alias(
                        alias=name,
                        entities=[entity_id],
                        probabilities=[1.0]
                    )

                    if aliases:
                        for alias in aliases:
                            if alias and alias.strip():
                                try:
                                    self._kb_cache.add_alias(
                                        alias=alias.strip(),
                                        entities=[entity_id],
                                        probabilities=[1.0]
                                    )
                                except Exception:
                                    pass

                    logger.debug(f"Added entity '{name}' to KB cache")
            except Exception as e:
                logger.warning(f"Failed to add entity to KB cache: {e}")

        # Add to patterns cache
        if self._patterns_cache is not None:
            # Add pattern for primary name
            self._patterns_cache.append({
                "label": label,
                "pattern": name,
                "id": entity_id
            })

            # Add patterns for aliases
            if aliases:
                for alias in aliases:
                    if alias and alias.strip():
                        self._patterns_cache.append({
                            "label": label,
                            "pattern": alias.strip(),
                            "id": entity_id
                        })

            logger.debug(f"Added entity '{name}' patterns to cache")

        return True

    def invalidate_cache(self):
        """Invalidate all cached data, forcing rebuild on next access."""
        self._kb_cache = None
        self._patterns_cache = None
        self._cache_timestamp = 0
        logger.info("KnowledgeBase cache invalidated")

    def clear_disk_cache(self):
        """Remove all cached files from disk."""
        try:
            kb_path = self._get_cache_path("knowledge_base")
            patterns_path = self._get_cache_path("entity_ruler_patterns.pkl")

            if kb_path.exists():
                import shutil
                shutil.rmtree(str(kb_path))

            if patterns_path.exists():
                patterns_path.unlink()

            self.invalidate_cache()
            logger.info("Disk cache cleared")

        except Exception as e:
            logger.error(f"Failed to clear disk cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the cached KB."""
        stats = {
            "cache_valid": self._is_cache_valid(),
            "cache_timestamp": self._cache_timestamp,
            "entity_count": self._entity_count,
        }

        if self._kb_cache is not None:
            stats["kb_entity_count"] = self._kb_cache.get_size_entities()
            stats["kb_alias_count"] = self._kb_cache.get_size_aliases()

        if self._patterns_cache is not None:
            stats["pattern_count"] = len(self._patterns_cache)

        return stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


# Global builder instance for reuse
_kb_builder: Optional[SpacyKnowledgeBaseBuilder] = None


def get_kb_builder(config: Optional[KnowledgeBaseConfig] = None) -> SpacyKnowledgeBaseBuilder:
    """Get or create a singleton KnowledgeBase builder instance."""
    global _kb_builder
    if _kb_builder is None:
        _kb_builder = SpacyKnowledgeBaseBuilder(config)
    return _kb_builder


async def build_knowledge_base(vocab, config: Optional[KnowledgeBaseConfig] = None):
    """
    Convenience function to build KnowledgeBase.

    Args:
        vocab: spaCy Vocab from nlp pipeline
        config: Optional KB configuration

    Returns:
        Populated KnowledgeBase
    """
    builder = get_kb_builder(config)
    return await builder.build_knowledge_base(vocab)


async def get_entity_ruler_patterns(config: Optional[KnowledgeBaseConfig] = None) -> List[Dict]:
    """
    Convenience function to get EntityRuler patterns.

    Args:
        config: Optional KB configuration

    Returns:
        List of pattern dictionaries for EntityRuler
    """
    builder = get_kb_builder(config)
    return await builder.build_entity_ruler_patterns()


async def add_entity_to_kb(
    entity_id: str,
    name: str,
    entity_type: str,
    embedding: List[float],
    aliases: Optional[List[str]] = None,
    config: Optional[KnowledgeBaseConfig] = None,
) -> bool:
    """
    Convenience function to add a new entity to cached KB.

    Args:
        entity_id: Entity ID
        name: Entity name
        entity_type: Entity type
        embedding: Entity embedding
        aliases: Optional aliases
        config: Optional KB configuration

    Returns:
        True if successfully added
    """
    builder = get_kb_builder(config)
    return await builder.add_entity_to_cache(
        entity_id, name, entity_type, embedding, aliases
    )
