"""
Entity Linking and Deduplication

Implements KNN-based entity deduplication using embedding similarity.
This is a key component of the HippoRAG approach.

See docs/KNOWLEDGE_GRAPH_IMPLEMENTATION_PLAN.md Phase 2.2 for documentation.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from open_notebook.database.repository import repo_query
from open_notebook.domain.knowledge_graph import (
    Entity,
    EntityType,
    compute_entity_hash,
)
from open_notebook.processors.embeddings import (
    EmbeddingConfig,
    KnowledgeGraphEmbeddings,
    get_kg_embeddings,
)
from open_notebook.processors.openie import ExtractedEntity, map_entity_type


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class EntityLinkingConfig:
    """Configuration for entity linking and deduplication."""

    # Similarity threshold for considering entities as duplicates
    similarity_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("ENTITY_SIMILARITY_THRESHOLD", "0.85")
        )
    )

    # Number of nearest neighbors to consider
    knn_k: int = field(
        default_factory=lambda: int(os.getenv("ENTITY_KNN_K", "10"))
    )

    # Whether to automatically create same_as links
    auto_link_duplicates: bool = True

    # Minimum confidence to store an entity
    min_entity_confidence: float = 0.6


# =============================================================================
# ENTITY LINKER
# =============================================================================


class EntityLinker:
    """
    Links and deduplicates entities using embedding similarity.

    Uses KNN search on entity embeddings to find potential duplicates,
    then creates same_as relationships for entities above the similarity threshold.
    """

    def __init__(
        self,
        config: Optional[EntityLinkingConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
    ):
        self.config = config or EntityLinkingConfig()
        self.embeddings = get_kg_embeddings(embedding_config)

    async def find_similar_entities(
        self,
        name: str,
        description: Optional[str] = None,
        entity_type: Optional[EntityType] = None,
        limit: int = None,
    ) -> List[Tuple[Entity, float]]:
        """
        Find entities similar to the given name/description.

        Args:
            name: Entity name to search for
            description: Optional description for better matching
            entity_type: Optional type to filter by
            limit: Max number of results (defaults to config.knn_k)

        Returns:
            List of (Entity, similarity_score) tuples, sorted by similarity
        """
        limit = limit or self.config.knn_k

        # Generate embedding for the query
        embedding = await self.embeddings.embed_entity(name, description)
        if not embedding:
            logger.warning(f"Failed to generate embedding for entity: {name}")
            return []

        try:
            # Build query with optional type filter
            if entity_type:
                query = """
                SELECT *,
                    vector::similarity::cosine(embedding, $embedding) AS similarity
                FROM entity
                WHERE embedding != NONE
                    AND entity_type = $entity_type
                    AND vector::similarity::cosine(embedding, $embedding) >= $threshold
                ORDER BY similarity DESC
                LIMIT $limit
                """
                params = {
                    "embedding": embedding,
                    "entity_type": entity_type.value,
                    "threshold": self.config.similarity_threshold * 0.8,  # Lower for candidates
                    "limit": limit,
                }
            else:
                query = """
                SELECT *,
                    vector::similarity::cosine(embedding, $embedding) AS similarity
                FROM entity
                WHERE embedding != NONE
                    AND vector::similarity::cosine(embedding, $embedding) >= $threshold
                ORDER BY similarity DESC
                LIMIT $limit
                """
                params = {
                    "embedding": embedding,
                    "threshold": self.config.similarity_threshold * 0.8,
                    "limit": limit,
                }

            results = await repo_query(query, params)

            entities = []
            for r in results:
                similarity = r.pop("similarity", 0.0)
                try:
                    entity = Entity(**r)
                    entities.append((entity, similarity))
                except Exception as e:
                    logger.debug(f"Skipping invalid entity result: {e}")

            return entities

        except Exception as e:
            logger.error(f"Error finding similar entities: {e}")
            return []

    async def find_or_create_entity(
        self,
        extracted: ExtractedEntity,
        source_id: Optional[str] = None,
    ) -> Tuple[Entity, bool, Optional[str]]:
        """
        Find an existing entity or create a new one.

        This is the main deduplication entry point. It:
        1. Checks for exact hash match (same normalized name)
        2. Searches for similar entities using embedding KNN
        3. Creates a new entity or returns existing match
        4. Optionally creates same_as links for near-duplicates

        Args:
            extracted: The extracted entity from OpenIE
            source_id: Optional source document ID for provenance

        Returns:
            Tuple of (Entity, is_new, linked_entity_id)
            - is_new: True if a new entity was created
            - linked_entity_id: ID of entity this was linked to via same_as
        """
        # Check confidence threshold
        if extracted.confidence < self.config.min_entity_confidence:
            logger.debug(
                f"Entity '{extracted.name}' below confidence threshold "
                f"({extracted.confidence} < {self.config.min_entity_confidence})"
            )
            # Still create but mark low confidence
            pass

        # Map entity type
        entity_type = map_entity_type(extracted.entity_type)

        # 1. Check for exact hash match first (fast path)
        hash_id = compute_entity_hash(extracted.name)
        existing = await Entity.get_by_hash(hash_id)
        if existing:
            logger.debug(f"Found exact match for entity: {extracted.name}")
            return existing, False, None

        # 2. Generate embedding and search for similar entities
        embedding = await self.embeddings.embed_entity(
            extracted.name, extracted.description
        )

        if embedding:
            similar = await self.find_similar_entities(
                extracted.name,
                extracted.description,
                entity_type,
            )

            # Check if any match is above threshold
            for entity, similarity in similar:
                if similarity >= self.config.similarity_threshold:
                    logger.info(
                        f"Found similar entity: '{entity.name}' "
                        f"(similarity={similarity:.3f}) for '{extracted.name}'"
                    )
                    return entity, False, None

        # 3. Create new entity
        entity = Entity(
            name=extracted.name,
            entity_type=entity_type,
            description=extracted.description,
            aliases=extracted.aliases,
            embedding=embedding if embedding else None,
        )

        await entity.save()
        logger.info(f"Created new entity: {entity.name} ({entity_type.value})")

        # 4. Create same_as links for near-duplicates if configured
        linked_id = None
        if self.config.auto_link_duplicates and embedding:
            # Re-search now that entity is saved to find near-duplicates
            for other_entity, similarity in similar:
                if (
                    similarity >= self.config.similarity_threshold * 0.9
                    and similarity < self.config.similarity_threshold
                    and other_entity.id != entity.id
                ):
                    # Create bidirectional same_as link
                    await entity.create_same_as_link(
                        other_entity.id,
                        similarity,
                        method="embedding_knn",
                    )
                    linked_id = other_entity.id
                    logger.info(
                        f"Created same_as link: {entity.name} <-> {other_entity.name} "
                        f"(similarity={similarity:.3f})"
                    )
                    break  # Only link to closest near-duplicate

        return entity, True, linked_id

    async def process_extracted_entities(
        self,
        entities: List[ExtractedEntity],
        source_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a list of extracted entities with deduplication.

        Args:
            entities: List of extracted entities from OpenIE
            source_id: Optional source document ID

        Returns:
            Dict with processing statistics and entity mappings
        """
        results = {
            "total": len(entities),
            "new": 0,
            "existing": 0,
            "linked": 0,
            "entities": [],  # List of (extracted_name, entity_id, is_new)
        }

        for extracted in entities:
            try:
                entity, is_new, linked_id = await self.find_or_create_entity(
                    extracted, source_id
                )

                if is_new:
                    results["new"] += 1
                else:
                    results["existing"] += 1

                if linked_id:
                    results["linked"] += 1

                results["entities"].append({
                    "extracted_name": extracted.name,
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                    "is_new": is_new,
                    "linked_to": linked_id,
                })

            except Exception as e:
                logger.error(f"Error processing entity '{extracted.name}': {e}")

        logger.info(
            f"Processed {results['total']} entities: "
            f"{results['new']} new, {results['existing']} existing, "
            f"{results['linked']} linked"
        )

        return results

    async def merge_entities(
        self,
        primary_id: str,
        secondary_id: str,
        update_references: bool = True,
    ) -> bool:
        """
        Merge two entities, keeping the primary and removing the secondary.

        Args:
            primary_id: ID of entity to keep
            secondary_id: ID of entity to merge into primary
            update_references: Whether to update references to secondary

        Returns:
            True if merge was successful
        """
        try:
            primary = await Entity.get(primary_id)
            secondary = await Entity.get(secondary_id)

            if not primary or not secondary:
                logger.error("One or both entities not found")
                return False

            # Merge aliases
            all_aliases = set(primary.aliases)
            all_aliases.add(secondary.name)
            all_aliases.update(secondary.aliases)
            all_aliases.discard(primary.name)
            primary.aliases = list(all_aliases)

            # Merge external IDs (prefer primary)
            for key, value in secondary.external_ids.items():
                if key not in primary.external_ids:
                    primary.external_ids[key] = value

            # Update description if primary is missing one
            if not primary.description and secondary.description:
                primary.description = secondary.description

            await primary.save()

            # Update references if requested
            if update_references:
                # Update mentions relationships
                await repo_query(
                    """
                    UPDATE mentions SET out = $primary
                    WHERE out = $secondary
                    """,
                    {"primary": primary_id, "secondary": secondary_id},
                )

                # Update same_as relationships
                await repo_query(
                    """
                    UPDATE same_as SET out = $primary
                    WHERE out = $secondary AND in != $primary
                    """,
                    {"primary": primary_id, "secondary": secondary_id},
                )
                await repo_query(
                    """
                    UPDATE same_as SET in = $primary
                    WHERE in = $secondary AND out != $primary
                    """,
                    {"primary": primary_id, "secondary": secondary_id},
                )

            # Delete secondary entity
            await secondary.delete()

            logger.info(f"Merged entity '{secondary.name}' into '{primary.name}'")
            return True

        except Exception as e:
            logger.error(f"Error merging entities: {e}")
            return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def link_entity(
    name: str,
    entity_type: str,
    description: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    config: Optional[EntityLinkingConfig] = None,
) -> Entity:
    """
    Convenience function to find or create an entity with deduplication.

    Args:
        name: Entity name
        entity_type: Entity type string
        description: Optional description
        aliases: Optional list of aliases
        config: Optional linking configuration

    Returns:
        The found or created Entity
    """
    extracted = ExtractedEntity(
        name=name,
        entity_type=entity_type,
        description=description,
        aliases=aliases or [],
        confidence=1.0,  # Manual entities are high confidence
    )

    linker = EntityLinker(config)
    entity, _, _ = await linker.find_or_create_entity(extracted)
    return entity


async def find_duplicate_entities(
    similarity_threshold: float = 0.9,
    limit: int = 100,
) -> List[Tuple[Entity, Entity, float]]:
    """
    Find potential duplicate entities in the database.

    Useful for manual review and cleanup.

    Args:
        similarity_threshold: Minimum similarity to consider as duplicate
        limit: Maximum number of pairs to return

    Returns:
        List of (entity1, entity2, similarity) tuples
    """
    try:
        # This query finds pairs of similar entities
        results = await repo_query(
            """
            SELECT
                e1.*,
                e2.id AS e2_id,
                e2.name AS e2_name,
                vector::similarity::cosine(e1.embedding, e2.embedding) AS similarity
            FROM entity AS e1, entity AS e2
            WHERE e1.embedding != NONE
                AND e2.embedding != NONE
                AND e1.id < e2.id
                AND vector::similarity::cosine(e1.embedding, e2.embedding) >= $threshold
            ORDER BY similarity DESC
            LIMIT $limit
            """,
            {"threshold": similarity_threshold, "limit": limit},
        )

        pairs = []
        for r in results:
            e2_id = r.pop("e2_id")
            e2_name = r.pop("e2_name")
            similarity = r.pop("similarity")

            entity1 = Entity(**r)
            entity2 = await Entity.get(e2_id)

            if entity2:
                pairs.append((entity1, entity2, similarity))

        return pairs

    except Exception as e:
        logger.error(f"Error finding duplicate entities: {e}")
        return []
