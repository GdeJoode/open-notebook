"""
Repository for entity resolution operations against the knowledge graph.

Provides lookup and alias management for canonical entity matching,
supporting the KG entity resolution pipeline.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query


class EntityRepository:
    """Repository for entity resolution operations.

    Handles entity lookup by alias, type-based candidate retrieval,
    alias registration, and embedding access for the knowledge graph
    entity resolution pipeline.

    This repository does not inherit from BaseRepository because
    there is no corresponding Entity model in the shared models package.
    It operates directly against the entity and entity_alias tables
    using raw SurrealQL queries.
    """

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        """Initialize the entity repository.

        Args:
            config: Optional SurrealDB configuration. Uses the global
                configuration if not provided.
        """
        self.config = config

    async def find_by_alias(self, alias_text: str) -> Optional[Dict[str, Any]]:
        """Find a canonical entity by exact alias text match.

        Queries the entity_alias table for an exact match on the alias text
        and returns the associated canonical entity information.

        Args:
            alias_text: The alias text to search for (exact match).

        Returns:
            A dictionary with keys ``id``, ``name``, ``match_type``, and
            ``similarity_score`` for the resolved canonical entity, or
            None if no match is found.
        """
        if not alias_text:
            return None

        try:
            result = await execute_query(
                "SELECT canonical_entity.id AS id, "
                "canonical_entity.name AS name, "
                "match_type, similarity_score "
                "FROM entity_alias "
                "WHERE alias_text = $alias_text LIMIT 1",
                {"alias_text": alias_text},
                self.config,
            )
            if result:
                return result[0]
            return None
        except Exception as e:
            logger.error(f"Failed to find entity by alias '{alias_text}': {e}")
            return None

    async def find_by_type(
        self, entity_type: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get candidate entities for matching by entity type.

        Retrieves entities of a given type along with their embeddings,
        which can be used for similarity-based entity resolution.

        Args:
            entity_type: The entity type to filter by (e.g. "PERSON", "ORG").
            limit: Maximum number of entities to return. Defaults to 100.

        Returns:
            A list of dictionaries, each containing ``id``, ``name``,
            ``embedding``, and ``weight`` for a matching entity. Returns
            an empty list on failure.
        """
        try:
            return await execute_query(
                "SELECT id, name, embedding, weight "
                "FROM entity "
                "WHERE entity_type = $entity_type LIMIT $limit",
                {"entity_type": entity_type, "limit": limit},
                self.config,
            )
        except Exception as e:
            logger.error(
                f"Failed to find entities by type '{entity_type}': {e}"
            )
            return []

    async def register_alias(
        self,
        canonical_entity_id: str,
        alias_text: str,
        match_type: str,
        similarity_score: float,
        method: str = "",
    ) -> bool:
        """Register a new alias for a canonical entity.

        Creates a mapping from the given alias text to the specified
        canonical entity. If the alias already exists for this entity,
        the operation is skipped and True is returned.

        Args:
            canonical_entity_id: The record ID of the canonical entity
                (e.g. "entity:abc123").
            alias_text: The alias text to register.
            match_type: How the match was determined (e.g. "exact",
                "fuzzy", "embedding").
            similarity_score: Confidence score of the match (0.0 to 1.0).
            method: Optional description of the resolution method used.

        Returns:
            True if the alias was registered or already exists, False on
            failure.
        """
        if not canonical_entity_id or not alias_text:
            return False

        try:
            entity_id = ensure_record_id(canonical_entity_id)

            # Check if alias already exists for this entity
            existing = await execute_query(
                "SELECT id FROM entity_alias "
                "WHERE canonical_entity = $entity_id "
                "AND alias_text = $alias_text LIMIT 1",
                {"entity_id": entity_id, "alias_text": alias_text},
                self.config,
            )
            if existing:
                return True

            # Insert the new alias
            await execute_query(
                "INSERT INTO entity_alias { "
                "canonical_entity: $entity_id, "
                "alias_text: $alias_text, "
                "match_type: $match_type, "
                "similarity_score: $similarity_score, "
                "method: $method, "
                "verified: false "
                "}",
                {
                    "entity_id": entity_id,
                    "alias_text": alias_text,
                    "match_type": match_type,
                    "similarity_score": similarity_score,
                    "method": method,
                },
                self.config,
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to register alias '{alias_text}' "
                f"for entity '{canonical_entity_id}': {e}"
            )
            return False

    async def get_entity_with_embedding(
        self, entity_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a single entity with its embedding vector.

        Args:
            entity_id: The record ID of the entity to retrieve
                (e.g. "entity:abc123").

        Returns:
            A dictionary with keys ``id``, ``name``, ``entity_type``, and
            ``embedding``, or None if the entity is not found.
        """
        if not entity_id:
            return None

        try:
            result = await execute_query(
                "SELECT id, name, entity_type, embedding "
                "FROM entity WHERE id = $id",
                {"id": ensure_record_id(entity_id)},
                self.config,
            )
            if result:
                return result[0]
            return None
        except Exception as e:
            logger.error(
                f"Failed to get entity with embedding '{entity_id}': {e}"
            )
            return None
