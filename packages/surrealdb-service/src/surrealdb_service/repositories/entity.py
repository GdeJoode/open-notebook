"""
Repository for entity resolution operations against the knowledge graph.

Provides lookup and alias management for canonical entity matching,
supporting the KG entity resolution pipeline.
"""

import hashlib
from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models.entity import Entity, Relation
from shared.models.export import ExportFilter
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query


def _union_preserve_order(
    existing: List[Any], incoming: List[Any]
) -> List[Any]:
    """Return ``existing + (incoming \\ existing)``, preserving order.

    Mirrors what ``array::union`` does in SurrealDB (dedup with stable order
    of first appearance). We implement it in Python because ``upsert_entity``
    pre-fetches the row and performs the merge client-side — see that
    docstring for rationale.

    Nested dicts/lists inside provenance_chain are JSON-comparable but not
    hashable, so we fall back to an O(n*m) scan rather than a set.
    """
    out = list(existing)
    for item in incoming:
        if item not in out:
            out.append(item)
    return out


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

    async def upsert_entity(self, entity: Entity) -> str:
        """Upsert a canonical entity, returning its record ID.

        Lookup is by ``(canonical_name, entity_type)`` — the unique-index pair
        declared in migration 39 (``idx_entity_name_type``). When a row with
        the same name+type already exists, the upsert merges (Python-side
        merge of provenance/scoring fields, then a single UPDATE):

        - confidence: ``max`` of existing vs new (monotonic, never drops)
        - source_documents: union (dedup-preserving)
        - properties: dict overlay — new keys win, existing keys retained
        - type_tags: union (multi-type accumulation from B1 merge)
        - primary_type: replaced with the new value when supplied
        - provenance_chain: union

        All other fields (``status``, ``embedding``, ``description``, ...) keep
        the existing row's value on update. New rows take all fields from
        ``entity`` directly, including the required ``embedding`` (which has
        no DB-side default — see migration 39 line 30).

        Why we merge in Python rather than in SurrealQL: ``object::extend`` /
        ``object::merge`` are not available on SurrealDB v2.x at the time of
        writing — calls produce
        ``Parse error: Invalid function/constant path``. Pre-fetching the
        existing row (one short SELECT) and applying the merge in Python keeps
        the contract identical while sidestepping the SurrealQL gap.

        Args:
            entity: An ``Entity`` model populated with the canonical-schema
                field names. ``embedding`` MUST be a list (use ``[]`` when
                no vector is available).

        Returns:
            The record ID of the upserted entity, e.g. ``"entity:abc123"``.

        Raises:
            RuntimeError: If the query fails.

        Note:
            The SELECT-then-UPDATE flow is not atomic — two concurrent
            ``upsert_entity`` calls for the same
            ``(canonical_name, entity_type)`` can both observe "no
            existing row" and then race on CREATE; the migration-39
            ``idx_entity_name_type`` UNIQUE index will reject one of
            them. This is acceptable today because all writers go
            through the single-process ``EntityPersistenceService``.
            **B.1e must wrap this in a per-canonical-name lock or move
            the merge into a SurrealDB transaction** before introducing
            parallel writers.
        """
        try:
            existing_rows = await execute_query(
                "SELECT * FROM entity "
                "WHERE canonical_name = $canonical_name "
                "AND entity_type = $entity_type LIMIT 1",
                {
                    "canonical_name": entity.canonical_name,
                    "entity_type": entity.entity_type,
                },
                self.config,
            )
        except Exception as e:
            logger.exception(
                f"upsert_entity lookup failed for "
                f"'{entity.canonical_name}' ({entity.entity_type}): {e}"
            )
            raise

        if existing_rows:
            existing = existing_rows[0]
            merged_confidence = max(
                float(existing.get("confidence", 0.0) or 0.0),
                entity.confidence,
            )
            merged_sources = _union_preserve_order(
                existing.get("source_documents") or [],
                list(entity.source_documents),
            )
            merged_type_tags = _union_preserve_order(
                existing.get("type_tags") or [],
                list(entity.type_tags),
            )
            merged_provenance = _union_preserve_order(
                existing.get("provenance_chain") or [],
                list(entity.provenance_chain),
            )
            merged_properties: Dict[str, Any] = dict(
                existing.get("properties") or {}
            )
            merged_properties.update(entity.properties)

            update_payload = {
                "id": existing["id"],
                "confidence": merged_confidence,
                "source_documents": merged_sources,
                "properties": merged_properties,
                "type_tags": merged_type_tags,
                "primary_type": entity.primary_type
                if entity.primary_type is not None
                else existing.get("primary_type"),
                "provenance_chain": merged_provenance,
            }
            try:
                await execute_query(
                    """
                    UPDATE type::thing($id) SET
                        confidence = $confidence,
                        source_documents = $source_documents,
                        properties = $properties,
                        type_tags = $type_tags,
                        primary_type = $primary_type,
                        provenance_chain = $provenance_chain,
                        updated_at = time::now();
                    """,
                    update_payload,
                    self.config,
                )
            except Exception as e:
                logger.exception(
                    f"upsert_entity update failed for "
                    f"'{entity.canonical_name}' ({entity.entity_type}): {e}"
                )
                raise
            return str(existing["id"])

        # No existing row — fresh CREATE.
        # ``hash_id`` is a required string on the live DB (schema drift — not in
        # any migration, so the testcontainer schema doesn't enforce it and the
        # roundtrip tests pass while live writes fail with "Found NONE for field
        # hash_id"). Derive it deterministically from the EXACT dedup identity
        # (canonical_name + entity_type) used by the existing-row lookup and the
        # case-sensitive ``idx_entity_name_type``. Do NOT lower-case/strip here:
        # the dedup key is case-sensitive, so folding case would make distinct
        # entities ("BZK" vs "bzk") collide on the UNIQUE ``idx_entity_hash``.
        _hash_basis = f"{entity.canonical_name}|{entity.entity_type}"
        create_payload: Dict[str, Any] = {
            "canonical_name": entity.canonical_name,
            "entity_type": entity.entity_type,
            "hash_id": hashlib.md5(_hash_basis.encode("utf-8")).hexdigest(),
            "description": entity.description,
            "source_documents": list(entity.source_documents),
            "extraction_method": entity.extraction_method,
            "confidence": entity.confidence,
            "provenance_chain": list(entity.provenance_chain),
            "properties": dict(entity.properties),
            "embedding": list(entity.embedding),
            "status": entity.status,
            "type_tags": list(entity.type_tags),
            "primary_type": entity.primary_type,
        }

        try:
            result = await execute_query(
                """
                CREATE entity SET
                    canonical_name = $canonical_name,
                    name = $canonical_name,
                    entity_type = $entity_type,
                    hash_id = $hash_id,
                    description = $description,
                    source_documents = $source_documents,
                    extraction_method = $extraction_method,
                    confidence = $confidence,
                    provenance_chain = $provenance_chain,
                    properties = $properties,
                    embedding = $embedding,
                    status = $status,
                    type_tags = $type_tags,
                    primary_type = $primary_type;
                """,
                create_payload,
                self.config,
            )
        except Exception as e:
            logger.exception(
                f"upsert_entity create failed for "
                f"'{entity.canonical_name}' ({entity.entity_type}): {e}"
            )
            raise

        if not result:
            raise RuntimeError(
                f"CREATE entity returned no rows for "
                f"'{entity.canonical_name}' ({entity.entity_type})"
            )
        return str(result[0]["id"])

    async def get_entity(self, record_id: str) -> Optional[Entity]:
        """Fetch a single entity by record ID, returning a typed ``Entity``.

        Selects all migration-39/44 fields and parses the result into the
        Pydantic model. Used by B.1e's merge step to pick canonical winners
        by recency (and elsewhere wherever a typed handle is preferred over
        a raw dict).

        Args:
            record_id: Entity record ID (e.g. ``"entity:abc123"``).

        Returns:
            An ``Entity`` instance, or ``None`` if no row exists.
        """
        if not record_id:
            return None
        try:
            rid = ensure_record_id(record_id)
            rows = await execute_query(
                "SELECT * FROM entity WHERE id = $id LIMIT 1",
                {"id": rid},
                self.config,
            )
        except Exception as e:
            logger.error(f"Failed to get entity '{record_id}': {e}")
            return None
        if not rows:
            return None
        # ``execute_query`` already converts RecordIDs to strings.
        return Entity(**rows[0])

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

    async def list_entities(
        self,
        limit: int = 50,
        offset: int = 0,
        entity_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List entities with pagination and optional type filter.

        Args:
            limit: Maximum number of entities to return.
            offset: Number of entities to skip.
            entity_type: Optional entity type filter.

        Returns:
            A list of entity dictionaries.
        """
        # NOTE: ``WITH NOINDEX`` is required here. ``entity.name`` is covered by
        # the full-text SEARCH index ``idx_entity_fulltext`` (name, description).
        # In SurrealDB v2 the query planner cannot satisfy ``ORDER BY name`` from
        # a SEARCH index and aborts the transaction with "No iterator has been
        # found." Forcing a no-index scan + in-memory sort avoids that planner
        # path. Entity tables are small enough that the full scan is acceptable.
        try:
            if entity_type:
                return await execute_query(
                    "SELECT id, name, entity_type, weight, confidence "
                    "FROM entity WITH NOINDEX WHERE entity_type = $entity_type "
                    "ORDER BY name LIMIT $limit START $offset",
                    {"entity_type": entity_type, "limit": limit, "offset": offset},
                    self.config,
                )
            return await execute_query(
                "SELECT id, name, entity_type, weight, confidence "
                "FROM entity WITH NOINDEX ORDER BY name LIMIT $limit START $offset",
                {"limit": limit, "offset": offset},
                self.config,
            )
        except Exception as e:
            logger.error(f"Failed to list entities: {e}")
            return []

    async def count_entities(
        self, entity_type: Optional[str] = None
    ) -> int:
        """Count entities with optional type filter.

        Args:
            entity_type: Optional entity type filter.

        Returns:
            Total count of matching entities.
        """
        try:
            if entity_type:
                result = await execute_query(
                    "SELECT count() AS total FROM entity "
                    "WHERE entity_type = $entity_type GROUP ALL",
                    {"entity_type": entity_type},
                    self.config,
                )
            else:
                result = await execute_query(
                    "SELECT count() AS total FROM entity GROUP ALL",
                    {},
                    self.config,
                )
            if result:
                return result[0].get("total", 0)
            return 0
        except Exception as e:
            logger.error(f"Failed to count entities: {e}")
            return 0

    async def get_entity_detail(
        self, entity_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a single entity with its relations.

        Args:
            entity_id: The entity record ID.

        Returns:
            Entity dict with a ``relations`` list, or None.
        """
        if not entity_id:
            return None
        try:
            eid = ensure_record_id(entity_id)
            result = await execute_query(
                "SELECT * FROM entity WHERE id = $id",
                {"id": eid},
                self.config,
            )
            if not result:
                return None

            entity = result[0]

            # Get relations where this entity is source or target. Confidence is
            # surfaced so the KG UI can render a per-relation confidence bar
            # (B.4 frontend).
            relations = await execute_query(
                "SELECT id, in AS source, out AS target, relation_type, confidence "
                "FROM relation WHERE in = $id OR out = $id",
                {"id": eid},
                self.config,
            )
            entity["relations"] = relations or []
            return entity
        except Exception as e:
            logger.error(f"Failed to get entity detail '{entity_id}': {e}")
            return None

    async def get_entity_types_summary(self) -> List[Dict[str, Any]]:
        """Get entity counts grouped by type.

        Returns:
            List of dicts with ``entity_type`` and ``count`` keys.
        """
        try:
            return await execute_query(
                "SELECT entity_type, count() AS count "
                "FROM entity GROUP BY entity_type ORDER BY count DESC",
                {},
                self.config,
            )
        except Exception as e:
            logger.error(f"Failed to get entity types summary: {e}")
            return []

    async def search_entities(
        self, query: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search entities by name using string containment.

        Args:
            query: Search text.
            limit: Maximum results.

        Returns:
            Matching entity dicts.
        """
        try:
            return await execute_query(
                "SELECT id, name, entity_type, weight, confidence "
                "FROM entity WHERE string::contains(string::lowercase(name), "
                "string::lowercase($query)) LIMIT $limit",
                {"query": query, "limit": limit},
                self.config,
            )
        except Exception as e:
            logger.error(f"Failed to search entities for '{query}': {e}")
            return []

    async def get_all_entities_and_relations(
        self,
        entity_type: Optional[str] = None,
        limit: int = 5000,
    ) -> Dict[str, Any]:
        """Get raw nodes and edges for graph visualization.

        Args:
            entity_type: Optional type filter.
            limit: Max nodes.

        Returns:
            Dict with ``nodes`` and ``edges`` lists.
        """
        try:
            if entity_type:
                nodes = await execute_query(
                    "SELECT id, name, entity_type, weight "
                    "FROM entity WHERE entity_type = $entity_type LIMIT $limit",
                    {"entity_type": entity_type, "limit": limit},
                    self.config,
                )
            else:
                nodes = await execute_query(
                    "SELECT id, name, entity_type, weight "
                    "FROM entity LIMIT $limit",
                    {"limit": limit},
                    self.config,
                )

            if not nodes:
                return {"nodes": [], "edges": []}

            # Get all edges connecting the retrieved nodes
            node_ids = [n["id"] for n in nodes]
            edges = await execute_query(
                "SELECT id, in AS source, out AS target, relation_type "
                "FROM relation WHERE in INSIDE $ids AND out INSIDE $ids",
                {"ids": node_ids},
                self.config,
            )

            return {"nodes": nodes or [], "edges": edges or []}
        except Exception as e:
            logger.error(f"Failed to get graph data: {e}")
            return {"nodes": [], "edges": []}

    async def list_orphans_for_source(
        self, source_id: str
    ) -> List[Dict[str, Any]]:
        """Return entities of this source with no incoming/outgoing relations.

        An "orphan" for the orphan-connector (B.5a) is an entity that:

        - lists *source_id* in its ``source_documents`` array (the
          provenance bag populated by ``upsert_entity``), AND
        - participates in zero rows of the ``relation`` RELATE table
          (no edge has ``in = entity.id`` and none has
          ``out = entity.id``).

        The query is implemented in two steps because SurrealDB's
        sub-select-count syntax for RELATE tables is awkward — we lift
        the per-entity edge probe to the Python side and reject any
        entity with non-zero degree. The probe uses ``LIMIT 1`` so each
        round-trip costs at most one row.

        Cross-source semantics (Minor-5, B.5a attempt 2):
            The edge-probe is GLOBAL — it considers every relation row
            regardless of which source produced it. An entity that
            appears in multiple sources is therefore "orphan" only when
            it has zero edges across ALL sources. This is intentional:
            a relation imported from source X already connects the
            entity to the graph, so re-running the orphan-connector for
            source Y has nothing to add.

        Args:
            source_id: Record ID of the source (e.g. ``"source:abc"``).

        Returns:
            A list of entity-row dicts (``id``, ``canonical_name``,
            ``entity_type``, ``source_documents``, ``properties``). Empty
            when the source has no entities or every entity is
            connected. Always shaped — never ``None``.
        """
        if not source_id:
            logger.warning(
                "list_orphans_for_source called with empty source_id"
            )
            return []

        try:
            entities = await execute_query(
                "SELECT id, canonical_name, entity_type, "
                "source_documents, properties "
                "FROM entity "
                "WHERE $source_id IN source_documents",
                {"source_id": source_id},
                self.config,
            )
        except Exception as e:
            logger.error(
                f"Failed to list source entities for orphan-detection "
                f"on '{source_id}': {e}"
            )
            return []

        if not entities:
            return []

        orphans: List[Dict[str, Any]] = []
        for entity in entities:
            eid = entity.get("id")
            if eid is None:
                continue
            try:
                edges = await execute_query(
                    "SELECT id FROM relation "
                    "WHERE in = $eid OR out = $eid LIMIT 1",
                    {"eid": eid},
                    self.config,
                )
            except Exception as e:
                logger.error(
                    f"Failed to count edges for entity '{eid}' "
                    f"in orphan-detection: {e}"
                )
                continue
            if not edges:
                orphans.append(entity)

        logger.info(
            "list_orphans_for_source: source={source_id} "
            "total_entities={te} orphans={no}",
            source_id=source_id,
            te=len(entities),
            no=len(orphans),
        )
        return orphans

    async def list_orphans_with_status(
        self,
        notebook_id: str,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List entities in a notebook filtered by ``orphan_status``.

        Powers the per-notebook orphans dashboard (B.5b). An entity
        "belongs" to a notebook when one of its ``source_documents``
        was imported via that notebook. Because source ↔ notebook is
        stored on the ``reference`` RELATE edge (NOT a column on
        ``source`` -- see migration 1), the query is a two-step:

        1. Resolve every source RecordID linked to the notebook via the
           ``reference`` edge.
        2. SELECT entities whose ``source_documents`` array intersects
           that set AND ``orphan_status`` matches the requested status.

        The ``source_documents`` field stores stringified record ids
        (see ``EntityRepository.upsert_entity`` / ``ExtractedEntity``)
        rather than ``record<source>`` values. The fix vs the original
        broken query (B.5b attempt 1):

        * The source-list step now traverses ``reference`` -- the table
          ``source`` carries no ``notebook`` column, so the previous
          ``WHERE notebook = $notebook_id`` always returned 0 rows.
        * Source ids are stringified before being handed to the entity
          query so the ``ANYINSIDE`` comparison happens in string space
          (matches how ``source_documents`` actually stores its values).

        Args:
            notebook_id: Record ID of the notebook
                (e.g. ``"notebook:abc"``).
            status: Optional ``orphan_status`` filter. When ``None``,
                returns entities with ANY non-NONE status
                (``pending_reconnect`` or ``archived``). When supplied,
                returns only entities matching that exact value.

        Returns:
            A list of entity rows shaped for the dashboard:
            ``id``, ``canonical_name``, ``entity_type``,
            ``orphan_status``, ``reconnect_attempts``,
            ``first_orphaned_at``, ``last_reconnect_attempt_at``,
            ``source_documents``. Empty on failure or no matches.
        """
        if not notebook_id:
            logger.warning(
                "list_orphans_with_status called with empty notebook_id"
            )
            return []

        try:
            # The ``reference`` edge runs source -> notebook (migration
            # 1, line 54-56); ``in`` is the source RecordID, ``out`` is
            # the notebook RecordID. Same projection-style query used by
            # ``SourceRepository.list_sources`` for the per-notebook view.
            source_rows = await execute_query(
                "SELECT VALUE in FROM reference WHERE out = type::thing($notebook_id)",
                {"notebook_id": notebook_id},
                self.config,
            )
        except Exception as e:
            logger.error(
                f"Failed to list sources for orphan dashboard on "
                f"'{notebook_id}': {e}"
            )
            return []

        if not source_rows:
            return []

        # ``SELECT VALUE in`` returns a flat list of RecordID values;
        # stringify so the comparison below happens against the same
        # representation ``source_documents`` stores (see upsert_entity).
        source_ids = [str(row) for row in source_rows if row]
        if not source_ids:
            return []

        try:
            if status is not None:
                rows = await execute_query(
                    "SELECT id, canonical_name, entity_type, "
                    "orphan_status, reconnect_attempts, first_orphaned_at, "
                    "last_reconnect_attempt_at, source_documents "
                    "FROM entity "
                    "WHERE orphan_status = $status "
                    "AND source_documents ANYINSIDE $source_ids "
                    "ORDER BY first_orphaned_at DESC",
                    {"status": status, "source_ids": source_ids},
                    self.config,
                )
            else:
                rows = await execute_query(
                    "SELECT id, canonical_name, entity_type, "
                    "orphan_status, reconnect_attempts, first_orphaned_at, "
                    "last_reconnect_attempt_at, source_documents "
                    "FROM entity "
                    "WHERE orphan_status != NONE "
                    "AND source_documents ANYINSIDE $source_ids "
                    "ORDER BY first_orphaned_at DESC",
                    {"source_ids": source_ids},
                    self.config,
                )
        except Exception as e:
            logger.error(
                f"Failed to list orphans-with-status for notebook "
                f"'{notebook_id}' (status={status}): {e}"
            )
            return []
        return rows or []

    async def update_orphan_status(
        self,
        entity_id: str,
        status: Optional[str],
        *,
        increment_attempts: bool = False,
        set_first_orphaned_at: bool = False,
        set_last_reconnect_attempt_at: bool = False,
    ) -> bool:
        """Update an entity's orphan-prune lifecycle fields.

        Single helper that drives all three lifecycle transitions in
        B.5b (mark pending, retry attempt, archive). The flags let the
        caller compose the exact field-set rather than running multiple
        round-trips per transition.

        Args:
            entity_id: Record ID of the entity to update.
            status: New ``orphan_status`` value. ``None`` clears the
                field (entity returns to "healthy" state).
            increment_attempts: When ``True``, ``reconnect_attempts``
                is incremented atomically. Used on retry-attempt rows.
            set_first_orphaned_at: When ``True`` AND
                ``first_orphaned_at`` is currently NONE, sets it to
                ``time::now()``. Idempotent — never overwrites an
                existing timestamp.
            set_last_reconnect_attempt_at: When ``True``,
                ``last_reconnect_attempt_at`` is set to ``time::now()``.

        Returns:
            ``True`` when the update query ran successfully (even if
            no row matched), ``False`` on transport-level failure.
        """
        if not entity_id:
            logger.warning(
                "update_orphan_status called with empty entity_id"
            )
            return False

        try:
            rid = ensure_record_id(entity_id)
        except Exception as e:
            logger.error(
                f"update_orphan_status: invalid entity_id '{entity_id}': {e}"
            )
            return False

        # Build the SET clause dynamically so we only touch the fields
        # the caller asked for. Each branch is independently testable.
        set_clauses = ["orphan_status = $status"]
        params: Dict[str, Any] = {"id": rid, "status": status}
        if increment_attempts:
            # Truthy-coalesce ``reconnect_attempts OR 0`` -- guards
            # against the NONE-on-legacy-row case where the migration
            # ``DEFAULT 0`` did not apply (rare, but cheap to defend).
            # Note (attempt 2 Minor-2): an earlier comment claimed
            # ``math::max([..., 0])`` here; rewritten to match the
            # actual SurrealQL OR-fallback above.
            set_clauses.append(
                "reconnect_attempts = "
                "(reconnect_attempts OR 0) + 1"
            )
        if set_first_orphaned_at:
            # Idempotent: only writes when the field is currently NONE.
            set_clauses.append(
                "first_orphaned_at = IF first_orphaned_at = NONE "
                "THEN time::now() ELSE first_orphaned_at END"
            )
        if set_last_reconnect_attempt_at:
            set_clauses.append(
                "last_reconnect_attempt_at = time::now()"
            )

        query = (
            "UPDATE type::thing($id) SET "
            + ", ".join(set_clauses)
            + ";"
        )

        try:
            await execute_query(query, params, self.config)
        except Exception as e:
            logger.error(
                f"update_orphan_status failed for '{entity_id}' "
                f"(status={status}): {e}"
            )
            return False
        return True

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

    # ------------------------------------------------------------------
    # Track D Phase D.0 — notebook-scoped export projections
    # ------------------------------------------------------------------
    #
    # ``list_entities_for_notebook`` / ``list_relations_for_notebook``
    # power every Track D exporter (D.1 Obsidian, D.2 JSONL, D.3 NetworkX).
    # They reuse the notebook→source→entity traversal proven by
    # ``list_orphans_with_status`` (line 688) but project away the
    # ``embedding`` field per Q-D-1 to bound memory — 10K entities * 768
    # floats * 8 bytes ≈ 60MB the exporter never needs.

    # Field list mirrors the Entity model **minus** ``embedding``. Kept
    # as a module-private constant so both list_* methods (and any future
    # export-side projection) share the same projection.
    _ENTITY_EXPORT_FIELDS = (
        "id, created_at, updated_at, canonical_name, entity_type, "
        "description, source_documents, extracted_at, extraction_method, "
        "confidence, provenance_chain, properties, "
        "pagerank, betweenness, community_id, status, merged_into, "
        "type_tags, primary_type, "
        # B.5b orphan-prune lifecycle. Selected so the WHERE filter can
        # see them; Entity.model_validate silently ignores extras.
        "orphan_status, reconnect_attempts, first_orphaned_at, "
        "last_reconnect_attempt_at"
    )

    async def list_entities_for_notebook(
        self,
        notebook_id: str,
        filter: ExportFilter,
    ) -> List[Entity]:
        """Notebook-scoped projection of the full Track-B entity field set.

        Reuses the reference RELATE edge pattern proven by
        ``list_orphans_with_status`` (line 688) for the
        notebook→source→entity traversal: source ↔ notebook lives on the
        ``reference`` RELATE edge (migration 1), so we resolve the
        notebook's source-ids first and then filter entities whose
        ``source_documents`` array intersects that set.

        Projects away the ``embedding`` field per Q-D-1 (Track D plan)
        to bound memory at scale — 10K entities × 768-float embeddings
        × 8 bytes ≈ 60MB that Obsidian/JSONL/NetworkX exporters never
        need. The Entity Pydantic model defaults ``embedding=[]`` so the
        rehydration is type-safe even without the column.

        Filters applied in SurrealQL (NOT in Python — keep the projection
        cheap when the notebook is large):

        * ``filter.min_confidence`` → ``confidence >= $min_confidence``
        * ``filter.entity_types`` → ``primary_type INSIDE $entity_types``
          (B.1a multi-type tagging: the "best" type is the gating one)
        * ``filter.include_orphans = False`` →
          ``orphan_status != 'pending_reconnect'``
        * ``filter.include_archived = False`` →
          ``orphan_status != 'archived'``

        ``min_connections`` is NOT enforced in this method -- counting
        an entity's degree across the ``relation`` table is a join the
        export pipeline does once via ``list_relations_for_notebook``.
        Callers (D.1/D.2/D.3) post-filter on degree there.

        Args:
            notebook_id: Record ID of the notebook (e.g. ``"notebook:abc"``).
            filter: ``ExportFilter`` from ``shared.models.export``.

        Returns:
            List of ``Entity`` rows matching the filter. Empty on
            failure, empty notebook, or fully-filtered output.
        """
        if not notebook_id:
            logger.warning(
                "list_entities_for_notebook called with empty notebook_id"
            )
            return []

        # Step 1: resolve notebook → sources via the reference edge.
        # Identical pattern to ``list_orphans_with_status`` (line 744).
        try:
            source_rows = await execute_query(
                "SELECT VALUE in FROM reference "
                "WHERE out = type::thing($notebook_id)",
                {"notebook_id": notebook_id},
                self.config,
            )
        except Exception as e:
            logger.error(
                f"Failed to list sources for notebook export "
                f"'{notebook_id}': {e}"
            )
            return []

        if not source_rows:
            return []

        # ``source_documents`` stores stringified record ids (see
        # ``upsert_entity``), so compare in string space.
        source_ids = [str(row) for row in source_rows if row]
        if not source_ids:
            return []

        # Step 2: build the WHERE clause incrementally so the planner
        # sees the most-selective predicate (source-ids intersection)
        # first. Unbound knobs (None/include_*=True) don't emit a clause.
        clauses: List[str] = [
            "source_documents ANYINSIDE $source_ids",
            "confidence >= $min_confidence",
        ]
        params: Dict[str, Any] = {
            "source_ids": source_ids,
            "min_confidence": filter.min_confidence,
        }

        if filter.entity_types is not None:
            clauses.append("primary_type INSIDE $entity_types")
            params["entity_types"] = filter.entity_types

        if not filter.include_orphans:
            # ``!=`` returns true when the field is NONE (healthy row),
            # so the negation works cleanly without an explicit IS NONE
            # branch.
            clauses.append("orphan_status != 'pending_reconnect'")

        if not filter.include_archived:
            clauses.append("orphan_status != 'archived'")

        query = (
            f"SELECT {self._ENTITY_EXPORT_FIELDS} FROM entity "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY canonical_name"
        )

        try:
            rows = await execute_query(query, params, self.config)
        except Exception as e:
            logger.error(
                f"list_entities_for_notebook failed for notebook "
                f"'{notebook_id}': {e}"
            )
            return []

        if not rows:
            return []

        # ``execute_query`` already stringifies RecordIDs; Entity model
        # tolerates the extra orphan_* fields (no ``extra="forbid"``).
        entities: List[Entity] = []
        for row in rows:
            try:
                entities.append(Entity.model_validate(row))
            except Exception as e:
                logger.warning(
                    f"list_entities_for_notebook: skipping malformed "
                    f"row id={row.get('id')!r}: {e}"
                )
        return entities

    async def list_relations_for_notebook(
        self,
        notebook_id: str,
        filter: ExportFilter,
    ) -> List[Relation]:
        """Relations between entities in the notebook (after entity filter).

        Two-phase pattern:

        1. Resolve the notebook's entity-id set via
           ``list_entities_for_notebook(notebook_id, filter)``.
        2. SELECT relations where BOTH ``in`` AND ``out`` belong to that
           set, with relation-confidence filtered by
           ``min_relation_confidence`` (or ``min_confidence`` when the
           caller left it ``None`` -- the shared-knob fallback documented
           in ``ExportFilter``).

        Per Q-D-4 (Track D plan), relations whose target was filtered
        out are silently dropped. The two-phase set-intersection
        implements that: an edge into a filtered entity disappears
        because that entity isn't in the id set.

        Args:
            notebook_id: Record ID of the notebook.
            filter: ``ExportFilter`` -- the same instance passed to
                ``list_entities_for_notebook`` for consistency.

        Returns:
            List of ``Relation`` rows. Empty on failure, empty notebook,
            or zero edges among filtered entities.
        """
        entities = await self.list_entities_for_notebook(notebook_id, filter)
        if not entities:
            return []

        # ``in`` / ``out`` on the relation table are ``record<entity>``
        # columns. SurrealQL compares them in RecordID space -- passing
        # plain strings here causes ``INSIDE`` to return zero rows, which
        # silently drops every edge. Coerce via ``ensure_record_id``.
        entity_ids: List[Any] = []
        for e in entities:
            if not e.id:
                continue
            try:
                entity_ids.append(ensure_record_id(e.id))
            except Exception as exc:
                logger.warning(
                    f"list_relations_for_notebook: bad entity id "
                    f"{e.id!r}: {exc}"
                )
        if not entity_ids:
            return []

        # ``min_relation_confidence`` inherits from ``min_confidence``
        # when None -- single-knob ergonomics documented on the model.
        min_rel_conf = (
            filter.min_relation_confidence
            if filter.min_relation_confidence is not None
            else filter.min_confidence
        )

        try:
            rows = await execute_query(
                "SELECT id, in, out, relation_type, properties, "
                "source_documents, extracted_at, extraction_method, "
                "confidence, provenance_chain, status, created_at "
                "FROM relation "
                "WHERE in INSIDE $entity_ids "
                "AND out INSIDE $entity_ids "
                "AND confidence >= $min_rel_conf",
                {
                    "entity_ids": entity_ids,
                    "min_rel_conf": min_rel_conf,
                },
                self.config,
            )
        except Exception as e:
            logger.error(
                f"list_relations_for_notebook failed for notebook "
                f"'{notebook_id}': {e}"
            )
            return []

        if not rows:
            return []

        relations: List[Relation] = []
        for row in rows:
            try:
                # ``in``/``out`` come back as RecordIDs (already string-
                # ified by execute_query); Relation has alias fields so
                # the raw row works.
                relations.append(Relation.model_validate(row))
            except Exception as e:
                logger.warning(
                    f"list_relations_for_notebook: skipping malformed "
                    f"row id={row.get('id')!r}: {e}"
                )
        return relations
