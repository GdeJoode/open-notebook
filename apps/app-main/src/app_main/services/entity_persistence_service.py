"""
Service for persisting filtered extraction results to the knowledge graph.

Takes a FilteredResult from the entity-filtering pipeline and upserts
entities into the ``entity`` table and relations into the ``relation``
table in SurrealDB, making them visible on the Knowledge Graph page.
"""

from typing import Any, Dict, List

from loguru import logger
from surrealdb_service.connection import execute_query


class ResolutionLogService:
    """Service for querying and updating the resolution_log table."""

    async def list_candidates(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        source_id: str | None = None,
        match_method: str | None = None,
        match_only: bool | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """List resolution log entries with filters.

        Returns:
            (items, total_count)
        """
        where_clauses = []
        params: dict[str, Any] = {"limit": limit, "offset": offset}

        if status:
            where_clauses.append("status = $status")
            params["status"] = status
        if source_id:
            where_clauses.append("source_id = $source_id")
            params["source_id"] = source_id
        if match_method:
            where_clauses.append("match_method = $match_method")
            params["match_method"] = match_method
        if match_only is not None:
            where_clauses.append("match = $match_only")
            params["match_only"] = match_only

        where = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        items_result = await execute_query(
            f"SELECT * FROM resolution_log{where} ORDER BY match_timestamp DESC LIMIT $limit START $offset",
            params,
        )
        count_result = await execute_query(
            f"SELECT count() AS total FROM resolution_log{where} GROUP ALL",
            params,
        )

        items = items_result[0] if items_result else []
        total = count_result[0][0].get("total", 0) if count_result and count_result[0] else 0

        return items, total

    async def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        """Get a single resolution log entry."""
        result = await execute_query(
            "SELECT * FROM $id",
            {"id": candidate_id},
        )
        if result and result[0]:
            return result[0][0]
        return None

    async def update_status(
        self, candidate_id: str, status: str
    ) -> dict[str, Any] | None:
        """Accept or reject a resolution log entry."""
        result = await execute_query(
            "UPDATE $id SET status = $status, reviewed_at = time::now() RETURN AFTER",
            {"id": candidate_id, "status": status},
        )
        if result and result[0]:
            return result[0][0]
        return None

    async def bulk_update_status(
        self, candidate_ids: list[str], status: str
    ) -> int:
        """Bulk accept/reject resolution log entries."""
        updated = 0
        for cid in candidate_ids:
            result = await self.update_status(cid, status)
            if result:
                updated += 1
        return updated

    async def get_methods(self) -> list[str]:
        """Get distinct match methods."""
        result = await execute_query(
            "SELECT match_method FROM resolution_log GROUP BY match_method",
            {},
        )
        if result and result[0]:
            return [r.get("match_method", "") for r in result[0]]
        return []

    async def get_stats(self) -> dict[str, Any]:
        """Get resolution log summary statistics."""
        result = await execute_query(
            """SELECT
                count() AS total,
                math::sum(IF match THEN 1 ELSE 0 END) AS matches,
                math::sum(IF status = 'pending' THEN 1 ELSE 0 END) AS pending,
                math::sum(IF status = 'accepted' THEN 1 ELSE 0 END) AS accepted,
                math::sum(IF status = 'rejected' THEN 1 ELSE 0 END) AS rejected,
                math::sum(IF status = 'auto_accepted' THEN 1 ELSE 0 END) AS auto_accepted
            FROM resolution_log GROUP ALL""",
            {},
        )
        if result and result[0]:
            return result[0][0]
        return {"total": 0, "matches": 0, "pending": 0, "accepted": 0, "rejected": 0, "auto_accepted": 0}


class EntityPersistenceService:
    """Persists filtered entities and relations to the KG tables."""

    async def persist_match_candidates(
        self,
        source_id: str,
        candidates: List[Dict[str, Any]],
    ) -> int:
        """Store match decisions in the resolution_log table.

        Args:
            source_id: Source record ID.
            candidates: List of MatchCandidate dicts.

        Returns:
            Number of candidates stored.
        """
        stored = 0
        for c in candidates:
            try:
                await execute_query(
                    """
                    CREATE resolution_log SET
                        entity_a_text = $entity_a_text,
                        entity_b_text = $entity_b_text,
                        entity_a_label = $entity_a_label,
                        entity_b_label = $entity_b_label,
                        match = $match,
                        confidence = $confidence,
                        match_method = $match_method,
                        match_reasoning = $match_reasoning,
                        iterations = $iterations,
                        source_document_a = $source_document_a,
                        source_document_b = $source_document_b,
                        source_section_a = $source_section_a,
                        source_section_b = $source_section_b,
                        matched_by_model = $matched_by_model,
                        status = $status,
                        source_id = $source_id,
                        match_timestamp = time::now()
                    """,
                    {
                        "entity_a_text": c.get("entity_a_text", ""),
                        "entity_b_text": c.get("entity_b_text", ""),
                        "entity_a_label": c.get("entity_a_label", "UNKNOWN"),
                        "entity_b_label": c.get("entity_b_label", "UNKNOWN"),
                        "match": c.get("match", False),
                        "confidence": c.get("confidence", 0.0),
                        "match_method": c.get("match_method", "unknown"),
                        "match_reasoning": c.get("match_reasoning", ""),
                        "iterations": c.get("iterations", 1),
                        "source_document_a": c.get("source_document_a"),
                        "source_document_b": c.get("source_document_b"),
                        "source_section_a": c.get("source_section_a"),
                        "source_section_b": c.get("source_section_b"),
                        "matched_by_model": c.get("matched_by_model"),
                        "status": c.get("status", "pending"),
                        "source_id": source_id,
                    },
                )
                stored += 1
            except Exception as e:
                logger.warning(f"Failed to store match candidate: {e}")
        logger.info(f"Stored {stored} match candidates for source {source_id}")
        return stored

    async def persist_filtered_result(
        self,
        source_id: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        merge_groups: List[List[str]] | None = None,
        match_candidates: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Upsert entities and create relations in the knowledge graph.

        Args:
            source_id: Source record ID (for provenance tracking).
            entities: Filtered entity dicts (text, label, confidence, properties).
            relations: Filtered relation dicts (source_entity, target_entity, relation_type, confidence).
            merge_groups: Optional dedup merge groups for provenance.

        Returns:
            Dict with ``entities_upserted`` and ``relations_created`` counts.
        """
        entities_upserted = 0
        relations_created = 0

        # Build merge group lookup: entity text → group members
        merge_lookup: Dict[str, List[str]] = {}
        if merge_groups:
            for group in merge_groups:
                for member in group:
                    merge_lookup[member] = group

        # 1. Upsert entities
        for entity in entities:
            text = entity.get("text", "")
            label = entity.get("label", "UNKNOWN")
            confidence = entity.get("confidence", 0.5)
            properties = entity.get("properties", {})

            if not text.strip():
                continue

            # Build entity properties for storage (exclude embedding to save space)
            stored_props = {
                k: v for k, v in properties.items()
                if k != "embedding" and v is not None
            }

            # Add merge history if available
            if text in merge_lookup:
                stored_props["merged_from"] = merge_lookup[text]

            try:
                # Upsert: find by name+type, update or create
                result = await execute_query(
                    """
                    LET $existing = (SELECT * FROM entity
                        WHERE name = $name AND entity_type = $entity_type
                        LIMIT 1);
                    IF array::len($existing) > 0 THEN {
                        UPDATE $existing[0].id SET
                            weight = weight + 1,
                            confidence = math::max(confidence, $confidence),
                            source_ids = array::union(source_ids, [$source_id]),
                            properties = object::extend(properties, $properties),
                            updated = time::now();
                        RETURN $existing[0].id;
                    } ELSE {
                        CREATE entity SET
                            name = $name,
                            entity_type = $entity_type,
                            weight = 1,
                            confidence = $confidence,
                            source_ids = [$source_id],
                            properties = $properties,
                            created = time::now(),
                            updated = time::now();
                    } END
                    """,
                    {
                        "name": text,
                        "entity_type": label,
                        "confidence": confidence,
                        "source_id": source_id,
                        "properties": stored_props,
                    },
                )
                entities_upserted += 1
            except Exception as e:
                logger.warning(f"Failed to upsert entity '{text}': {e}")

        # 2. Create relations
        for rel in relations:
            src_text = rel.get("source_entity", "")
            tgt_text = rel.get("target_entity", "")
            rel_type = rel.get("relation_type", "RELATED")
            confidence = rel.get("confidence", 0.5)
            properties = rel.get("properties", {})

            if not src_text or not tgt_text:
                continue

            try:
                await execute_query(
                    """
                    LET $src = (SELECT id FROM entity WHERE name = $src_name LIMIT 1);
                    LET $tgt = (SELECT id FROM entity WHERE name = $tgt_name LIMIT 1);
                    IF array::len($src) > 0 AND array::len($tgt) > 0 THEN {
                        RELATE $src[0].id->relation->$tgt[0].id SET
                            relation_type = $rel_type,
                            confidence = $confidence,
                            source_id = $source_id,
                            properties = $properties,
                            created = time::now();
                    } END
                    """,
                    {
                        "src_name": src_text,
                        "tgt_name": tgt_text,
                        "rel_type": rel_type,
                        "confidence": confidence,
                        "source_id": source_id,
                        "properties": properties,
                    },
                )
                relations_created += 1
            except Exception as e:
                logger.warning(
                    f"Failed to create relation {src_text} -> {tgt_text}: {e}"
                )

        # 3. Store match candidates in resolution_log (stap 2C)
        candidates_stored = 0
        if match_candidates:
            candidates_stored = await self.persist_match_candidates(
                source_id, match_candidates
            )

        logger.info(
            f"Persisted to KG: {entities_upserted} entities, "
            f"{relations_created} relations, "
            f"{candidates_stored} match candidates for source {source_id}"
        )

        return {
            "entities_upserted": entities_upserted,
            "relations_created": relations_created,
            "candidates_stored": candidates_stored,
        }
